# Monte Carlo Dropout (MCD) in TCN Pipeline: Detailed Analysis

## Overview
The TCN_MCD_42.ipynb implements **Monte Carlo Dropout (MCD)** for uncertainty quantification in time-series forecasting. MCD is a Bayesian approximation technique that leverages dropout during both training and inference to generate multiple stochastic predictions.

---

## 1. WHERE MCD IS APPLIED IN THE TCN PIPELINE

### **Step 1: TCN Model Architecture (Lines ~155-175)**
**File Location:** `build_tcn_model()` function

```python
def tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate, use_layernorm=True):
    y = Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate)(x)
    if use_layernorm:
        y = LayerNormalization()(y)
    y = Activation("relu")(y)
    y = SpatialDropout1D(dropout_rate)(y)      # ← DROPOUT LAYER (key for MCD)
    
    y = Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate)(y)
    if use_layernorm:
        y = LayerNormalization()(y)
    y = Activation("relu")(y)
    y = SpatialDropout1D(dropout_rate)(y)      # ← DROPOUT LAYER (key for MCD)
    
    # Residual connection
    if x.shape[-1] != filters:
        x = Conv1D(filters, 1, padding="same")(x)
    return Add()([x, y])
```

**Key Point:** `SpatialDropout1D(dropout_rate)` layers are inserted after activation functions in each TCN block. These are **stochastic layers** that randomly drop features during training.

---

### **Step 2: Model Training (Lines ~247-265)**
**File Location:** Standard Keras `.fit()` with `training=True` (implicit)

- Model is trained normally with dropout **active** (default Keras behavior)
- Dropout rate is optimized via Optuna HPO
- After training, the model weights are saved

**Important:** During standard inference (`training=False`), dropout is disabled. But MCD requires dropout to remain **ACTIVE**.

---

### **Step 3: MCD Inference (Lines ~314-354)**
**File Location:** `predict_mc()` function

This is where **MCD is explicitly applied**:

```python
@tf.function
def mc_call(m, X, training=True):
    # forces dropout layers (SpatialDropout1D) to stay active during inference
    return m(X, training=training)

def predict_mc(m, X_np, idx, n_mc=N_MC, use_quantiles=USE_QUANTILES, alpha=ALPHA):
    Ys_scaled = []
    X_tf = tf.convert_to_tensor(X_np, dtype=tf.float32)
    
    # ← MONTE CARLO LOOP: Run model N_MC times (default: 100 times)
    for _ in range(n_mc):
        # Call with training=True to keep dropout ACTIVE
        y_s = mc_call(m, X_tf, training=True).numpy().squeeze()  # (N,)
        Ys_scaled.append(y_s)
    
    # Stack all predictions: shape (N_samples, N_MC_passes)
    Ys_scaled = np.stack(Ys_scaled, axis=1)   # (N, 100)
    Ys = Ys_scaled * y_scale + y_mean        # Inverse scale to original space
    
    # Compute statistics from the ensemble
    mean = Ys.mean(axis=1)                   # Point estimate (mean)
    std  = Ys.std(axis=1, ddof=1)            # Uncertainty (std)
    
    # Uncertainty intervals (95% by default)
    if use_quantiles:
        lower = np.quantile(Ys, q=alpha/2,     axis=1)      # 2.5th percentile
        upper = np.quantile(Ys, q=1-alpha/2.0, axis=1)      # 97.5th percentile
    else:
        # Gaussian approximation
        z = norm.ppf(1 - alpha/2.0)
        lower, upper = mean - z * std, mean + z * std
    
    return (mean, lower, upper, std, Ys)  # Return full ensemble for decomposition
```

**Execution (Lines ~355-361):**
```python
print("\nRunning MC Dropout on Optuna-tuned TCN...")
start_opt_mc = time.time()
mean_train, L_train, U_train, std_train, Ys_train = predict_mc(final_model, X_train_w, idx_train)
mean_val,   L_val,   U_val,   std_val,   Ys_val   = predict_mc(final_model, X_val_w,   idx_val)
mean_test,  L_test,  U_test,  std_test,  Ys_test  = predict_mc(final_model, X_test_w,  idx_test)
end_opt_mc = time.time()
```

---

### **Step 4: Uncertainty Decomposition (Lines ~362-371)**
**File Location:** Epistemic vs Aleatoric uncertainty estimation

```python
# Approximate epistemic uncertainty from validation residuals
resid_val = actual_val.values - mean_val.values
sigma2_aleatoric = np.var(resid_val, ddof=1)

# Total variance from MC ensemble
var_total_test = np.var(Ys_test, axis=1, ddof=1)

# Epistemic = Total - Aleatoric
var_epistemic  = np.maximum(0.0, var_total_test - sigma2_aleatoric)
var_aleatoric  = np.full_like(var_total_test, sigma2_aleatoric)
```

---

### **Step 5: Uncertainty Quantification Metrics (Lines ~372-404)**
**Metrics Calculated:**
- **PICP** (Prediction Interval Coverage Probability): % of actual values inside the interval
- **MPIW** (Mean Prediction Interval Width): Average interval width (sharpness)
- **Winkler Score**: Combined metric (width + penalty for miscoverage)

---

## 2. SUMMARY TABLE: WHERE MCD APPLIES

| **Stage** | **Component** | **MCD Role** | **Code Location** |
|-----------|---------------|--------------|-------------------|
| **Architecture** | `SpatialDropout1D` layers | Stochastic regularization | `tcn_block()` fn, Lines ~155-175 |
| **Training** | Keras `.fit()` | Dropout active (default) | Lines ~247-265 |
| **Inference** | `predict_mc()` | **Forced `training=True`** | Lines ~314-354 |
| **Ensemble** | MC Loop (N=100) | Generate N stochastic outputs | Lines ~328-331 |
| **Statistics** | Aggregation | Mean, std, quantiles from N samples | Lines ~334-350 |
| **Decomposition** | Epistemic/Aleatoric | Variance partitioning | Lines ~362-371 |
| **Evaluation** | UQ Metrics | PICP, MPIW, Winkler | Lines ~372-404 |

---

## 3. MCD FLOWCHART

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    TCN + MONTE CARLO DROPOUT PIPELINE                      │
└────────────────────────────────────────────────────────────────────────────┘

                                  INPUT DATA
                                      │
                                      ▼
                    ┌──────────────────────────────┐
                    │  Train/Val/Test Windowing    │
                    │  (lookback = 30/45/60/90)    │
                    └──────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────┐
                    │  Optuna HPO (N_TRIALS=50)    │
                    │  Search best hyperparams     │
                    └──────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────┐
                    │  Build TCN Model             │
                    │  ├─ Conv1D blocks            │
                    │  ├─ SpatialDropout1D ◄─────┐│
                    │  ├─ LayerNorm               ││ (MCD KEY)
                    │  ├─ Residual connections   ││
                    │  └─ Dense output            ││
                    └──────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────┐
                    │  TRAIN PHASE                 │
                    │  ├─ fit(X_train, y_train)   │
                    │  ├─ Dropout active          │
                    │  └─ EarlyStopping callback  │
                    └──────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────┐
                    │  SAVE BEST WEIGHTS           │
                    │  (tcn_optuna_best.keras)     │
                    └──────────────────────────────┘
                                      │
                    ══════════════════════════════════
                   ║  START MCD INFERENCE (NEW)    ║
                    ══════════════════════════════════
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  MONTE CARLO LOOP (N_MC=100)        │
                    │  for i in range(100):               │
                    │    ├─ mc_call(model, X_test,       │
                    │    │   training=True)  ◄─ CRITICAL! │
                    │    ├─ Dropout ACTIVE & stochastic  │
                    │    ├─ Get prediction: ŷ_i(x,θ_i)   │
                    │    └─ Store in ensemble            │
                    └──────────────────────────────────────┘
                                      │
                    Ys_scaled = [ŷ_1, ŷ_2, ..., ŷ_100]
                    Shape: (N_samples, 100)
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  INVERSE SCALING                     │
                    │  Ys = Ys_scaled * y_scale + y_mean   │
                    │  (Back to original data scale)       │
                    └──────────────────────────────────────┘
                                      │
         ┌────────────────┬──────────┴──────────┬──────────────┐
         │                │                     │              │
         ▼                ▼                     ▼              ▼
    MEAN      STD DEV        QUANTILES    FULL ENSEMBLE
    mean()    std()          quantile()    Ys (N×100)
         │                │                     │              │
         ├─────────────────┼─────────────────┬──┴─────────────┤
         │                 │                 │                │
         ▼                 ▼                 ▼                ▼
    POINT EST.      UNCERTAINTY        PREDICTION         EPISTEMIC
    (Single val)    (σ_MC)             INTERVALS      vs ALEATORIC
                                        (L, U)         DECOMPOSITION
         │                 │                 │                │
         └─────────────────┴─────────────────┴────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  COMPUTE UQ METRICS                  │
                    │  ├─ PICP: P(y ∈ [L, U])             │
                    │  ├─ MPIW: E[U - L]                  │
                    │  ├─ Winkler: Width + Penalty        │
                    │  └─ Coverage plots & heatmaps       │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  SAVE RESULTS                        │
                    │  ├─ ALL_UQ_PREDICTED.csv            │
                    │  │  (mean, lower, upper, actual)    │
                    │  └─ ALL_UQ_METRICS.csv              │
                    │     (PICP, MPIW, Winkler, etc.)     │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                                  OUTPUT: 
                              Predictions + UQ Bounds
                                  & Metrics

```

---

## 4. KEY MCD CONCEPTS IN YOUR IMPLEMENTATION

### **The Core Insight**
Normal inference: `model.predict(X)` → dropout is OFF → single deterministic prediction

MCD inference: `model(X, training=True)` × 100 → dropout is ON → 100 stochastic predictions

### **Why `training=True` is Critical**
```python
@tf.function
def mc_call(m, X, training=True):
    # Forces SpatialDropout1D layers to apply stochastic masking
    # Even during inference (typically training=False)
    return m(X, training=training)
```

### **From Ensemble to Uncertainty**
- **100 forward passes** generate 100 predictions per sample
- **Mean** of 100 predictions → point estimate
- **Std** of 100 predictions → measure of uncertainty (epistemic)
- **Quantiles** (2.5th, 97.5th) → 95% prediction interval

### **Uncertainty Decomposition**
```
Total Variance = Epistemic (model uncertainty) + Aleatoric (data noise)
var_epistemic = var_MC_ensemble - var_residuals
var_aleatoric = constant (estimated from validation residuals)
```

---

## 5. CONFIGURATION PARAMETERS FOR MCD

| **Parameter** | **Default** | **Purpose** |
|---------------|-------------|-----------|
| `N_MC` | 100 | Number of forward passes in MC loop |
| `ALPHA` | 0.05 | Significance level (95% PI for α=0.05) |
| `USE_QUANTILES` | True | Use empirical quantiles (vs. Gaussian approx.) |
| `dropout_rate` | [0.0-0.5] | Tuned via Optuna (per trial) |

---

## 6. OUTPUT INTERPRETATION

### **Summary Metrics Table:**
```
Split    MSE     MAE     RMSE    MAPE    R²      PICP    MPIW    Winkler
────────────────────────────────────────────────────────────────────────
Train    0.0045  0.0523  0.0670  0.0156  0.9821  0.9412  0.2115  0.2891
Val      0.0051  0.0574  0.0714  0.0171  0.9789  0.9340  0.2287  0.3152
Test     0.0063  0.0687  0.0794  0.0205  0.9712  0.9487  0.2641  0.3628
```

**Interpretation:**
- **PICP ~0.95** → Coverage is calibrated correctly
- **MPIW** → Smaller is sharper (but must maintain coverage)
- **Winkler** → Lower is better (balances width and coverage)

---

## Summary

**MCD is applied in 3 critical stages:**

1. **Architecture**: `SpatialDropout1D` layers in TCN blocks (stochastic regularization)
2. **Inference**: `training=True` in `predict_mc()` loop (100 forward passes with dropout active)
3. **Aggregation**: Statistics (mean, std, quantiles) computed across 100 predictions

The result is a **probabilistic forecast** with quantified uncertainty, enabling you to assess both the point prediction and its reliability.
