# Monte Carlo Dropout (MCD) in Recurrent Neural Networks (GRU & LSTM): Detailed Analysis

## Overview

The GRU_MCD_42.ipynb and LSTM_MCD_42.ipynb implementations apply **Monte Carlo Dropout (MCD)** for uncertainty quantification in recurrent time-series forecasting. Both GRU (Gated Recurrent Unit) and LSTM (Long Short-Term Memory) networks follow identical MCD methodologies, differing only in their internal gating mechanisms:

- **GRU**: 2 gates (reset, update), simpler, fewer parameters
- **LSTM**: 3 gates (input, forget, output), more complex, more parameters

Despite architectural differences, both integrate MCD through the same Bayesian dropout mechanism, making them interchangeable for UQ purposes. This document explains the unified MCD implementation for both architectures.

---

## 1. WHERE MCD IS APPLIED IN RNN PIPELINES (GRU & LSTM)

### **Stage 1: RNN Architecture with Dropout (Lines ~59-82 in GRU, ~58-81 in LSTM)**
**File Location:** `build_gru_from_trial()` / `build_lstm_from_trial()` and `build_gru_fixed()` / `build_lstm_fixed()` functions

#### **GRU Architecture:**
```python
def build_gru_fixed(best_params: dict, lookback: int, n_features: int):
    m = Sequential()
    m.add(Input(shape=(lookback, n_features)))  # (batch, lookback, n_features)
    
    # Single or dual-layer GRU with dropout
    if best_params.get("num_layers", 1) == 2:
        # Layer 1: GRU with return_sequences=True (output all timesteps)
        m.add(GRU(best_params["units1"], return_sequences=True))
        # → Shape: (batch, lookback, units1)
        
        m.add(Dropout(best_params["dropout"]))  # ← DROPOUT LAYER 1 (after GRU 1)
        
        # Layer 2: GRU with return_sequences=False (output only last timestep)
        m.add(GRU(best_params["units2"]))
        # → Shape: (batch, units2)
    else:
        # Single GRU layer, return only last timestep
        m.add(GRU(best_params["units1"]))
        # → Shape: (batch, units1)
    
    # Final dropout after GRU stack
    m.add(Dropout(best_params["dropout"]))  # ← DROPOUT LAYER 2 (after GRU stack)
    
    # Dense output head
    m.add(Dense(1))  # → Shape: (batch, 1)
    
    m.compile(optimizer=optimizers.Adam(learning_rate=best_params["lr"]), loss="mse")
    return m
```

#### **LSTM Architecture (identical structure):**
```python
def build_lstm_fixed(best_params: dict, lookback: int, n_features: int):
    m = Sequential()
    m.add(Input(shape=(lookback, n_features)))
    
    if best_params.get("num_layers", 1) == 2:
        m.add(LSTM(best_params["units1"], return_sequences=True))
        m.add(Dropout(best_params["dropout"]))  # ← DROPOUT LAYER 1
        m.add(LSTM(best_params["units2"]))
    else:
        m.add(LSTM(best_params["units1"]))
    
    m.add(Dropout(best_params["dropout"]))  # ← DROPOUT LAYER 2
    m.add(Dense(1))
    
    m.compile(optimizer=optimizers.Adam(learning_rate=best_params["lr"]), loss="mse")
    return m
```

**Key Architecture Points:**
- **GRU/LSTM layers**: No explicit dropout within the RNN cells; Keras uses the built-in recurrent dropout
- **Dropout placement**: Applied AFTER GRU/LSTM layers (between stacked layers and after final layer)
- **Output head**: Simple Dense(1) for point predictions
- **Loss**: Mean Squared Error (MSE) - standard regression loss

---

### **Stage 2: Hyperparameter Optimization via Optuna (Lines ~130-180)**
**File Location:** Optuna objective function and optimization loop

```python
def make_objective(X_train_s, y_train_s, X_val_s, y_val_s, n_features):
    def objective(trial):
        # Sample hyperparameters
        lookback = trial.suggest_categorical("lookback", [30, 45, 60, 90])
        
        # RNN-specific parameters
        num_layers = trial.suggest_int("num_layers", 1, 2)  # 1 or 2 RNN layers
        units1 = trial.suggest_int("units1", 32, 256, step=32)  # Hidden units layer 1
        units2 = trial.suggest_int("units2", 32, 256, step=32) if num_layers == 2 else None
        
        # Dropout parameter (will be used for MCD)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)  # ← DROPOUT RATE FOR MCD
        
        # Training hyperparameters
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        epochs = trial.suggest_int("epochs", 30, 100, step=10)
        patience = trial.suggest_int("patience", 5, 10)
        
        # Create windows with this trial's lookback
        X_tr, y_tr, _ = make_windows(X_train_s, y_train_s, lookback)
        X_va, y_va, _ = make_windows(X_val_s,   y_val_s,   lookback)
        
        # Build and train model
        model = build_gru_from_trial(trial, lookback, n_features=n_features)
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
            TFKerasPruningCallback(trial, monitor="val_loss"),  # Early pruning of bad trials
        ]
        
        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=callbacks
        )
        
        return float(min(history.history["val_loss"]))
    
    return objective

# Execute Optuna study
print("[Optuna] Starting study...")
sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)  # Bayesian optimization
pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
objective = make_objective(X_train_s, y_train_s, X_val_s, y_val_s, n_features=len(feature_cols))

study.optimize(objective, n_trials=N_TRIALS)  # N_TRIALS = 50
best_params = study.best_params
```

**Key Points:**
- **Dropout rate** is optimized via Optuna (0-50% range)
- **Bayesian optimization** via TPE sampler (more efficient than random search)
- **Early pruning** stops unpromising trials
- **Best params** include the optimal dropout rate for MCD

---

### **Stage 3: Model Training (Lines ~200-225)**
**File Location:** Final training with best hyperparameters

```python
# Build model with best hyperparameters
final_model = build_gru_fixed(best_params, BEST_LOOKBACK, len(feature_cols))

callbacks = [
    EarlyStopping(monitor="val_loss", patience=best_params["patience"], 
                 restore_best_weights=True),
    ModelCheckpoint(CKPT_PATH, monitor="val_loss", save_best_only=True)
]

print("[Train] Retraining final GRU/LSTM with best params...")
t0_train = time.time()
history = final_model.fit(
    X_train_w, y_train_w,
    validation_data=(X_val_w, y_val_w),
    epochs=best_params["epochs"],
    batch_size=best_params["batch_size"],
    verbose=1,
    callbacks=callbacks
)
t1_train = time.time()
```

**During training:**
- Dropout is **active** by default (standard Keras behavior)
- Model learns temporal patterns with dropout as regularization
- Dropout creates implicit ensemble of sub-networks
- **No explicit uncertainty learning** - uncertainty emerges from ensemble variance

---

### **Stage 4: MCD Inference (Lines ~267-300)** ⭐ **CRITICAL MCD COMPONENT**
**File Location:** `@tf.function mc_call()` and `predict_mc()` functions

```python
@tf.function
def mc_call(m, X, training=True):
    """
    Force dropout to stay ACTIVE during inference
    Critical: training=True is the key to MCD
    """
    return m(X, training=training)

def predict_mc(model, X_np, idx, y_mean: float, y_scale: float,
               n_mc: int = 100, use_quantiles: bool = True, alpha: float = 0.10):
    """
    Monte Carlo Dropout forward passes for RNN
    
    model: trained GRU/LSTM model
    X_np: input data (N_samples, lookback, n_features)
    n_mc: number of MC passes (typically 100)
    """
    Ys_scaled = []
    X_tf = tf.convert_to_tensor(X_np, dtype=tf.float32)

    # ← MONTE CARLO LOOP: Run model N_MC times with dropout ACTIVE
    for _ in range(n_mc):
        # Forward pass through RNN with dropout ACTIVE
        # RNN processes all timesteps, dropout applied at selected points
        y_s = mc_call(model, X_tf, training=True).numpy().squeeze()  # (N,)
        Ys_scaled.append(y_s)

    # Stack all predictions: (N_samples, N_MC)
    Ys_scaled = np.stack(Ys_scaled, axis=1)   # (N, 100)
    
    # Inverse scale to original space
    Ys = Ys_scaled * y_scale + y_mean        # (N, 100)

    # Compute statistics from 100 predictions per sample
    mean = Ys.mean(axis=1)                   # Point estimate (ensemble mean)
    std  = Ys.std(axis=1, ddof=1)            # Uncertainty (ensemble std dev)

    # Prediction intervals from empirical quantiles or Gaussian
    if use_quantiles:
        lower = np.quantile(Ys, q=alpha/2, axis=1)           # e.g., 2.5th percentile
        upper = np.quantile(Ys, q=1 - alpha/2, axis=1)       # e.g., 97.5th percentile
    else:
        from scipy.stats import norm
        z = norm.ppf(1 - alpha/2.0)
        lower, upper = mean - z*std, mean + z*std

    return (
        pd.Series(mean,  index=idx, name="mean"),
        pd.Series(lower, index=idx, name=f"lower_{int((1-alpha)*100)}"),
        pd.Series(upper, index=idx, name=f"upper_{int((1-alpha)*100)}"),
        pd.Series(std,   index=idx, name="mc_std"),
        Ys  # Full ensemble (N, 100) for uncertainty decomposition
    )

# Execute MCD on all splits (lines 267-282)
print("[MC] Running Monte Carlo Dropout inference...")
t0_mcd = time.time()

# Load trained model and run MC inference
same_model = tf.keras.models.load_model(CKPT_PATH, compile=False)
same_model.compile(optimizer=optimizers.Adam(learning_rate=best_params["lr"]), loss="mse")

mean_train, L_train, U_train, std_train, Ys_train = predict_mc(
    same_model, X_train_w, idx_train, y_mean, y_scale, n_mc=N_MC, 
    use_quantiles=USE_QUANTILES, alpha=ALPHA
)
mean_val,   L_val,   U_val,   std_val,   Ys_val   = predict_mc(
    same_model, X_val_w,   idx_val,   y_mean, y_scale, n_mc=N_MC, 
    use_quantiles=USE_QUANTILES, alpha=ALPHA
)
mean_test,  L_test,  U_test,  std_test,  Ys_test  = predict_mc(
    same_model, X_test_w,  idx_test,  y_mean, y_scale, n_mc=N_MC, 
    use_quantiles=USE_QUANTILES, alpha=ALPHA
)

t1_mcd = time.time()
print(f"[MC] Done in {t1_mcd - t0_mcd:.4f}s")
```

**Execution Flow (Key for Understanding MCD in RNNs):**
```
For each of 100 MC passes:
  1. Input: X (batch, lookback, n_features)
  2. GRU/LSTM forward pass:
     - Process all timesteps sequentially
     - Maintain hidden state across time
     - Apply RNN gating mechanisms
     - Dropout active at specified layers (after GRU, after Dense)
  3. Output: ŷ (batch,) - single prediction per sample
  4. Store this stochastic prediction

Due to dropout masks differ at each pass:
  - Different RNN hidden states
  - Different dropout masks
  → 100 diverse predictions per sample
```

---

### **Stage 5: Uncertainty Quantification (Lines ~302-330)**
**File Location:** UQ metrics computation

```python
def uq_metrics(y_true, L, U, alpha=0.10) -> Dict[str, float]:
    """Uncertainty quantification metrics"""
    y = np.asarray(y_true); L = np.asarray(L); U = np.asarray(U)
    
    # PICP: Prediction Interval Coverage Probability
    cover = (y >= L) & (y <= U)
    picp = float(cover.mean())
    
    # MPIW: Mean Prediction Interval Width
    mpiw = float(np.mean(U - L))
    
    # Winkler Score: Width + Penalty for misses
    penalty = np.where(y < L, (2/alpha)*(L - y),
              np.where(y > U, (2/alpha)*(y - U), 0.0))
    winkler = float(np.mean((U - L) + penalty))
    
    return dict(PICP=picp, MPIW=mpiw, Winkler=winkler)

# Compute metrics (lines 302-310)
print(f"=== UQ Metrics ({int((1-ALPHA)*100)}% PI) ===")
print("Train:", uq_metrics(actual_train.values, L_train.values, U_train.values, ALPHA))
print("Val:  ", uq_metrics(actual_val.values,   L_val.values,   U_val.values,   ALPHA))
print("Test: ", uq_metrics(actual_test.values,  L_test.values,  U_test.values,  ALPHA))
```

---

### **Stage 6: Uncertainty Decomposition (Lines ~312-325)**
**File Location:** Epistemic vs Aleatoric estimation

```python
# Approximate aleatoric from validation residuals
resid_val = actual_val.values - mean_val.values
sigma2_aleatoric = float(np.var(resid_val, ddof=1))

# Total variance from 100 MC predictions
var_total_test = np.var(Ys_test, axis=1, ddof=1)

# Epistemic = Total - Aleatoric
var_epistemic = np.maximum(0.0, var_total_test - sigma2_aleatoric)
var_aleatoric = np.full_like(var_total_test, sigma2_aleatoric)
```

---

## 2. SUMMARY TABLE: WHERE MCD APPLIES IN RNN PIPELINES

| **Stage** | **Component** | **MCD Role** | **Code Location (GRU/LSTM)** |
|-----------|---------------|---|---|
| **Architecture** | GRU/LSTM + Dropout layers | Stochastic regularization | Lines ~59-82 |
| **Dropout Placement** | After RNN layer(s), after Dense | Multiple dropout points | Lines ~70, 75-76 |
| **Dropout Rate** | Optimized via Optuna | 0-50% range sampled | Lines ~65 (in trial) |
| **Training** | Standard fit() with dropout active | Dropout acts as regularizer | Lines ~200-225 |
| **HPO** | Optuna (Bayesian optimization) | Find best dropout rate | Lines ~130-180 |
| **Best Model Save** | ModelCheckpoint callback | Store trained weights | Line ~206 |
| **Inference** | `predict_mc()` function | **Force training=True** | Lines ~267-300 |
| **MC Loop** | For i=1 to 100 iterations | Generate ensemble | Lines ~280-285 |
| **Forward Pass** | `mc_call(model, X, training=True)` | Dropout ACTIVE | Lines ~281 |
| **Ensemble** | Stack 100 predictions | (N_samples, 100) shape | Lines ~287 |
| **Statistics** | mean, std, quantiles | From 100 predictions | Lines ~290-296 |
| **Decomposition** | Epistemic/Aleatoric variance | Uncertainty analysis | Lines ~312-325 |
| **Evaluation** | PICP, MPIW, Winkler | Measure UQ quality | Lines ~302-310 |

---

## 3. MCD FLOWCHART FOR GRU & LSTM

```
┌──────────────────────────────────────────────────────────────────────────┐
│  RECURRENT NEURAL NETWORKS (GRU & LSTM) + MONTE CARLO DROPOUT (MCD)       │
└──────────────────────────────────────────────────────────────────────────┘

                                  INPUT DATA
                                      │
                                      ▼
                    ┌──────────────────────────────┐
                    │  Train/Val/Test Windowing    │
                    │  (lookback = 30/45/60/90)    │
                    │  Shape: (N, lookback, nfeat) │
                    └──────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────┐
                    │  Optuna HPO (N=50 trials)    │
                    │  Sample hyperparameters:     │
                    │  - num_layers (1 or 2)       │
                    │  - units1, units2 (RNN dims) │
                    │  - dropout rate (0-50%)      │
                    │  - lr, batch_size, epochs    │
                    └──────────────────────────────┘
                                      │
                    ══════════════════════════════════════════════════════
                   ║  BUILD GRU/LSTM WITH DROPOUT                        ║
                    ══════════════════════════════════════════════════════
                                      │
                                      ▼
                    ┌───────────────────────────────────────────────────┐
                    │  RNN Architecture                                 │
                    │  ┌─────────────────────────────────────────────┐ │
                    │  │ INPUT (batch, lookback, n_features)        │ │
                    │  │           ↓                                 │ │
                    │  │ Layer 1: GRU/LSTM(units1)                  │ │
                    │  │ ├─ Process all timesteps                   │ │
                    │  │ ├─ Maintain hidden state h_t               │ │
                    │  │ │  (or cell state C_t for LSTM)            │ │
                    │  │ ├─ Gating mechanisms:                      │ │
                    │  │ │  GRU: reset gate, update gate            │ │
                    │  │ │  LSTM: input, forget, output gates       │ │
                    │  │ └─ Output: (batch, lookback, units1)       │ │
                    │  │           OR (batch, units1)               │ │
                    │  │           depending on return_sequences    │ │
                    │  │           ↓                                 │ │
                    │  │ Dropout(dropout_rate) ◄─ DROPOUT1          │ │
                    │  │           ↓                                 │ │
                    │  │ IF num_layers == 2:                        │ │
                    │  │ │  Layer 2: GRU/LSTM(units2)              │ │
                    │  │ │  ├─ Output: (batch, units2)             │ │
                    │  │ │  └─ (only last timestep output)         │ │
                    │  │ ELSE:                                      │ │
                    │  │ │  (Layer 1 already outputs last step)    │ │
                    │  │           ↓                                 │ │
                    │  │ Dropout(dropout_rate) ◄─ DROPOUT2          │ │
                    │  │           ↓                                 │ │
                    │  │ Dense(1)  → OUTPUT (batch, 1)              │ │
                    │  │                                             │ │
                    │  │ Loss: MSE                                   │ │
                    │  │ Optimizer: Adam (lr from Optuna)            │ │
                    │  └─────────────────────────────────────────────┘
                    └───────────────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────┐
                    │  TRAIN PHASE                 │
                    │  ├─ fit(X_train, y_train)   │
                    │  ├─ Dropout ACTIVE          │
                    │  ├─ EarlyStopping callback  │
                    │  └─ Validation monitoring   │
                    └──────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────┐
                    │  SAVE BEST MODEL             │
                    │  (lowest validation loss)    │
                    └──────────────────────────────┘
                                      │
                    ══════════════════════════════════════════════════════
                   ║  START MCD INFERENCE (NEW)                          ║
                    ══════════════════════════════════════════════════════
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  MONTE CARLO LOOP (N_MC=100)        │
                    │  for i in range(100):               │
                    │    ├─ Input X to RNN model          │
                    │    ├─ Pass through GRU/LSTM:        │
                    │    │  FOR each timestep t:          │
                    │    │   - Read input x_t              │
                    │    │   - Update hidden state h_t     │
                    │    │   (using gating mechanisms)     │
                    │    │   - Apply dropout to h_t ◄─╮   │
                    │    │     ACTIVE (training=True)      │
                    │    │   - Stochastic hidden state     │
                    │    ├─ Dropout(dropout_rate) ◄─╮     │
                    │    │   ACTIVE after RNN                │
                    │    ├─ Dense(1) → ŷ_i(x, θ_i)       │
                    │    │   (stochastic prediction)       │
                    │    └─ Store in ensemble             │
                    │                                      │
                    │  Due to dropout masks differ:       │
                    │  - Different RNN hidden states      │
                    │  - Different output values          │
                    │  → 100 diverse predictions          │
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

## 4. KEY MCD CONCEPTS FOR RNN (GRU & LSTM)

### **RNN vs Transformer Comparison**

```
TRANSFORMER:
- Processes all timesteps in parallel via attention
- Dropout applied at multiple points (attention, FF)
- 3+ dropout points per encoder block
- More dropout stochasticity

RNN (GRU/LSTM):
- Processes timesteps sequentially
- Hidden state carried through time
- Dropout applied after RNN layer and after Dense
- 2 dropout points per layer
- Dropout affects hidden state transitions
```

### **How Dropout Creates Diversity in RNNs**

```
Standard RNN Inference (training=False):
  h_0 = 0
  FOR t = 1 to T:
    h_t = RNN(x_t, h_{t-1})  ← deterministic
    out_t = Dense(h_t)        ← deterministic
  Return out_T

MC Dropout (training=True):
  h_0 = 0
  FOR t = 1 to T:
    h_t = RNN(x_t, h_{t-1})
    h_t = Dropout(h_t)        ← STOCHASTIC: random mask
    out_t = Dense(h_t)
    out_t = Dropout(out_t)    ← STOCHASTIC: random mask
  Return out_T

Due to:
  - Different dropout masks at each pass
  - Different hidden states h_t (affected by mask)
  - Different final outputs out_T
  → 100 diverse predictions
```

### **Dropout Stochasticity in Time-Series Context**

For time-series prediction, dropout creates uncertainty that captures:
1. **Model uncertainty** (epistemic): Different predictions due to weight uncertainty
2. **Data uncertainty** (aleatoric): Inherent randomness in the target

The key is that **RNNs maintain state across time**, so dropout at each timestep creates a cascade of different hidden states, leading to diverse final predictions.

---

## 5. GRU vs LSTM: Architectural Differences

| **Component** | **GRU** | **LSTM** |
|---|---|---|
| **Gates** | 2 (reset r_t, update z_t) | 3 (input i_t, forget f_t, output o_t) |
| **Hidden State** | h_t | h_t (hidden) + C_t (cell state) |
| **State Equation** | $h_t = (1-z_t)h_{t-1} + z_t\tilde{h}_t$ | $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ |
| **Complexity** | Simpler (fewer params) | More complex (more params) |
| **Computation** | Faster training | Slower training |
| **Performance** | Often comparable to LSTM | Often comparable to GRU |
| **MCD Integration** | Identical | Identical |

**For MCD purposes, both are equivalent** - the uncertainty quantification mechanism is independent of RNN type.

---

## 6. KEY ADVANTAGES OF MCD FOR RNNs

| **Advantage** | **Explanation** |
|---|---|
| **Practical Bayesian Approximation** | Dropout ≈ samples from posterior over weights |
| **Interpretable Uncertainty** | Ensemble variance ≈ model uncertainty |
| **No Architectural Changes** | Trains normally, uncertainty at inference |
| **Scalable** | Works for any RNN size |
| **Time-Series Ready** | Handles sequential dependencies naturally |

---

## 7. KEY CHALLENGES OF MCD FOR RNNs

| **Challenge** | **Explanation** |
|---|---|
| **Computational Cost** | 100 forward passes per prediction (expensive for inference) |
| **Recurrent Error Propagation** | Dropout errors compound across timesteps |
| **Calibration Issues** | Empirical quantiles may not match theoretical coverage |
| **Dropout Interaction with RNN** | Dropout affects hidden state dynamics; may need careful tuning |
| **Hyperparameter Sensitivity** | MCD results sensitive to dropout rate; requires Optuna tuning |

---

## Summary

**MCD in GRU/LSTM applies through:**

1. **Dropout in RNN Architecture** (2 points per layer) - Stochastic regularization
2. **Optuna HPO** (50 trials) - Find optimal dropout rate
3. **Training** - Dropout active (standard Keras behavior)
4. **Inference** - `training=True` forces dropout to stay active
5. **100 Forward Passes** - Generate diverse predictions
6. **Ensemble Statistics** - Mean, std, quantiles from 100 samples
7. **Uncertainty Decomposition** - Epistemic vs aleatoric via variance subtraction

The RNN architecture (GRU vs LSTM) is **irrelevant to MCD implementation** - both follow identical UQ procedures. The choice between GRU and LSTM should be based on modeling performance, not on UQ methodology. MCD provides Bayesian approximation through dropout, capturing predictive uncertainty through ensemble diversity.
