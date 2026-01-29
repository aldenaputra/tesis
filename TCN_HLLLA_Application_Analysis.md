# Heteroscedastic NLL + Last-Layer Laplace Approximation (HLLLA) in TCN Pipeline: Detailed Analysis

## Overview
The TCN_HLLLA_42.ipynb implements **Heteroscedastic NLL (Negative Log Likelihood) with Last-Layer Laplace Approximation (LLLA)** for uncertainty quantification in time-series forecasting. This method combines:
1. **Heteroscedastic prediction**: Model learns both mean (μ) and variance (σ²) of the prediction
2. **Last-Layer Laplace Approximation**: Bayesian approximation on the final layer to estimate epistemic uncertainty

---

## 1. WHERE HLLLA IS APPLIED IN THE TCN PIPELINE

### **Stage 1: Loss Function Definition (Lines ~112-125)**
**File Location:** `nll_gaussian_heteroscedastic()` function

```python
def nll_gaussian_heteroscedastic(y_true, y_pred):
    """
    Heteroscedastic Gaussian NLL Loss
    
    y_true: (batch,) or (batch,1)
    y_pred: (batch,2) = [mu_s, log_var_s]
                 ↑
        Model outputs TWO channels:
        - mu_s: predicted mean (scaled)
        - log_var_s: log of variance (learned heteroscedastic uncertainty)
    """
    y_true = tf.cast(tf.reshape(y_true, (-1,)), tf.float32)
    mu      = y_pred[:, 0]                          # Mean prediction
    log_var = tf.clip_by_value(y_pred[:, 1], -20.0, 5.0)  # Log-variance (clipped for stability)
    
    inv_var = tf.exp(-log_var)                      # Inverse variance
    
    # NLL = 0.5 * (log_var + (y - mu)²/var)
    nll = 0.5 * (log_var + (y_true - mu)**2 * inv_var)
    return tf.reduce_mean(nll)
```

**Key Point:** The loss function forces the model to **learn to predict uncertainty** (σ²) alongside the mean (μ).

---

### **Stage 2: TCN Architecture with Heteroscedastic Head (Lines ~126-152)**
**File Location:** `build_tcn_model()` function

```python
def build_tcn_model(lookback, n_features, filters, kernel_size, dropout, 
                    dilations, num_stacks, lr, use_layernorm=True):
    inp = Input(shape=(lookback, n_features))
    x = inp
    
    # Standard TCN blocks
    for _ in range(num_stacks):
        for d in dilations:
            x = tcn_block(x, filters, kernel_size, d, dropout, use_layernorm)
                      # └─ Residual conv blocks with SpatialDropout1D

    # Compress to single output
    x = Conv1D(1, 1, padding="same")(x)
    x = Lambda(lambda t: t[:, -1, :])(x)      # Take last timestep: (batch, 1)

    # ⭐ HETEROSCEDASTIC HEAD: Output 2 channels ⭐
    # Instead of Dense(1), output Dense(2) = [μ_s, log_σ²_s]
    out = Dense(2)(x)                         # ← OUTPUTS 2 VALUES

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=nll_gaussian_heteroscedastic     # ← HETEROSCEDASTIC LOSS
    )
    return model
```

**Architecture Flow:**
```
Input → TCN Blocks → Conv1D(1,1) → Lambda(last_step) → Dense(2)
                                                          ├─ [0]: μ_s (mean)
                                                          └─ [1]: log_σ²_s (log-variance)
```

---

### **Stage 3: Training with Heteroscedastic Loss (Lines ~194-215)**
**File Location:** Final model training

```python
final_model = build_tcn_model(...)

history = final_model.fit(
    X_train_w, y_train_w,
    validation_data=(X_val_w, y_val_w),
    epochs=best["epochs"],
    batch_size=best["batch_size"],
    verbose=1,
    callbacks=[EarlyStopping(...), ModelCheckpoint(...)]
)
```

**During training:**
- Model learns to predict (μ, log σ²)
- Loss function `nll_gaussian_heteroscedastic` encourages calibrated uncertainty estimates
- This captures **aleatoric uncertainty** (data-dependent noise)

---

### **Stage 4: Last-Layer Laplace Approximation (Lines ~216-244)** ⭐ **CRITICAL HLLLA COMPONENT**
**File Location:** LLLA extraction and epistemic uncertainty computation

#### **4a: Extract Penultimate Layer Features (Lines ~216-229)**
```python
# Last layer: Dense(2) → penultimate layer: Lambda (last timestep)
penultimate_layer = final_model.layers[-2]

# Create feature extractor φ(x) = output of penultimate layer
phi_model = tf.keras.Model(
    inputs=final_model.inputs[0],
    outputs=penultimate_layer.output    # ← Features before Dense(2)
)

# Extract features for each split
phi_train = phi_model.predict(X_train_w, verbose=0)  # (N_train, H)
phi_val   = phi_model.predict(X_val_w,   verbose=0)  # (N_val,   H)
phi_test  = phi_model.predict(X_test_w,  verbose=0)  # (N_test,  H)

# Typical shape: (N_samples, 1) since we did Conv1D(1,1)
```

**Key Point:** φ(x) = penultimate features represent the "representation" before the final Dense layer.

---

#### **4b: Estimate Noise Variance from Training Residuals (Lines ~230-236)**
```python
# Get predictions in scaled space
yhat_train_s = final_model.predict(X_train_w, verbose=0)  # (N_train, 2)
mu_train_s   = yhat_train_s[:, 0]                         # Extract μ
y_true_train_s = y_train_w                                # True labels (scaled)

# Residuals = observed - predicted
residuals_train_s = y_true_train_s - mu_train_s

# Estimate noise variance from training residuals
sigma_n2 = np.mean(residuals_train_s**2)   # ← Aleatoric variance estimate
```

**Interpretation:** This estimates the irreducible noise in the training data.

---

#### **4c: Compute Last-Layer Hessian Approximation (Lines ~237-250)**
```python
# Build extended features Φ = [φ(x), 1] for bias term
N_train, H = phi_train.shape
Phi_train = np.concatenate(
    [phi_train, np.ones((N_train, 1), dtype=phi_train.dtype)],
    axis=1
)  # (N_train, H+1)

# Prior precision (regularization)
lambda_prior = 1.0

# Diagonal Hessian approximation (simplified version of Laplace)
# H_diag ≈ (1/σ_n²) * Σ(Φ^T Φ) + λ_prior
H_diag = (1.0 / sigma_n2) * np.sum(Phi_train**2, axis=0) + lambda_prior  # (H+1,)

# Posterior variance = inverse Hessian diagonal
var_w_diag = 1.0 / H_diag  # (H+1,)
```

**Mathematical Insight:**
```
Laplace Approximation:
  - Approximate posterior at weights w* as Gaussian
  - Covariance ≈ inverse Hessian: P(w|D) ≈ N(w*, H^-1)
  - H = (1/σ_n²) * Σ(∇ℓ_i)² + λI (Fisher/empirical Hessian)
  - We use diagonal approximation for computational efficiency
```

---

#### **4d: Compute Epistemic Uncertainty (Lines ~251-266)**
```python
def compute_sigma_epi(phi_block, idx_block):
    """
    Epistemic variance from Laplace approximation on last layer.
    
    Var_epi(ŷ|x) ≈ Φ(x)^T Cov_w Φ(x)
                  = Φ(x)^T H^-1 Φ(x)
                  ≈ Σ_i (Φ_i(x)^2 * var_w_i)
    """
    N, H = phi_block.shape
    Phi_block = np.concatenate(
        [phi_block, np.ones((N, 1), dtype=phi_block.dtype)],
        axis=1
    )  # (N, H+1)

    # Epistemic variance in scaled space
    var_epi_scaled = np.sum((Phi_block**2) * var_w_diag.reshape(1, -1), axis=1)
    
    # Transform back to original scale
    y_std = y_scaler.scale_[0]
    var_epi_orig   = var_epi_scaled * (y_std**2)
    sigma_epi_orig = np.sqrt(np.maximum(var_epi_orig, 1e-12))

    return pd.Series(sigma_epi_orig, index=idx_block, name="sigma_epi")

# Apply to all splits
sigma_epi_train = compute_sigma_epi(phi_train, idx_train)
sigma_epi_val   = compute_sigma_epi(phi_val,   idx_val)
sigma_epi_test  = compute_sigma_epi(phi_test,  idx_test)
```

**Key Point:** This quantifies **epistemic uncertainty** (model's ignorance about parameters).

---

### **Stage 5: Prediction with Uncertainty Decomposition (Lines ~267-285)**
**File Location:** `predict_series_hetero()` function

```python
def predict_series_hetero(model, X_block, idx_block):
    """
    Returns:
      - mu: predictive mean (original scale)
      - sigma_ale: aleatoric std (from learned heteroscedastic head)
    """
    yhat_s = model.predict(X_block, verbose=0)   # (N, 2) = [μ_s, log_σ²_s]
    mu_s      = yhat_s[:, 0]
    log_var_s = yhat_s[:, 1]

    y_mean = y_scaler.mean_[0]
    y_std  = y_scaler.scale_[0]

    # Transform mean to original scale
    mu_orig = mu_s * y_std + y_mean

    # Transform variance to original scale
    var_orig = np.exp(log_var_s) * (y_std ** 2)
    sigma_ale = np.sqrt(var_orig)  # ← ALEATORIC UNCERTAINTY

    return pd.DataFrame({
        "mu": mu_orig,
        "sigma_ale": sigma_ale
    }, index=idx_block)

# Get predictions
pred_train = predict_series_hetero(final_model, X_train_w, idx_train)
pred_val   = predict_series_hetero(final_model, X_val_w,   idx_val)
pred_test  = predict_series_hetero(final_model, X_test_w,  idx_test)
```

---

### **Stage 6: Uncertainty Decomposition (Lines ~267-285)**
**File Location:** Combine aleatoric + epistemic uncertainties

```python
# Aleatoric (from model head)
sigma_ale_train = pred_train["sigma_ale"]
sigma_ale_test  = pred_test["sigma_ale"]

# Epistemic (from LLLA)
sigma_epi_train = pred_train["sigma_epi"]
sigma_epi_test  = pred_test["sigma_epi"]

# Total uncertainty = sqrt(σ_ale² + σ_epi²)
sigma_total_train = np.sqrt(sigma_ale_train**2 + sigma_epi_train**2)
sigma_total_test  = np.sqrt(sigma_ale_test**2  + sigma_epi_test**2)

# Store in DataFrame
pred_train["sigma_epi"]   = sigma_epi_train
pred_test["sigma_epi"]    = sigma_epi_test
pred_train["sigma_total"] = sigma_total_train
pred_test["sigma_total"]  = sigma_total_test
```

**Decomposition Formula:**
```
σ_total² = σ_ale² + σ_epi²

Where:
  σ_ale = aleatoric (data noise), learned via heteroscedastic head
  σ_epi = epistemic (model uncertainty), from LLLA posterior over last layer
  σ_total = combined predictive uncertainty
```

---

### **Stage 7: UQ Metrics Computation (Lines ~298-334)**
**File Location:** `compute_pi_metrics_from_sigma()` function

```python
def compute_pi_metrics_from_sigma(y_true, mu, sigma, z_level=1.96, alpha=0.05):
    """
    Prediction Intervals using Gaussian assumption:
    L = μ - z*σ_total
    U = μ + z*σ_total
    
    Where σ_total includes both aleatoric + epistemic components
    """
    L = mu - z_level * sigma     # Lower bound (2.5th percentile)
    U = mu + z_level * sigma     # Upper bound (97.5th percentile)
    
    # PICP: Prediction Interval Coverage Probability
    inside = (y_true >= L) & (y_true <= U)
    picp = inside.mean()
    
    # MPIW: Mean Prediction Interval Width
    width = U - L
    mpiw = width.mean()
    
    # Winkler Score: width + penalty for miscoverage
    penalties = np.zeros_like(y_true)
    below = y_true < L
    above = y_true > U
    penalties[below] = (L[below] - y_true[below])
    penalties[above] = (y_true[above] - U[above])
    winkler = np.mean(width + (2.0 / alpha) * penalties)
    
    return picp, mpiw, winkler

# Compute for test set
picp_test, mpiw_test, winkler_test = compute_pi_metrics_from_sigma(
    actual_test.values,
    pred_test["mu"].values,
    pred_test["sigma_total"].values,  # ← Uses COMBINED uncertainty
    z_level=Z_LEVEL,
    alpha=ALPHA
)
```

---

## 2. SUMMARY TABLE: WHERE HLLLA APPLIES

| **Stage** | **Component** | **HLLLA Role** | **Code Location** |
|-----------|---------------|---|---|
| **Loss Function** | `nll_gaussian_heteroscedastic` | Learn both μ and σ² | Lines ~112-125 |
| **Architecture** | Heteroscedastic head `Dense(2)` | Output (μ, log σ²) | Lines ~146-147 |
| **Training** | Model fitting | Optimize mean + uncertainty | Lines ~194-215 |
| **Feature Extraction** | `phi_model` (penultimate layer) | Extract φ(x) before Dense | Lines ~216-229 |
| **Noise Estimation** | Training residuals | Estimate σ_n² (aleatoric) | Lines ~230-236 |
| **Hessian Approx** | Diagonal Fisher/empirical Hessian | Build H_diag for posterior | Lines ~237-250 |
| **Epistemic UQ** | `compute_sigma_epi()` | Calculate σ_epi from Laplace | Lines ~251-266 |
| **Predictions** | `predict_series_hetero()` | Extract (μ, σ_ale) | Lines ~267-285 |
| **Decomposition** | σ_total = √(σ_ale² + σ_epi²) | Combine uncertainties | Lines ~267-285 |
| **Intervals** | Gaussian-based (μ ± z*σ_total) | Generate prediction intervals | Lines ~298-334 |
| **Evaluation** | PICP, MPIW, Winkler | Measure UQ quality | Lines ~298-334 |

---

## 3. HLLLA FLOWCHART

```
┌────────────────────────────────────────────────────────────────────────────┐
│      TCN + HETEROSCEDASTIC NLL + LAST-LAYER LAPLACE APPROXIMATION          │
│                           (TCN HLLLA)                                       │
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
        ┌───────────────────────────────────────────────────┐
        │  Build TCN with Heteroscedastic Head             │
        │  ┌─────────────────────────────────────────────┐ │
        │  │  Conv1D blocks                              │ │
        │  ├─ SpatialDropout1D (regularization)         │ │
        │  ├─ LayerNorm, Residual connections          │ │
        │  │                                             │ │
        │  ├─ Conv1D(1,1)  → take last step             │ │
        │  │                                             │ │
        │  └─ Dense(2)  ◄──── HETEROSCEDASTIC HEAD     │ │
        │      ├─ [0]: μ_s (mean)                      │ │
        │      └─ [1]: log_σ²_s (log-variance)         │ │
        │                                               │ │
        │  Loss: nll_gaussian_heteroscedastic            │ │
        │  = 0.5 * (log_var + (y-μ)²/var)              │ │
        └───────────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────┐
                    │  TRAIN PHASE                 │
                    │  fit(X_train, y_train)       │
                    │  - Learns: μ AND σ²         │
                    │  - Captures aleatoric UQ    │
                    └──────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────┐
                    │  SAVE BEST WEIGHTS           │
                    └──────────────────────────────┘
                                      │
                    ══════════════════════════════════════
                   ║   START LAPLACE APPROXIMATION      ║
                    ══════════════════════════════════════
                                      │
                    ┌──────────────────────────────────────┐
                    │  1️⃣  EXTRACT PENULTIMATE FEATURES   │
                    │  ├─ Remove last layer: Dense(2)    │
                    │  ├─ phi_model = Model(input,        │
                    │  │              Lambda_output)      │
                    │  └─ φ(x) = features before Dense    │
                    │     Shape: (N_samples, H)           │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  2️⃣  ESTIMATE NOISE VARIANCE         │
                    │  ├─ Get predictions on TRAIN         │
                    │  │  yhat = model.predict(X_train)   │
                    │  ├─ residuals = y_true - mu         │
                    │  └─ σ_n² = mean(residuals²)         │
                    │     (Aleatoric noise estimate)      │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  3️⃣  BUILD EXTENDED FEATURES         │
                    │  ├─ Φ = [φ(x), 1]  (add bias term) │
                    │  └─ Shape: (N_train, H+1)          │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  4️⃣  COMPUTE DIAGONAL HESSIAN        │
                    │                                      │
                    │  H_diag = (1/σ_n²)*Σ(Φ²) + λ*I     │
                    │                                      │
                    │  H ≈ Fisher Information Matrix       │
                    │  λ = Prior precision (regularization)│
                    │  Shape: (H+1,)                       │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  5️⃣  POSTERIOR COVARIANCE            │
                    │                                      │
                    │  var_w_diag = 1 / H_diag            │
                    │                                      │
                    │  Represents: Cov(w|D) ≈ H^-1        │
                    │  Shape: (H+1,)                       │
                    └──────────────────────────────────────┘
                                      │
                    ══════════════════════════════════════
                   ║    EPISTEMIC UNCERTAINTY PROPAGATION  ║
                    ══════════════════════════════════════
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  6️⃣  COMPUTE σ_EPISTEMIC             │
                    │  for TEST set:                        │
                    │                                      │
                    │  σ_epi(x) = √(Φ(x)² · var_w)       │
                    │                                      │
                    │  Sum over all weights w_i            │
                    │  Reflects model's posterior           │
                    │  uncertainty about parameters        │
                    │  Shape: (N_test,)                    │
                    └──────────────────────────────────────┘
                                      │
                    ══════════════════════════════════════
                   ║        FINAL PREDICTIONS & UQ          ║
                    ══════════════════════════════════════
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  7️⃣  PREDICTIONS                      │
                    │  ├─ yhat = model.predict(X_test)    │
                    │  ├─ μ = yhat[:, 0]  (mean)          │
                    │  └─ log_var = yhat[:, 1]            │
                    │     σ_ale = √exp(log_var)           │
                    │           (Aleatoric from head)     │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  8️⃣  UNCERTAINTY DECOMPOSITION       │
                    │                                      │
                    │  σ_ale = learned by heteroscedastic │
                    │          head (data noise)          │
                    │                                      │
                    │  σ_epi = from Laplace posterior      │
                    │          (model uncertainty)         │
                    │                                      │
                    │  σ_total = √(σ_ale² + σ_epi²)       │
                    │           (predictive uncertainty)   │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  9️⃣  PREDICTION INTERVALS            │
                    │                                      │
                    │  L = μ - z * σ_total                │
                    │  U = μ + z * σ_total                │
                    │                                      │
                    │  z = 1.96 for 95% interval          │
                    │  Uses COMBINED uncertainty          │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  🔟  COMPUTE UQ METRICS               │
                    │  ├─ PICP: P(y ∈ [L,U])              │
                    │  ├─ MPIW: E[U - L]                  │
                    │  ├─ Winkler: width + penalty        │
                    │  └─ Coverage plots & heatmaps       │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  SAVE RESULTS                        │
                    │  ├─ ALL_UQ_PREDICTED.csv            │
                    │  │  (μ, L, U, actual)               │
                    │  └─ ALL_UQ_METRICS.csv              │
                    │     (PICP, MPIW, Winkler, etc.)    │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                                  OUTPUT: 
                            Predictions + UQ Bounds
                            (Aleatoric + Epistemic)
                                  & Metrics
```

---

## 4. KEY HLLLA CONCEPTS IN YOUR IMPLEMENTATION

### **The Core Insight**

**Traditional Point Prediction:**
```
Model → single output → single prediction
Loss: MSE (mean squared error)
```

**Heteroscedastic Prediction:**
```
Model → TWO outputs (μ, log σ²) → mean + learned uncertainty
Loss: NLL (negative log likelihood)
∼ Captures aleatoric uncertainty (data noise)
```

**HLLLA (Heteroscedastic + Laplace):**
```
Model → TWO outputs (μ, log σ²)
           ↓
    Aleatoric uncertainty (σ_ale from log σ²)
           +
    Extract features before final Dense layer
           ↓
    Apply Laplace Approximation to last layer weights
           ↓
    Epistemic uncertainty (σ_epi from posterior)
           ↓
    σ_total = √(σ_ale² + σ_epi²)
```

---

### **Why Last-Layer Laplace Approximation?**

**Standard Bayesian Approach (Expensive):**
- Approximate full posterior P(w|D) over ALL weights
- Requires expensive computation (full Hessian)

**Last-Layer Laplace (Efficient):**
- Approximate posterior ONLY over last layer weights
- Keep earlier layers fixed at trained values
- Reduces computation from O(n_params²) to O(n_last_layer²)
- Still captures meaningful epistemic uncertainty

**Formula:**
```
P(w|D) ≈ N(w*, H^-1)

Where:
  w* = trained weights (MAP estimate)
  H = Hessian of loss at w*
  H ≈ (1/σ_n²) * Σ(∇ℓ_i)² + λI  (Fisher/empirical Hessian)

For prediction:
  σ_epi(x) = √(φ(x)^T H^-1 φ(x))
           ≈ √(Σ_i φ_i(x)² * var_w_i)
```

---

### **Aleatoric vs Epistemic in HLLLA**

| **Type** | **Source** | **Interpretation** |
|----------|-----------|-------------------|
| **Aleatoric (σ_ale)** | Model head Dense(2), learning log_var | Data noise, irreducible uncertainty |
| **Epistemic (σ_epi)** | LLLA posterior over last layer weights | Model's uncertainty about its parameters |
| **Total (σ_total)** | √(σ_ale² + σ_epi²) | Complete predictive uncertainty |

---

## 5. CONFIGURATION PARAMETERS FOR HLLLA

| **Parameter** | **Default** | **Purpose** |
|---------------|-------------|-----------|
| `Z_LEVEL` | 1.96 | z-score for 95% Gaussian interval |
| `ALPHA` | 0.05 | Significance level (95% coverage target) |
| `lambda_prior` | 1.0 | Prior precision for regularization |
| `log_var_clip` | [-20.0, 5.0] | Stability bounds for log-variance |

---

## 6. MATHEMATICAL FOUNDATION

### **Heteroscedastic NLL Loss**
```
L(θ) = Σ_i [0.5 * log(σ_i²) + 0.5 * (y_i - μ_i)² / σ_i²]

This loss function:
  1. Penalizes high variance (log(σ²) term)
  2. Scales residual error by predicted variance (adaptive weighting)
  3. Encourages the model to express uncertainty about hard examples
```

### **Laplace Approximation at Last Layer**
```
Posterior variance contribution:
  var(ŷ|x) = Var_aleatoric + Var_epistemic
            = σ_ale² + φ(x)^T Cov(w) φ(x)

Where Cov(w) ≈ H^-1 (posterior covariance)

Diagonal approximation:
  var_epistemic ≈ Σ_i φ_i(x)² * [H^-1]_{ii}
```

---

## 7. OUTPUT INTERPRETATION

### **Example Metrics Summary:**
```
Split    MSE     MAE     RMSE    MAPE    R²      PICP    MPIW    Winkler
────────────────────────────────────────────────────────────────────────
Train    0.0047  0.0531  0.0685  0.0159  0.9815  0.9520  0.2287  0.3001
Val      0.0054  0.0597  0.0735  0.0178  0.9769  0.9410  0.2450  0.3215
Test     0.0067  0.0712  0.0818  0.0212  0.9685  0.9524  0.2789  0.3589
```

**Interpretation:**
- **PICP ~0.95** → 95% of actuals fall within predicted intervals
- **MPIW** → Interval width (smaller is better if coverage maintained)
- **Winkler** → Penalizes both width and miscoverage

### **Uncertainty Decomposition Plot:**
```
At any time t:
  ├─ σ_ale(t) = learned data noise (from heteroscedastic head)
  ├─ σ_epi(t) = model parameter uncertainty (from LLLA)
  └─ σ_total(t) = √(σ_ale² + σ_epi²) ← used for intervals
```

---

## 8. COMPARISON: MCD vs HLLLA

| **Aspect** | **MCD** | **HLLLA** |
|-----------|---------|----------|
| **Point Pred** | MC mean of 100 passes | Deterministic μ from Dense(2) |
| **Uncertainty Source** | Dropout stochasticity | Model outputs + LLLA |
| **Aleatoric Est.** | Implicit (MC variance) | Explicit (learned σ² head) |
| **Epistemic Est.** | From MC ensemble | From LLLA posterior |
| **Computation** | 100 forward passes | Single pass + LLLA (matrix ops) |
| **Speed** | Slower (100×) | Faster (1 pass + Hessian) |
| **Hyperparams** | N_MC, dropout rate | lambda_prior, log_var bounds |
| **Interpretability** | Black-box ensemble | Explicit uncertainty decomposition |

---

## Summary

**HLLLA applies in 6 key stages:**

1. **Architecture**: Model outputs (μ, log_var) via Dense(2)
2. **Loss Function**: Heteroscedastic NLL trains both mean and variance
3. **Feature Extraction**: Extract φ(x) from penultimate layer
4. **Laplace Approximation**: Build diagonal Hessian to get posterior covariance
5. **Epistemic UQ**: Compute σ_epi from Laplace posterior
6. **Uncertainty Decomposition**: Combine σ_ale (learned) + σ_epi (LLLA) = σ_total

The result is a **calibrated, interpretable uncertainty estimate** where:
- **Aleatoric uncertainty** captures data noise (what the model can't know)
- **Epistemic uncertainty** captures model uncertainty (what the model could learn with more data)
- **Total uncertainty** enables reliable prediction intervals

This is computationally efficient (unlike full Bayesian inference) while still providing meaningful uncertainty decomposition.
