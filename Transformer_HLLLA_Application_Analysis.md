# Heteroscedastic NLL + Last-Layer Laplace Approximation (HLLLA) in Transformer Pipeline: Detailed Analysis

## Overview
The Transformer_HLLLA_42.ipynb implements **Heteroscedastic Gaussian Likelihood + Last-Layer Laplace Approximation (HLLLA)** for uncertainty quantification in time-series forecasting. This method combines:
- **Training phase:** Learning both mean predictions (μ) and prediction variance (σ²_ale) via heteroscedastic loss
- **Post-training phase:** Approximating epistemic uncertainty through Laplace approximation on the last layer

This approach captures both **aleatoric uncertainty** (during training) and **epistemic uncertainty** (post-hoc), providing explicit uncertainty decomposition.

---

## 1. WHERE HLLLA IS APPLIED IN THE TRANSFORMER PIPELINE

### **Stage 1: Heteroscedastic Loss Function Definition (Lines ~112-125)**
**File Location:** `nll_gaussian_heteroscedastic()` function

```python
def nll_gaussian_heteroscedastic(y_true, y_pred):
    """
    Heteroscedastic Gaussian Negative Log-Likelihood Loss
    
    y_true: actual target values (scaled)
    y_pred: (mu_s, log_var_s) - both in scaled space
    
    Formula:
    NLL = 0.5 * [log(σ²) + (y - μ)² / σ²]
    
    Rearranged using log_var for numerical stability:
    NLL = 0.5 * [log_var + (y - μ)² * exp(-log_var)]
    """
    y_true = tf.cast(tf.reshape(y_true, (-1,)), tf.float32)  # Flatten to (N,)
    
    mu      = y_pred[:, 0]                                    # Mean (batch,)
    log_var = tf.clip_by_value(y_pred[:, 1], -20.0, 5.0)     # Log-variance clipped for stability
    
    # Compute 1/σ² = exp(-log_var)
    inv_var = tf.exp(-log_var)
    
    # Negative log-likelihood
    nll = 0.5 * (log_var + (y_true - mu)**2 * inv_var)
    
    return tf.reduce_mean(nll)  # Average over batch
```

**Key Insights:**
- Output has **2 components** per prediction: `[μ_s, log_σ²_s]`
- Log-variance is clipped to `[-20, 5]` for numerical stability (prevents extreme values)
- Loss encourages the model to:
  - Minimize prediction error: `(y - μ)²`
  - Learn appropriate uncertainty: `σ²` (higher when uncertain)
  - This is **aleatoric uncertainty** (data-dependent)

---

### **Stage 2: Transformer Architecture with Heteroscedastic Output (Lines ~145-180)**
**File Location:** `build_transformer()` function

```python
def build_transformer(input_shape, params):
    """
    Transformer with heteroscedastic output head
    """
    lb        = params["lookback"]
    d_model   = params["d_model"]
    num_heads = params["num_heads"]
    dff       = params["dff"]
    dropout   = params["dropout"]

    inp = Input(shape=input_shape)
    
    # Input projection
    x = Dense(d_model)(inp)
    
    # Positional encoding (sequence position awareness)
    x = AddPE(lb, d_model)(x)
    
    # Encoder blocks (num_layers stacked)
    for _ in range(params["num_layers"]):
        x = encoder_block(
            x,
            num_heads=num_heads,
            d_model=d_model,
            dff=dff,
            dropout_rate=dropout
        )
    
    # Extract last timestep: (batch, lookback, d_model) → (batch, d_model)
    x = Lambda(lambda t: t[:, -1, :])(x)
    
    # ═══════════════════════════════════════════════════════
    # CRITICAL: Heteroscedastic output head
    # ═══════════════════════════════════════════════════════
    # Instead of Dense(1), use Dense(2):
    #   out[:, 0] = μ_s (predicted mean in scaled space)
    #   out[:, 1] = log_σ²_s (predicted log-variance in scaled space)
    # ═══════════════════════════════════════════════════════
    out = Dense(2)(x)  # ← OUTPUT TWO VALUES per prediction
    
    model = Model(inputs=inp, outputs=out)
    
    # Compile with heteroscedastic loss
    opt = Adam(learning_rate=params["lr"])
    model.compile(optimizer=opt, loss=nll_gaussian_heteroscedastic)
    
    return model
```

**Architecture Comparison:**

| **Component** | **MCD (Dense(1))** | **HLLLA (Dense(2))** |
|---|---|---|
| **Output shape** | (batch, 1) | (batch, 2) |
| **Output 0** | Point prediction | Mean (μ_s) |
| **Output 1** | — | Log-variance (log_σ²_s) |
| **Loss** | MSE | Gaussian NLL |
| **Uncertainty source** | Dropout ensemble | Model output + LLLA |

---

### **Stage 3: Hyperparameter Search (Random Search) (Lines ~181-225)**
**File Location:** Same as MCD (random search with N_TRIALS=50)

```python
def sample_params():
    lookback = random.choice([30, 45, 60, 90])
    d_model = random.choice([32, 64, 96, 128])
    
    # Ensure d_model divisible by num_heads
    valid_heads = [h for h in (2, 4, 8) if d_model % h == 0 and d_model // h >= 8]
    num_heads = random.choice(valid_heads)
    
    dff = random.choice([2*d_model, 3*d_model, 4*d_model])

    params = {
        "lookback": lookback,
        "d_model": d_model,
        "num_heads": num_heads,
        "dff": dff,
        "num_layers": random.choice([1, 2, 3]),
        "dropout": np.random.uniform(0.0, 0.3),
        "optimizer": "adam",
        "lr": 10 ** np.random.uniform(-4, math.log10(5e-3)),
        "batch_size": random.choice([32, 64, 128]),
        "epochs": random.choice([30, 40, 50, 60, 70, 80, 90, 100]),
        "patience": random.choice([5, 6, 7, 8, 9, 10])
    }
    
    return params
```

**Key Point:** Hyperparameter space is identical to MCD, but the loss function is different.

---

### **Stage 4: Model Training (Lines ~226-260)**
**File Location:** Training with heteroscedastic loss

```python
# Combine train + val for final training
X_trainval_s = pd.concat([X_train_s, X_val_s], axis=0)
y_trainval_s = pd.concat([y_train_s, y_val_s], axis=0)

X_trv, y_trv, _ = make_windows(X_trainval_s, y_trainval_s, lb)

# Build model with best hyperparameters
best_model = build_transformer((lb, len(feature_cols)), best["params"])

# Train with heteroscedastic loss and early stopping
callbacks_final = [
    EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
    ModelCheckpoint(MODEL_BEST, monitor="val_loss", save_best_only=True),
]

start_train = time.time()
hist_final = best_model.fit(
    X_trv, y_trv,
    validation_split=0.1,
    epochs=best["params"]["epochs"],
    batch_size=best["params"]["batch_size"],
    callbacks=callbacks_final,
    verbose=VERBOSE_TRAIN
)
end_train = time.time()
```

**During Training:**
- Model learns TWO outputs: μ_s and log_σ²_s
- Loss penalizes:
  - High prediction error: `(y - μ)²`
  - Inappropriate variance estimates: σ² too large/small
- This creates **aleatoric uncertainty** - data-dependent, learned uncertainty

---

### **Stage 5: Heteroscedastic Inference (Lines ~261-305)** ⭐ **ALEATORIC UNCERTAINTY**
**File Location:** `predict_hetero()` function

```python
def predict_hetero(model, X_block, idx_block):
    """
    Get heteroscedastic predictions (μ and σ²_ale)
    """
    # Forward pass: (N, 2) → [mu_s, log_var_s]
    yhat_s = model.predict(X_block, verbose=0)  # Shape: (N, 2)
    
    mu_s      = yhat_s[:, 0]                     # Predicted mean (scaled)
    log_var_s = yhat_s[:, 1]                     # Predicted log-variance (scaled)
    
    # Inverse scaling back to original space
    y_mean = y_scaler.mean_[0]
    y_std  = y_scaler.scale_[0]
    
    # Inverse transform mean
    mu_orig = mu_s * y_std + y_mean              # (N,)
    
    # Inverse transform variance
    # σ²_orig = σ²_s × y_std²
    var_orig = np.exp(log_var_s) * (y_std ** 2)
    sigma_ale = np.sqrt(var_orig)                # (N,) - Aleatoric std dev
    
    return pd.DataFrame({
        "mu": mu_orig,
        "sigma_ale": sigma_ale                   # ALEATORIC UNCERTAINTY
    }, index=idx_block)

# Apply to all splits (lines 307-320)
X_tr, y_tr, idx_tr = make_windows(X_train_s, y_train_s, lb)
X_vl, y_vl, idx_vl = make_windows(X_val_s,   y_val_s,   lb)

pred_train = predict_hetero(best_model, X_tr, idx_tr)
pred_val   = predict_hetero(best_model, X_vl, idx_vl)
pred_test  = predict_hetero(best_model, X_te, idx_te)
```

**Outputs:**
```
For each sample:
  μ_orig = Point prediction (original scale)
  σ_ale = Aleatoric uncertainty (original scale)
```

---

### **Stage 6: Last-Layer Laplace Approximation (LLLA) (Lines ~322-395)** ⭐ **EPISTEMIC UNCERTAINTY**
**File Location:** LLLA computation

#### **6a: Extract Penultimate Layer Features (Lines ~322-330)**

```python
# Build a model that extracts features from the penultimate layer
penultimate_layer = best_model.layers[-2]  # Layer before output Dense(2)

# Create feature extractor: Input → ... → Penultimate layer
phi_model = tf.keras.Model(
    inputs=best_model.inputs[0],
    outputs=penultimate_layer.output
)

# Extract features for all splits
phi_train = phi_model.predict(X_tr, verbose=0)      # Shape: (N_train, d_model)
phi_val   = phi_model.predict(X_vl, verbose=0)      # Shape: (N_val, d_model)
phi_test  = phi_model.predict(X_te, verbose=0)      # Shape: (N_test, d_model)

phi_trainval = phi_model.predict(X_trv, verbose=0)  # All training data

print("phi_trainval shape:", phi_trainval.shape)    # (N_trainval, d_model)
print("phi_test shape:", phi_test.shape)             # (N_test, d_model)
```

**What are φ features?**
```
Transformer architecture:
Input → Dense(d_model) → Positional Encoding 
  → Encoder blocks (num_layers)
    → Extract last timestep
      → penultimate_layer (= Lambda layer extracting last timestep)
        → φ features = (d_model,) dimensional representation

These φ features are fed into the final Dense(2) layer.
The Laplace approximation assumes the last layer (Dense(2)) is linear,
and approximates its posterior distribution using the Hessian.
```

---

#### **6b: Compute Noise Variance Estimate (Lines ~331-345)**

```python
# Get predictions on training+val set (in scaled space)
yhat_trainval_s = best_model.predict(X_trv, verbose=0)  # (N_trainval, 2)
mu_trainval_s   = yhat_trainval_s[:, 0]                 # First output: μ_s

# Flatten the true targets
y_trainval_s_flat = y_trv.squeeze()  # (N_trainval,)

# Compute residuals on training data
residuals_s = y_trainval_s_flat - mu_trainval_s

# Estimate noise variance (aleatoric)
sigma_n2 = np.mean(residuals_s ** 2)  # Mean squared error as noise estimate

print("Estimated noise variance (scaled):", sigma_n2)
```

**Interpretation:**
```
σ_n² = (1/N) Σ(y_train - μ_train)²

This is the **empirical noise variance** used in Laplace:
- Assumes Gaussian likelihood: p(y|x,w) ∝ exp(-(y - f(x,w))² / σ_n²)
- Typical Laplace assumes fixed noise; here we estimate from residuals
```

---

#### **6c: Compute Hessian Diagonal Approximation (Lines ~346-352)**

```python
N_tr, H = phi_trainval.shape  # N_tr = N_trainval, H = d_model

# Augment features with bias term: [φ₁, φ₂, ..., φ_H, 1]
Phi_tr = np.concatenate([phi_trainval, np.ones((N_tr, 1))], axis=1)  # (N_tr, H+1)

# L2 regularization prior weight
lambda_prior = 1.0

# Fisher Information Matrix diagonal approximation:
# H_diag = (1/σ_n²) × Φᵀ Φ + λ × I
#
# Φᵀ Φ ≈ Σ(φᵢ ⊙ φᵢ) where ⊙ is element-wise multiplication
H_diag = (1.0 / sigma_n2) * np.sum(Phi_tr**2, axis=0) + lambda_prior

# Posterior precision diagonal: H_diag = [H_diag,1, ..., H_diag,H+1]
# Posterior covariance diagonal: var_w = 1 / H_diag
var_w_diag = 1.0 / H_diag

print("Posterior variance (diagonal):", var_w_diag)
```

**Mathematical Background:**
```
Last-layer Laplace Approximation (LLLA):

Standard Laplace:
  p(w|D) ≈ N(w*, Σ_w)  where Σ_w = H⁻¹ (posterior covariance)
  H = ∇²log p(w|D) = Hessian at MAP estimate w*

For classification: Hessian ≈ Σ p(y=c|x,w*) φ(x) φ(x)ᵀ
For regression with Gaussian: Hessian ≈ (1/σ_n²) Σ φ(x) φ(x)ᵀ + λI

Diagonal approximation (ignoring off-diagonals):
  var_w ≈ diag(H⁻¹) = 1 / diag(H)

This gives per-weight uncertainty.
For classification/regression head with input φ(x):
  var_f(x) = φ(x)ᵀ × Σ_w × φ(x) ≈ φ(x)ᵀ × diag(Σ_w) ⊙ φ(x)
                                    = Σ φᵢ(x)² × var_wᵢ
```

---

#### **6d: Compute Epistemic Uncertainty (Lines ~353-365)**

```python
def compute_sigma_epi(phi_block, idx_block):
    """
    Compute epistemic uncertainty via LLLA
    
    var_epi(x) = φ(x)ᵀ × Σ_w × φ(x)
                ≈ Σ φᵢ(x)² × var_wᵢ
    
    Epistemic uncertainty comes from uncertainty in weights w,
    not from data noise.
    """
    N, H = phi_block.shape
    
    # Augment with bias: [φ, 1]
    Phi_b = np.concatenate([phi_block, np.ones((N, 1))], axis=1)  # (N, H+1)
    
    # Compute var_epi in scaled space
    # var_epi_s = Σ φᵢ² × var_w_diag
    var_epi_scaled = np.sum((Phi_b**2) * var_w_diag.reshape(1, -1), axis=1)  # (N,)
    
    # Inverse scale to original space
    # var_epi_orig = var_epi_s × y_std²
    var_epi_orig = var_epi_scaled * (y_scaler.scale_[0] ** 2)
    
    # Return standard deviation (epistemic)
    return pd.Series(np.sqrt(np.maximum(var_epi_orig, 1e-12)), index=idx_block)

# Compute epistemic uncertainty for all splits
sigma_epi_train = compute_sigma_epi(phi_train, idx_tr)
sigma_epi_val   = compute_sigma_epi(phi_val,   idx_vl)
sigma_epi_test  = compute_sigma_epi(phi_test,  idx_te)
```

**Output:** Epistemic standard deviation per sample

---

### **Stage 7: Uncertainty Combination (Lines ~366-380)**
**File Location:** Combining aleatoric + epistemic

```python
# TRAIN: Combine aleatoric + epistemic
pred_train["sigma_epi"] = sigma_epi_train
pred_train["sigma_total"] = np.sqrt(
    pred_train["sigma_ale"]**2 + pred_train["sigma_epi"]**2
)

# VAL: Same process
pred_val["sigma_epi"] = sigma_epi_val
pred_val["sigma_total"] = np.sqrt(
    pred_val["sigma_ale"]**2 + pred_val["sigma_epi"]**2
)

# TEST: Same process
pred_test["sigma_epi"] = sigma_epi_test
pred_test["sigma_total"] = np.sqrt(
    pred_test["sigma_ale"]**2 + pred_test["sigma_epi"]**2
)

# Result: Each sample has
# - mu: point prediction
# - sigma_ale: aleatoric uncertainty (from heteroscedastic output)
# - sigma_epi: epistemic uncertainty (from LLLA)
# - sigma_total: √(σ_ale² + σ_epi²) combined uncertainty
```

**Formula:**
$$\sigma_{total}^2 = \sigma_{ale}^2 + \sigma_{epi}^2$$

where:
- $\sigma_{ale}$ = Aleatoric (data-dependent, learned from heteroscedastic output)
- $\sigma_{epi}$ = Epistemic (model uncertainty via LLLA)

---

### **Stage 8: Prediction Intervals & UQ Metrics (Lines ~381-430)**
**File Location:** Computing PICP, MPIW, Winkler

```python
def compute_pi_metrics_from_sigma(y_true, mu, sigma, z_level=1.96, alpha=0.05):
    """
    Build prediction intervals from μ and σ_total using z-score
    
    CI = [μ - z × σ, μ + z × σ]
    where z = 1.96 for 95% confidence (≈ 2 standard deviations)
    """
    y_true = np.asarray(y_true, dtype=float)
    mu     = np.asarray(mu, dtype=float)
    sigma  = np.asarray(sigma, dtype=float)

    # Build interval
    L = mu - z_level * sigma      # Lower bound
    U = mu + z_level * sigma      # Upper bound

    # 1) PICP: Prediction Interval Coverage Probability
    inside = (y_true >= L) & (y_true <= U)
    picp = inside.mean()

    # 2) MPIW: Mean Prediction Interval Width
    width = U - L
    mpiw = width.mean()

    # 3) Winkler Score: width + penalty for misses
    penalties = np.zeros_like(y_true)
    
    below = y_true < L
    above = y_true > U
    
    # Penalty: (2/α) × distance outside interval
    penalties[below] = (L[below] - y_true[below])
    penalties[above] = (y_true[above] - U[above])

    winkler = np.mean(width + (2.0 / alpha) * penalties)

    return picp, mpiw, winkler

# Compute for all splits
trans_picp_train, trans_mpiw_train, trans_wink_train = compute_pi_metrics_from_sigma(
    actual_train.values,
    pred_train["mu"].values,
    pred_train["sigma_total"].values,
    z_level=Z_LEVEL,
    alpha=ALPHA
)

trans_picp_test, trans_mpiw_test, trans_wink_test = compute_pi_metrics_from_sigma(
    actual_test.values,
    pred_test["mu"].values,
    pred_test["sigma_total"].values,
    z_level=Z_LEVEL,
    alpha=ALPHA
)

print("=== UQ Metrics ===")
print(f"TEST — PICP: {trans_picp_test:.4f} | MPIW: {trans_mpiw_test:.4f} | Winkler: {trans_wink_test:.4f}")
```

---

## 2. SUMMARY TABLE: WHERE HLLLA APPLIES IN TRANSFORMER

| **Stage** | **Component** | **HLLLA Role** | **Code Location** |
|-----------|---------------|---|---|
| **Output Head** | Dense(2) instead of Dense(1) | Heteroscedastic outputs | Lines ~148-151 |
| **Loss Function** | Gaussian NLL | Learn μ and σ²_ale | Lines ~112-125 |
| **Training** | fit() with heteroscedastic loss | Learn both mean & variance | Lines ~226-260 |
| **Inference (Stage 1)** | Forward pass → (μ_s, log_σ²_s) | Get aleatoric uncertainty | Lines ~307-320 |
| **Inverse Scaling** | Transform to original scale | Scale predictions back | Lines ~268-280 |
| **Feature Extraction** | phi_model (penultimate layer) | Get feature vectors φ | Lines ~322-330 |
| **Noise Estimation** | Residuals on training set | Estimate σ_n² | Lines ~331-345 |
| **Hessian Diagonal** | Fisher Information Matrix | Approximate posterior precision | Lines ~346-352 |
| **Epistemic Var** | `var_epi = Σ φ² × var_w` | Compute from LLLA | Lines ~353-365 |
| **Uncertainty Combine** | √(σ_ale² + σ_epi²) | Total uncertainty | Lines ~366-380 |
| **PI Metrics** | PICP, MPIW, Winkler | Evaluate UQ quality | Lines ~381-430 |

---

## 3. HLLLA FLOWCHART FOR TRANSFORMER

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  TRANSFORMER + HETEROSCEDASTIC NLL + LAST-LAYER LAPLACE APPROXIMATION (HLLLA) │
└──────────────────────────────────────────────────────────────────────────────┘

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
                    │  Random Search (N=50)        │
                    │  Sample hyperparameters:     │
                    │  - d_model, num_heads        │
                    │  - num_layers, dff           │
                    │  - dropout, lr, batch_size   │
                    └──────────────────────────────┘
                                      │
                    ══════════════════════════════════════════
                   ║  BUILD TRANSFORMER + HETERO OUTPUT       ║
                    ══════════════════════════════════════════
                                      │
                                      ▼
                    ┌───────────────────────────────────────────────┐
                    │  Transformer Architecture                     │
                    │  ┌─────────────────────────────────────────┐ │
                    │  │ INPUT (batch, lookback, n_features)    │ │
                    │  │           ↓                             │ │
                    │  │ Dense(d_model)                          │ │
                    │  │           ↓                             │ │
                    │  │ Positional Encoding                     │ │
                    │  │           ↓                             │ │
                    │  │ FOR each encoder layer (num_layers):   │ │
                    │  │ ┌───────────────────────────────────┐ │ │
                    │  │ │ MultiHeadAttention (causal)      │ │ │
                    │  │ │           ↓                       │ │ │
                    │  │ │ Dropout + Residual + LayerNorm   │ │ │
                    │  │ │           ↓                       │ │ │
                    │  │ │ Dense(dff, relu)                 │ │ │
                    │  │ │           ↓                       │ │ │
                    │  │ │ Dropout + Residual + LayerNorm   │ │ │
                    │  │ └───────────────────────────────────┘ │ │
                    │  │           ↓                             │ │
                    │  │ Extract last timestep (batch, d_model) │ │
                    │  │           ↓                             │ │
                    │  │ Dense(2) OUTPUT ◄─ HETERO HEAD         │ │
                    │  │ ├─ out[:, 0] = μ_s (mean)              │ │
                    │  │ └─ out[:, 1] = log_σ²_s (log-var)      │ │
                    │  │                                         │ │
                    │  │ Loss: nll_gaussian_heteroscedastic      │ │
                    │  │ Optimizer: Adam                         │ │
                    │  └─────────────────────────────────────────┘
                    └───────────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  TRAINING PHASE                      │
                    │  ├─ Fit model on train+val           │
                    │  ├─ Loss: Heteroscedastic NLL        │
                    │  ├─ Learns: μ and σ²_ale             │
                    │  ├─ EarlyStopping + Checkpointing    │
                    │  └─ Result: (μ, σ²_ale) per sample   │
                    └──────────────────────────────────────┘
                                      │
                    ══════════════════════════════════════════════════════
                   ║  START POST-TRAINING: LAST-LAYER LAPLACE APPROX      ║
                    ══════════════════════════════════════════════════════
                                      │
                                      ▼
                    ┌──────────────────────────────────────────┐
                    │  STAGE 1: HETEROSCEDASTIC INFERENCE      │
                    │  (Get Aleatoric Uncertainty)             │
                    │                                          │
                    │  For each sample x in train/val/test:   │
                    │  ┌──────────────────────────────────┐   │
                    │  │ Forward pass through Transformer  │   │
                    │  │ Get (μ_s, log_σ²_s) ← Output      │   │
                    │  │         ↓                         │   │
                    │  │ Inverse scale:                    │   │
                    │  │  μ_orig = μ_s × y_scale + y_mean  │   │
                    │  │  σ²_orig = exp(log_σ²_s) × y²    │   │
                    │  │  σ_ale = √(σ²_orig)               │   │
                    │  │         ↓                         │   │
                    │  │ Output: (μ_orig, σ_ale)           │   │
                    │  │ ← ALEATORIC UNCERTAINTY           │   │
                    │  └──────────────────────────────────┘   │
                    └──────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────────┐
                    │  STAGE 2: FEATURE EXTRACTION (LLLA)      │
                    │  (Get φ features from penultimate layer) │
                    │                                          │
                    │  Transformer structure:                  │
                    │  Input → Dense(d_model)                 │
                    │        → Encoder blocks (num_layers)     │
                    │          → Extract last timestep         │
                    │            ↓                             │
                    │  φ_model = Input → ... → Last timestep  │
                    │            ↓                             │
                    │  φ = φ_model(X) ← Feature vectors       │
                    │      Shape: (N, d_model)                │
                    │  ← PENULTIMATE LAYER FEATURES           │
                    └──────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────────┐
                    │  STAGE 3: NOISE VARIANCE ESTIMATION      │
                    │                                          │
                    │  ŷ_s = model(X_trainval)[:, 0]          │
                    │  residuals = y_trainval - ŷ_s           │
                    │  σ_n² = mean(residuals²)                 │
                    │  ← ESTIMATED DATA NOISE VARIANCE        │
                    └──────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────────┐
                    │  STAGE 4: HESSIAN DIAGONAL APPROXIMATION │
                    │                                          │
                    │  Φ = [φ₁, φ₂, ..., φₙ, 1 (bias)]        │
                    │      Shape: (N, d_model+1)              │
                    │                                          │
                    │  H_diag = (1/σ_n²) × Σ(Φ ⊙ Φ) + λI     │
                    │           ← Fisher Information (diagonal)│
                    │                                          │
                    │  var_w = 1 / H_diag                      │
                    │  ← POSTERIOR WEIGHT VARIANCE             │
                    └──────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────────┐
                    │  STAGE 5: EPISTEMIC UNCERTAINTY (LLLA)   │
                    │                                          │
                    │  For each sample φ(x):                  │
                    │  var_epi(x) = Σ φᵢ(x)² × var_wᵢ        │
                    │                                          │
                    │  Inverse scale:                         │
                    │  var_epi_orig = var_epi_scaled × y²    │
                    │  σ_epi = √(var_epi_orig)                │
                    │  ← EPISTEMIC UNCERTAINTY (LLLA)         │
                    │                                          │
                    │  Computed for train, val, test splits   │
                    └──────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────────┐
                    │  STAGE 6: TOTAL UNCERTAINTY              │
                    │                                          │
                    │  σ_total = √(σ_ale² + σ_epi²)           │
                    │                                          │
                    │  Combines:                              │
                    │  - σ_ale: Data-dependent (from hetero)  │
                    │  - σ_epi: Model uncertainty (from LLLA) │
                    │  ← TOTAL PREDICTIVE UNCERTAINTY         │
                    └──────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────────┐
                    │  STAGE 7: PREDICTION INTERVALS           │
                    │                                          │
                    │  L = μ - 1.96 × σ_total  (95% lower)    │
                    │  U = μ + 1.96 × σ_total  (95% upper)    │
                    │  ← PREDICTION INTERVAL [L, U]           │
                    └──────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────────┐
                    │  STAGE 8: UQ METRICS EVALUATION          │
                    │                                          │
                    │  ├─ PICP: P(y ∈ [L, U])                │
                    │  ├─ MPIW: E[U - L]                      │
                    │  ├─ Winkler: Width + Penalty            │
                    │  └─ Coverage/Sharpness plots            │
                    │                                          │
                    │  ← UNCERTAINTY QUANTIFICATION METRICS   │
                    └──────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────────┐
                    │  SAVE RESULTS & DOCUMENTATION           │
                    │  ├─ ALL_UQ_PREDICTED.csv                │
                    │  │  (μ, L, U per sample)                │
                    │  └─ ALL_UQ_METRICS.csv                  │
                    │     (PICP, MPIW, Winkler, etc.)        │
                    └──────────────────────────────────────────┘
                                      │
                                      ▼
                                   OUTPUT:
                          Predictions + UQ Bounds
                              & Metrics
```

---

## 4. KEY HLLLA CONCEPTS FOR TRANSFORMER

### **Uncertainty Decomposition**

```
ALEATORIC UNCERTAINTY (σ_ale):
├─ Source: Data-dependent noise
├─ Learned via heteroscedastic output head (Dense(2))
├─ Cannot be reduced by collecting more data
├─ Represents irreducible randomness in y given x
└─ Example: sensor noise, measurement error

EPISTEMIC UNCERTAINTY (σ_epi):
├─ Source: Model/weight uncertainty via LLLA
├─ Computed post-training from Hessian diagonal
├─ Reduces with more data (model becomes more certain)
├─ Represents knowledge gap about true weights w
└─ Example: insufficient training data coverage

TOTAL UNCERTAINTY (σ_total):
├─ σ_total = √(σ_ale² + σ_epi²)
├─ Combines both sources
├─ Used for prediction intervals
└─ Better calibrated than either alone
```

---

### **Last-Layer Laplace Approximation (LLLA) Mathematics**

**Laplace Approximation (Bayesian Posterior):**

For a trained neural network with loss $\mathcal{L}(w)$:

$$p(w|D) \approx \mathcal{N}(w^*, H^{-1})$$

where:
- $w^*$ = MAP estimate (trained weights)
- $H = \nabla^2 \mathcal{L}(w^*)$ = Hessian of loss at $w^*$

**Full-Network Laplace:** Compute Hessian of entire network → computationally infeasible

**Last-Layer Laplace Approximation (LLLA):**
- Only approximate posterior of final layer weights
- Assume intermediate layers' weights are fixed at $w^*$
- Treat final layer as linear model with basis $\phi(x) = $ penultimate layer output

**For Heteroscedastic Regression:**

Given Gaussian likelihood:
$$p(y|x,w) = \mathcal{N}(\mu(x,w), \sigma_n^2)$$

The Fisher Information Matrix diagonal:
$$H_{diag} = \frac{1}{\sigma_n^2} \sum_{i=1}^N \phi(x_i) \otimes \phi(x_i) + \lambda I$$

Posterior variance of weights:
$$\text{Var}(w) \approx H_{diag}^{-1}$$

Epistemic uncertainty at test point $x$:
$$\text{Var}_{epi}(f(x)) = \phi(x)^T \text{Var}(w) \phi(x) \approx \sum_j \phi_j(x)^2 \text{Var}(w_j)$$

---

### **Heteroscedastic Loss vs Standard MSE**

```python
# Standard MSE Loss
MSE Loss:
  loss = mean((y - ŷ)²)
  Problem: Treats all predictions equally
           Doesn't learn uncertainty

# Heteroscedastic Gaussian NLL Loss
nll_gaussian_heteroscedastic:
  loss = mean(0.5 × [log(σ²) + (y - μ)²/σ²])
  
  Benefits:
  ├─ Encourages high uncertainty when wrong
  ├─ Encourages low uncertainty when right
  ├─ Automatically learns appropriate σ per sample
  └─ Directly optimizes log-likelihood

# Mathematical intuition:
  - If σ² is too small: penalty ∝ (y - μ)²/σ² → large loss
  - If σ² is too large: penalty ∝ log(σ²) → loss increases
  - Balances prediction accuracy with uncertainty estimation
```

---

### **Why Two-Stage Uncertainty (HLLLA)?**

**Stage 1: Aleatoric via Heteroscedastic Output**
```
Advantages:
├─ Directly learned during training
├─ Captures data-dependent noise
├─ Varies per sample
└─ Computationally cheap at inference

Limitations:
├─ Only captures noise learned from training data
├─ Doesn't account for weight uncertainty
└─ Can underestimate uncertainty in OOD regions
```

**Stage 2: Epistemic via LLLA**
```
Advantages:
├─ Captures model/weight uncertainty
├─ Higher in regions with sparse training data
├─ Principled Bayesian approach
└─ Complements aleatoric uncertainty

Limitations:
├─ Post-training computation (extra cost)
├─ Diagonal approximation (ignores correlations)
├─ Assumes fixed noise variance σ_n²
└─ Requires access to training data
```

**Combined Benefits:**
```
σ_total = √(σ_ale² + σ_epi²)

├─ Data noise (σ_ale) → tight intervals in dense regions
├─ Model uncertainty (σ_epi) → wider intervals in sparse regions
└─ Better calibrated uncertainty than either alone
```

---

## 5. TRANSFORMER HLLLA VS TCN HLLLA

| **Aspect** | **Transformer HLLLA** | **TCN HLLLA** |
|---|---|---|
| **Penultimate Layer** | Lambda (extract last timestep) | Dense or Conv1D layer |
| **Feature Dimension** | d_model (32-128) | n_filters or dense dim |
| **Attention Mechanism** | Multi-head attention in encoder | None (convolutional) |
| **Position Awareness** | Positional encoding (sinusoidal) | Implicit via receptive field |
| **Computational Cost** | O(L²) attention + O(L) inference | O(L) convolution |
| **Parallelization** | Better (attention parallelizable) | Better (conv parallelizable) |
| **Hessian Diagonal** | Same Fisher approximation | Same Fisher approximation |
| **Epistemic Uncertainty** | Via LLLA on last timestep feature | Via LLLA on penultimate feature |

---

## 6. KEY DIFFERENCES: MCD vs HLLLA IN TRANSFORMER

| **Aspect** | **MCD** | **HLLLA** |
|---|---|---|
| **Output Head** | Dense(1) | Dense(2) |
| **Training Uncertainty** | Via dropout regularization | Via heteroscedastic loss |
| **Uncertainty Mechanism** | Ensemble during inference | Explicit model outputs |
| **Epistemic** | Approximated from ensemble variance | LLLA Hessian diagonal |
| **Aleatoric** | Residual variance estimate | Learned from σ²_s output |
| **Inference Cost** | 100× forward passes | 1 forward pass + LLLA |
| **Loss Function** | MSE | Gaussian NLL |
| **Calibration** | From ensemble quantiles | From learned variance + LLLA |
| **Theoretical** | Bayesian NN approximation | Laplace approximation |

---

## Summary

**HLLLA in Transformer applies through:**

1. **Heteroscedastic Output Head** (Dense(2)) - Learn μ and σ²
2. **Gaussian NLL Loss** - Train both mean and variance
3. **Heteroscedastic Inference** - Get aleatoric uncertainty (σ_ale)
4. **Feature Extraction** - Extract φ from penultimate layer
5. **Hessian Diagonal** - Fisher Information approximation
6. **Epistemic Computation** - LLLA weight uncertainty (σ_epi)
7. **Uncertainty Combination** - σ_total = √(σ_ale² + σ_epi²)
8. **Prediction Intervals** - CI = [μ ± 1.96 × σ_total]

The method provides **principled uncertainty decomposition** into data noise (aleatoric) and model uncertainty (epistemic), with better calibration than either component alone.
