# Heteroscedastic NLL + Last-Layer Laplace Approximation (HLLLA) in RNNs (GRU & LSTM): Detailed Analysis

## Overview

The GRU_HLLLA_42.ipynb and LSTM_HLLLA_42.ipynb implementations apply **Heteroscedastic Negative Log-Likelihood (NLL) + Last-Layer Laplace Approximation (LLLA)** for uncertainty quantification in recurrent time-series forecasting. This method decomposes uncertainty into:

1. **Aleatoric (Data) Uncertainty**: Learned directly by the model via heteroscedastic output
2. **Epistemic (Model) Uncertainty**: Estimated via Last-Layer Laplace Approximation on penultimate layer features

Both GRU and LSTM follow identical HLLLA methodologies, with the same 6-stage pipeline regardless of RNN type. This document explains the unified HLLLA implementation.

---

## 1. WHERE HLLLA IS APPLIED IN RNN PIPELINES (GRU & LSTM)

### **Stage 1: RNN Architecture with Heteroscedastic Output (Lines ~58-82 in GRU, ~57-81 in LSTM)**
**File Location:** `build_gru()` / `build_lstm()` functions

#### **GRU Architecture:**
```python
def build_gru(trial, lookback, n_features):
    # Hyperparameters
    num_layers = trial.suggest_int("num_layers", 1, 2)
    units1     = trial.suggest_int("units1", 32, 256, step=32)
    units2     = trial.suggest_int("units2", 32, 256, step=32) if num_layers == 2 else None
    dropout    = trial.suggest_float("dropout", 0.0, 0.5)
    lr         = trial.suggest_float("lr", 1e-4, 5e-3, log=True)

    model = Sequential()
    model.add(Input(shape=(lookback, n_features)))  # (batch, lookback, n_features)

    # RNN layers
    if num_layers == 2:
        model.add(GRU(units1, return_sequences=True))  # → (batch, lookback, units1)
        model.add(Dropout(dropout))
        model.add(GRU(units2))                         # → (batch, units2)
    else:
        model.add(GRU(units1))                         # → (batch, units1)

    model.add(Dropout(dropout))
    
    # ← CRITICAL: Heteroscedastic output head (2 outputs, not 1)
    model.add(Dense(2))  # Output: [μ_s, log(σ²)_s] in scaled space
    
    # ← CRITICAL: Heteroscedastic Gaussian NLL loss
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss=nll_gaussian_heteroscedastic)
    return model
```

#### **LSTM Architecture (identical structure):**
```python
def build_lstm(trial, lookback, n_features):
    num_layers = trial.suggest_int("num_layers", 1, 2)
    units1     = trial.suggest_int("units1", 32, 256, step=32)
    units2     = trial.suggest_int("units2", 32, 256, step=32) if num_layers == 2 else None
    dropout    = trial.suggest_float("dropout", 0.0, 0.5)
    lr         = trial.suggest_float("lr", 1e-4, 5e-3, log=True)

    model = Sequential()
    model.add(Input(shape=(lookback, n_features)))

    if num_layers == 2:
        model.add(LSTM(units1, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(units2))
    else:
        model.add(LSTM(units1))

    model.add(Dropout(dropout))
    model.add(Dense(2))  # ← Heteroscedastic output: [μ_s, log(σ²)_s]
    
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss=nll_gaussian_heteroscedastic)
    return model
```

#### **Heteroscedastic Loss Function (Lines ~43-52):**
```python
def nll_gaussian_heteroscedastic(y_true, y_pred):
    """
    Heteroscedastic Gaussian Negative Log-Likelihood
    
    y_true: (batch,) or (batch,1) - actual target
    y_pred: (batch,2) - [μ_s, log(σ²)_s] in SCALED SPACE
    
    NLL = 0.5 * [log(σ²) + (y - μ)²/σ²]
    """
    y_true = tf.cast(tf.reshape(y_true, (-1,)), tf.float32)
    mu      = y_pred[:, 0]                    # Mean head
    log_var = tf.clip_by_value(y_pred[:, 1], -20.0, 5.0)  # Log-variance head (clipped)
    
    inv_var = tf.exp(-log_var)               # 1/σ²
    
    # NLL = 0.5 * (log_var + (y - mu)² / var)
    nll = 0.5 * (log_var + (y_true - mu)**2 * inv_var)
    
    return tf.reduce_mean(nll)
```

**Key Architecture Points:**
- **Heteroscedastic Output**: Dense(2) with two heads instead of Dense(1)
  - Head 1: Mean μ_s (in scaled space)
  - Head 2: Log-variance log(σ²)_s (in scaled space)
- **Loss Function**: Heteroscedastic Gaussian NLL (not MSE)
  - Learns both mean AND variance from data
  - Allows model to express higher uncertainty for harder samples
- **Penultimate Layer**: The Dropout layer before Dense(2) is crucial for LLLA feature extraction

---

### **Stage 2: Hyperparameter Optimization via Optuna (Lines ~115-165)**
**File Location:** Optuna objective function

```python
def objective(trial):
    # Sample hyperparameters
    lookback = trial.suggest_categorical("lookback", [30, 45, 60, 90])
    
    X_tr, y_tr, _ = make_windows(X_train_s, y_train_s, lookback)
    X_va, y_va, _ = make_windows(X_val_s,   y_val_s,   lookback)
    
    # Reshape targets for NLL loss
    y_tr = y_tr.reshape(-1, 1)   # (N_train, 1)
    y_va = y_va.reshape(-1, 1)   # (N_val, 1)

    # RNN hyperparameters
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    epochs     = trial.suggest_int("epochs", 30, 100, step=10)
    patience   = trial.suggest_int("patience", 5, 10)

    # Build model with heteroscedastic output
    model = build_gru(trial, lookback, n_features=len(feature_cols))
    
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        TFKerasPruningCallback(trial, monitor="val_loss"),
    ]

    history = model.fit(
        X_tr, y_tr,                    # Training on NLL loss
        validation_data=(X_va, y_va),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    return min(history.history["val_loss"])

# Execute Optuna
sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
pruner  = optuna.pruners.MedianPruner(n_warmup_steps=5)
study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

study.optimize(objective, n_trials=N_TRIALS)  # N_TRIALS = 50
best_params = study.best_params
BEST_LOOKBACK = best_params["lookback"]
```

**Key Points:**
- **Heteroscedastic loss**: Optimizes both mean and variance jointly
- **NLL metric**: Lower NLL means better fit + better uncertainty calibration
- **No explicit dropout tuning for LLLA**: Dropout is just regularization here, not for Bayesian inference

---

### **Stage 3: Model Training (Lines ~175-200)**
**File Location:** Final training with best hyperparameters

```python
final_model = Sequential()
final_model.add(Input(shape=(BEST_LOOKBACK, len(feature_cols))))

# Build with best hyperparameters
if best_params["num_layers"] == 2:
    final_model.add(GRU(best_params["units1"], return_sequences=True))
    final_model.add(Dropout(best_params["dropout"]))
    final_model.add(GRU(best_params["units2"]))
else:
    final_model.add(GRU(best_params["units1"]))

final_model.add(Dropout(best_params["dropout"]))
final_model.add(Dense(2))  # Heteroscedastic output

final_model.compile(
    optimizer=optimizers.Adam(learning_rate=best_params["lr"]),
    loss=nll_gaussian_heteroscedastic
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=best_params["patience"], 
                 restore_best_weights=True),
    ModelCheckpoint("gru_optuna_best.keras", monitor="val_loss", save_best_only=True)
]

print("Retraining final GRU...")
history = final_model.fit(
    X_train_w, y_train_w,
    validation_data=(X_val_w, y_val_w),
    epochs=best_params["epochs"],
    batch_size=best_params["batch_size"],
    verbose=1,
    callbacks=callbacks
)
```

**During Training:**
- Model learns to output both **mean prediction** and **variance estimate**
- Heteroscedastic NLL loss guides learning: low loss ⟹ good fit + good variance calibration
- Dropout acts as regularization (not Bayesian inference yet)
- Saves best model for LLLA analysis

---

### **Stage 4: Feature Extraction via Penultimate Layer (Lines ~210-230)** ⭐ **CRITICAL FOR LLLA**
**File Location:** `phi_model` construction

```python
# ← CRITICAL: Extract features from penultimate layer (before Dense(2))
penultimate_layer = final_model.layers[-2]  # The Dropout layer

# Create feature extraction model
phi_model = tf.keras.Model(
    inputs=final_model.layers[0].input,  # Input layer
    outputs=penultimate_layer.output      # Penultimate layer output φ(x)
)

# Get feature representations for each split
phi_train = phi_model.predict(X_train_w, verbose=0)  # Shape: (N_train, H)
phi_val   = phi_model.predict(X_val_w,   verbose=0)  # Shape: (N_val, H)
phi_test  = phi_model.predict(X_test_w,  verbose=0)  # Shape: (N_test, H)

print("phi_train shape:", phi_train.shape)  # e.g., (1000, 128)
print("phi_val shape:",   phi_val.shape)
print("phi_test shape:",  phi_test.shape)
```

**Key Concept:**
- **Penultimate layer**: The layer just before the output head (Dense(2))
  - In our case: The Dropout layer after GRU/LSTM
  - Outputs: Hidden state representations φ(x) ∈ ℝ^H
  - H = units1 or units2 (from GRU/LSTM)
- **Why penultimate?**: LLLA approximates posterior distribution of the final layer weights
  - By fixing earlier layers and only approximating final layer
  - Makes computation tractable (only H+1 weights to approximate, not all millions)

---

### **Stage 5: Last-Layer Laplace Approximation (Lines ~232-270)** ⭐ **CORE HLLLA METHOD**
**File Location:** Hessian approximation and posterior variance computation

#### **5.1 Get MAP Predictions in Scaled Space (Lines ~232-242):**
```python
# Get MAP (Maximum A-Posteriori) predictions from trained model
yhat_train_s = final_model.predict(X_train_w, verbose=0)  # (N_train, 2)
mu_train_s   = yhat_train_s[:, 0]                         # Extract only mean head

# Get true targets in scaled space
y_true_train_s = y_train_w  # (N_train,) - already scaled

# Estimate noise variance σ_n² from training residuals IN SCALED SPACE
residuals_train_s = y_true_train_s - mu_train_s
sigma_n2 = np.mean(residuals_train_s**2)  # Scalar noise variance

print("Estimated noise variance (scaled):", sigma_n2)
```

**Key Points:**
- **MAP estimates**: Mean predictions from the heteroscedastic model
- **Noise variance**: Estimated from training residuals (in scaled space)
  - This becomes the likelihood precision for LLLA
  - Used for Hessian approximation

#### **5.2 Build Extended Features with Bias Term (Lines ~244-260):**
```python
# Extended feature matrix: Φ = [φ, 1] for implicit bias
N_train, H = phi_train.shape

Phi_train = np.concatenate(
    [phi_train, np.ones((N_train, 1), dtype=phi_train.dtype)],
    axis=1
)  # Shape: (N_train, H+1)

print("Phi_train shape:", Phi_train.shape)  # e.g., (1000, 129)

# Prior precision: λ (regularization term)
lambda_prior = 1.0  # Hyperparameter (can tune: 0.1, 1.0, 10.0)

# Diagonal Hessian approximation
# H_diag = (1/σ_n²) * Σ_i φ_i ⊗ φ_i + λI
# For diagonal only: H_diag[j] = (1/σ_n²) * Σ_i φ_i[j]²  + λ
H_diag = (1.0 / sigma_n2) * np.sum(Phi_train**2, axis=0) + lambda_prior
# Shape: (H+1,)

# Posterior variance: Σ = H^{-1} (diagonal approximation)
var_w_diag = 1.0 / H_diag  # (H+1,)

print("var_w_diag shape:", var_w_diag.shape)  # e.g., (129,)
```

**Mathematical Background:**

The Laplace Approximation uses a quadratic expansion of the log-posterior around the MAP estimate:

$$\log p(\mathbf{w}|\mathcal{D}) \approx \log p(\hat{\mathbf{w}}|\mathcal{D}) - \frac{1}{2}(\mathbf{w} - \hat{\mathbf{w}})^T \mathbf{H}(\mathbf{w} - \hat{\mathbf{w}})$$

Where:
- $\hat{\mathbf{w}}$ = MAP estimate (from trained model)
- $\mathbf{H}$ = Hessian of negative log-posterior at $\hat{\mathbf{w}}$
- The posterior is approximately Gaussian: $p(\mathbf{w}|\mathcal{D}) \approx \mathcal{N}(\hat{\mathbf{w}}, \mathbf{H}^{-1})$

**Diagonal Hessian:**
$$H_{jj} = \frac{1}{\sigma_n^2} \sum_{i=1}^{N} \phi_i^2[j] + \lambda$$

Where:
- First term: Data-dependent (likelihood Hessian)
- Second term: Prior precision (ridge regularization)
- Only diagonal elements kept (computational efficiency)

---

### **Stage 6: Epistemic Uncertainty Estimation (Lines ~272-300)**
**File Location:** Epistemic variance computation for test data

```python
# Helper function to compute epistemic std for any split
y_std = y_scaler.scale_[0]  # Scale factor for inverse transformation

def compute_sigma_epi(phi_block, idx_block):
    """
    Compute epistemic std from LLLA posterior variance
    
    σ_epi² = Φ Σ Φ^T (diagonal approximation)
    """
    N, H = phi_block.shape
    
    # Extend features with bias
    Phi_block = np.concatenate(
        [phi_block, np.ones((N, 1), dtype=phi_block.dtype)],
        axis=1
    )  # (N, H+1)

    # Epistemic variance in SCALED space
    # var_epi_scaled[i] = Σ_j Phi_block[i,j]² * var_w_diag[j]
    # This is: Φ_i Σ Φ_i^T (diagonal only)
    var_epi_scaled = np.sum((Phi_block**2) * var_w_diag.reshape(1, -1), axis=1)  # (N,)
    
    # Transform back to ORIGINAL SCALE
    # If y_scaled = (y - mean)/std, then var_original = var_scaled * std²
    var_epi_orig   = var_epi_scaled * (y_std**2)
    
    # Standard deviation (avoid negatives)
    sigma_epi_orig = np.sqrt(np.maximum(var_epi_orig, 1e-12))

    return pd.Series(sigma_epi_orig, index=idx_block, name="sigma_epi")

# Compute epistemic std for each split
sigma_epi_train = compute_sigma_epi(phi_train, idx_train)  # (N_train,)
sigma_epi_val   = compute_sigma_epi(phi_val,   idx_val)    # (N_val,)
sigma_epi_test  = compute_sigma_epi(phi_test,  idx_test)   # (N_test,)

print("sigma_epi_train shape:", sigma_epi_train.shape)
```

**Interpretation:**
- **σ_epi**: Uncertainty in the model's final layer weights (Last-Layer Laplace)
- **High σ_epi**: Model is uncertain about predictions (e.g., extrapolation region)
- **Low σ_epi**: Model is confident (e.g., interpolation, well-seen data)
- **Epistemic sources**: Limited training data, model capacity, parameter uncertainty

---

### **Stage 7: Aleatoric Uncertainty Extraction (Lines ~302-330)**
**File Location:** Direct extraction from heteroscedastic output

```python
def predict_series_hetero(model, X_block, idx_block):
    """
    Extract both mean and aleatoric uncertainty from heteroscedastic output
    """
    # Forward pass: model outputs [μ_s, log(σ²)_s] in SCALED space
    yhat_s = model.predict(X_block, verbose=0)  # (N, 2)
    
    mu_s      = yhat_s[:, 0]         # Mean in scaled space
    log_var_s = yhat_s[:, 1]         # Log-variance in scaled space

    # Transform MEAN back to original scale
    mu_orig = mu_s * y_scaler.scale_[0] + y_scaler.mean_[0]
    
    # Transform VARIANCE back to original scale
    # var = exp(log_var), and if y_scaled = (y - mean)/std, then:
    # var_original = var_scaled * std²
    var_orig   = np.exp(log_var_s) * (y_scaler.scale_[0] ** 2)
    
    # Aleatoric std in original scale
    sigma_ale = np.sqrt(var_orig)

    return pd.DataFrame({
        "mu": mu_orig,
        "sigma_ale": sigma_ale
    }, index=idx_block)

# Get aleatoric uncertainty for each split
pred_train = predict_series_hetero(final_model, X_train_w, idx_train)
pred_val   = predict_series_hetero(final_model, X_val_w,   idx_val)
pred_test  = predict_series_hetero(final_model, X_test_w,  idx_test)

print("pred_train columns:", pred_train.columns)  # [mu, sigma_ale]
```

**Interpretation:**
- **σ_ale**: Intrinsic noise/randomness in the target (aleatoric uncertainty)
- **Learned by model**: During heteroscedastic NLL training
- **Sample-dependent**: Can vary across data points (heteroscedastic!)
- **Aleatoric sources**: Measurement noise, inherent randomness, missing features

---

### **Stage 8: Uncertainty Combination (Lines ~332-350)**
**File Location:** Combine aleatoric + epistemic for total uncertainty

```python
# Extract aleatoric uncertainties
sigma_ale_train = pred_train["sigma_ale"]
sigma_ale_test  = pred_test["sigma_ale"]

# Compute TOTAL uncertainty (both sources combined)
sigma_total_train = np.sqrt(sigma_ale_train**2 + sigma_epi_train**2)
sigma_total_test  = np.sqrt(sigma_ale_test**2  + sigma_epi_test**2)

# Add to DataFrames for convenience
pred_train["sigma_epi"]   = sigma_epi_train
pred_test["sigma_epi"]    = sigma_epi_test

pred_train["sigma_total"] = sigma_total_train
pred_test["sigma_total"]  = sigma_total_test

# Build prediction intervals from total uncertainty
L_test = mu_test - Z_LEVEL * sigma_total_test  # Z_LEVEL = 1.96 for 95% CI
U_test = mu_test + Z_LEVEL * sigma_total_test
```

**Formula:**
$$\sigma_{\text{total}}^2 = \sigma_{\text{ale}}^2 + \sigma_{\text{epi}}^2$$

Where:
- $\sigma_{\text{ale}}^2$ = Aleatoric variance (from heteroscedastic output)
- $\sigma_{\text{epi}}^2$ = Epistemic variance (from LLLA)
- Both sources contribute to final prediction intervals

---

### **Stage 9: Uncertainty Quantification Metrics (Lines ~352-375)**
**File Location:** Compute PICP, MPIW, Winkler

```python
def compute_pi_metrics_from_sigma(y_true, mu, sigma, z_level=1.96, alpha=0.05):
    """
    Compute prediction interval metrics
    """
    y_true = np.asarray(y_true, dtype=float)
    mu     = np.asarray(mu, dtype=float)
    sigma  = np.asarray(sigma, dtype=float)

    # Prediction intervals
    L = mu - z_level * sigma
    U = mu + z_level * sigma

    # PICP: Prediction Interval Coverage Probability
    inside = (y_true >= L) & (y_true <= U)
    picp = inside.mean()  # Proportion of points inside interval

    # MPIW: Mean Prediction Interval Width
    width = U - L
    mpiw = width.mean()

    # Winkler Score: width + penalty for misses
    penalties = np.zeros_like(y_true)
    penalties[y_true < L] = (L[y_true < L] - y_true[y_true < L])
    penalties[y_true > U] = (y_true[y_true > U] - U[y_true > U])
    
    winkler = np.mean(width + (2.0 / alpha) * penalties)

    return picp, mpiw, winkler

# Compute metrics
picp_test, mpiw_test, winkler_test = compute_pi_metrics_from_sigma(
    actual_test.values,
    pred_test["mu"].values,
    pred_test["sigma_total"].values,
    z_level=Z_LEVEL,
    alpha=ALPHA
)

print(f"Test: PICP={picp_test:.4f}, MPIW={mpiw_test:.4f}, Winkler={winkler_test:.4f}")
```

---

## 2. SUMMARY TABLE: WHERE HLLLA APPLIES IN RNN PIPELINES

| **Stage** | **Component** | **HLLLA Role** | **Code Location (GRU/LSTM)** |
|-----------|---------------|---|---|
| **Architecture** | Dense(2) heteroscedastic output | Two heads: mean + log-variance | Lines ~77 |
| **Loss Function** | Heteroscedastic Gaussian NLL | Learn both mean and variance | Lines ~43-52 |
| **Training** | Standard fit() with NLL loss | Joint optimization of μ and σ | Lines ~175-200 |
| **HPO** | Optuna (50 trials) | Optimize NLL metric | Lines ~115-165 |
| **Best Model Save** | ModelCheckpoint callback | Store trained weights | Line ~190 |
| **Feature Extraction** | Penultimate layer (Dropout) | Extract φ(x) before Dense(2) | Lines ~210-230 |
| **MAP Estimation** | Get predictions from model | Use μ_s as MAP estimate | Lines ~232-242 |
| **Noise Variance** | Training residuals | σ_n² from training error | Lines ~236-242 |
| **Extended Features** | Φ = [φ, 1] | Add bias term to features | Lines ~244-260 |
| **Hessian** | Diagonal approximation | $H_{jj} = (1/σ_n²) Σ φ²[j] + λ$ | Lines ~256-260 |
| **Posterior Variance** | Σ = H^{-1} | Diagonal: var_w = 1/H_diag | Lines ~262-270 |
| **Epistemic Variance** | σ_epi² = Φ Σ Φ^T | Per-sample epistemic uncertainty | Lines ~272-300 |
| **Aleatoric Variance** | σ_ale² from model | Per-sample aleatoric uncertainty | Lines ~302-330 |
| **Total Variance** | σ_total² = σ_ale² + σ_epi² | Combined uncertainty | Lines ~332-350 |
| **Prediction Intervals** | [μ - 1.96σ, μ + 1.96σ] | 95% CI from total uncertainty | Lines ~352-375 |
| **Evaluation** | PICP, MPIW, Winkler | Measure UQ quality | Lines ~352-375 |

---

## 3. HLLLA FLOWCHART FOR GRU & LSTM

```
┌──────────────────────────────────────────────────────────────────────────┐
│ RNN (GRU & LSTM) + Heteroscedastic NLL + Last-Layer Laplace (HLLLA)      │
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
                    │  - lookback (30/45/60/90)    │
                    └──────────────────────────────┘
                                      │
                    ══════════════════════════════════════════════════════
                   ║  BUILD GRU/LSTM WITH HETEROSCEDASTIC OUTPUT         ║
                    ══════════════════════════════════════════════════════
                                      │
                                      ▼
                    ┌───────────────────────────────────────────────────┐
                    │  ARCHITECTURE: Heteroscedastic Output             │
                    │  ┌─────────────────────────────────────────────┐ │
                    │  │ INPUT (batch, lookback, n_features)        │ │
                    │  │           ↓                                 │ │
                    │  │ Layer 1: GRU/LSTM(units1)                  │ │
                    │  │ ├─ Process all timesteps                   │ │
                    │  │ ├─ Maintain hidden state h_t               │ │
                    │  │ └─ Output: (batch, units1) or              │ │
                    │  │           (batch, lookback, units1)        │ │
                    │  │           ↓                                 │ │
                    │  │ Dropout(dropout_rate)                      │ │
                    │  │           ↓                                 │ │
                    │  │ IF num_layers == 2:                        │ │
                    │  │ │  Layer 2: GRU/LSTM(units2)              │ │
                    │  │ │  Output: (batch, units2)                │ │
                    │  │           ↓                                 │ │
                    │  │ Dropout(dropout_rate)                      │ │
                    │  │           ↓                                 │ │
                    │  │ φ = Lambda layer features (penultimate)    │ │
                    │  │       Shape: (batch, H)                    │ │
                    │  │           ↓                                 │ │
                    │  │ Dense(2) ← HETEROSCEDASTIC OUTPUT HEAD     │ │
                    │  │ ├─ Output 1: μ_s (mean in scaled space)   │ │
                    │  │ ├─ Output 2: log(σ²)_s (log-var scaled)   │ │
                    │  │ └─ Shape: (batch, 2)                       │ │
                    │  │                                             │ │
                    │  │ Loss: Heteroscedastic Gaussian NLL         │ │
                    │  │ ├─ NLL = 0.5 * [log_var + (y-μ)²/var]    │ │
                    │  │ ├─ Learns both μ and σ jointly            │ │
                    │  │ └─ Variance is DATA-DEPENDENT (heteroscedastic) │ │
                    │  │                                             │ │
                    │  │ Optimizer: Adam (lr from Optuna)            │ │
                    │  └─────────────────────────────────────────────┘ │
                    └───────────────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────┐
                    │  TRAINING PHASE              │
                    │  ├─ fit(X_train, y_train)   │
                    │  ├─ NLL loss active         │
                    │  ├─ Learn μ and σ           │
                    │  └─ EarlyStopping callback  │
                    └──────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────┐
                    │  SAVE BEST MODEL             │
                    │  (lowest validation NLL)     │
                    └──────────────────────────────┘
                                      │
                    ══════════════════════════════════════════════════════
                   ║  START LAST-LAYER LAPLACE APPROXIMATION (NEW)      ║
                    ══════════════════════════════════════════════════════
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │ PHASE 1: Feature Extraction via      │
                    │          Penultimate Layer           │
                    │                                      │
                    │ phi_model = Model(                  │
                    │   inputs=input_layer,               │
                    │   outputs=Dropout_layer.output      │
                    │ )                                   │
                    │                                      │
                    │ φ_train = phi_model.predict(...)    │
                    │ Shape: (N_train, H)                 │
                    │                                      │
                    │ Extract RNN hidden states           │
                    │ BEFORE the Dense(2) layer           │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │ PHASE 2: MAP Estimation & Noise      │
                    │                                      │
                    │ yhat_train_s = model(X_train)       │
                    │ μ_train_s = yhat_train_s[:, 0]     │
                    │ (extract mean head only)            │
                    │                                      │
                    │ residuals_s = y_true_s - μ_s       │
                    │ σ_n² = mean(residuals_s²)          │
                    │                                      │
                    │ Noise variance in SCALED space      │
                    │ Used for Hessian approximation      │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │ PHASE 3: Hessian Computation         │
                    │                                      │
                    │ Extended features: Φ = [φ, 1]       │
                    │ Shape: (N, H+1)                     │
                    │                                      │
                    │ H_diag[j] = (1/σ_n²) * Σ_i φ²[i,j] │
                    │            + λ_prior                │
                    │                                      │
                    │ Diagonal Hessian (efficient)        │
                    │ Shape: (H+1,)                       │
                    │                                      │
                    │ Posterior variance: Σ = 1/H_diag    │
                    │ var_w_diag Shape: (H+1,)            │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │ PHASE 4: Epistemic Uncertainty       │
                    │                                      │
                    │ For each test sample i:              │
                    │ σ²_epi[i] = Φ_i Σ Φ_i^T            │
                    │           = Σ_j (Φ[i,j]² * var_w[j])│
                    │                                      │
                    │ σ_epi[i] = sqrt(σ²_epi[i] * y_std²) │
                    │                                      │
                    │ Per-sample epistemic uncertainty    │
                    │ High when model is uncertain        │
                    │ (extrapolation, low training data)  │
                    │                                      │
                    │ σ_epi_test Shape: (N_test,)        │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │ PHASE 5: Aleatoric Uncertainty       │
                    │                                      │
                    │ yhat_test = model(X_test)           │
                    │ μ_test = yhat_test[:, 0]           │
                    │ log_var_test = yhat_test[:, 1]     │
                    │ (in SCALED space)                   │
                    │                                      │
                    │ Scale back to original space:       │
                    │ σ²_ale = exp(log_var) * y_std²     │
                    │ σ_ale = sqrt(σ²_ale)                │
                    │                                      │
                    │ Per-sample aleatoric uncertainty    │
                    │ Learned by heteroscedastic model    │
                    │ (heteroscedastic!)                  │
                    │                                      │
                    │ σ_ale_test Shape: (N_test,)        │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │ PHASE 6: Total Uncertainty           │
                    │                                      │
                    │ σ²_total = σ²_ale + σ²_epi         │
                    │ σ_total = sqrt(σ²_total)            │
                    │                                      │
                    │ Combines both uncertainty sources   │
                    │ Used for prediction intervals       │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  PREDICTION INTERVALS                │
                    │  L = μ - 1.96 * σ_total             │
                    │  U = μ + 1.96 * σ_total             │
                    │                                      │
                    │  95% Confidence Interval            │
                    │  (Gaussian approximation)           │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  COMPUTE UQ METRICS                  │
                    │  ├─ PICP: P(y ∈ [L, U])             │
                    │  ├─ MPIW: E[U - L]                  │
                    │  ├─ Winkler: Width + Penalty        │
                    │  └─ Calibration analysis            │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  SAVE RESULTS                        │
                    │  ├─ ALL_UQ_PREDICTED.csv            │
                    │  │  (μ, σ_ale, σ_epi, σ_total,     │
                    │  │   L, U, actual)                  │
                    │  └─ ALL_UQ_METRICS.csv              │
                    │     (PICP, MPIW, Winkler, etc.)     │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                                  OUTPUT: 
                    Predictions + Decomposed UQ Bounds
                            & Metrics
```

---

## 4. KEY HLLLA CONCEPTS FOR RNN (GRU & LSTM)

### **Aleatoric vs Epistemic Uncertainty**

```
ALEATORIC (Data) UNCERTAINTY:
├─ Source: Inherent randomness in data
├─ Examples: Measurement noise, missing features, market noise
├─ Learned by: Heteroscedastic output head (log-variance)
├─ Heteroscedastic: Can vary per sample
└─ Reducible: Only with more/better features, not more data

EPISTEMIC (Model) UNCERTAINTY:
├─ Source: Uncertainty in model parameters/structure
├─ Examples: Limited training data, model capacity, parameter uncertainty
├─ Estimated by: Last-Layer Laplace Approximation
├─ Via: Posterior variance of final layer weights
└─ Reducible: With more training data, better regularization
```

### **Why Last-Layer Laplace?**

```
FULL BAYESIAN INFERENCE:
├─ Goal: Approximate full posterior p(w|D)
├─ Challenge: W = millions of parameters
├─ Complexity: Intractable for large networks

LAST-LAYER LAPLACE:
├─ Insight: Fix early layers, approximate only final layer
├─ Final layer: W_final ≈ H + 1 parameters (H = hidden units)
├─ Much more tractable!
├─ Assumption: Early layer features φ(x) are reliable
└─ Result: Reasonable epistemic uncertainty estimates

DIAGONAL HESSIAN:
├─ Full Hessian: O(H² + H) parameters
├─ Diagonal: O(H) parameters
├─ Trade-off: Less accurate but much faster
└─ Works well in practice for RNNs
```

### **Heteroscedastic Loss vs Homoscedastic MSE**

```
HOMOSCEDASTIC (MSE):
├─ Loss: L = (y - ŷ)²
├─ Assumes: σ² = constant (doesn't depend on x)
├─ Issue: Can't express uncertainty variation
└─ Output: Single value (mean only)

HETEROSCEDASTIC (NLL):
├─ Loss: L = 0.5 * [log(σ²) + (y - μ)²/σ²]
├─ Allows: σ²(x) = varies with input
├─ Benefits: Express per-sample uncertainty
│           Automatically balance fit vs uncertainty
└─ Output: Both mean μ and variance σ²
```

---

## 5. HLLLA vs MCD Comparison

| **Aspect** | **HLLLA** | **MCD** |
|---|---|---|
| **Aleatoric** | Learned via heteroscedastic output | Not directly captured |
| **Epistemic** | Via Laplace approximation on final layer | Via ensemble variance (100 passes) |
| **Inference Cost** | 1 forward pass | 100 forward passes |
| **Calibration** | Explicit heteroscedastic learning | Empirical quantiles |
| **Interpretability** | Clear uncertainty decomposition | Black-box ensemble |
| **Computational** | Fast inference (1 pass) | Slow inference (100 passes) |
| **Theoretical** | Bayesian approximation | Approximate Bayesian |
| **Implementation** | Deterministic + approximation | Stochastic ensemble |

---

## 6. GRU vs LSTM: Identical for HLLLA

| **Component** | **GRU** | **LSTM** |
|---|---|---|
| **Gates** | 2 (reset, update) | 3 (input, forget, output) |
| **Heteroscedastic output** | Dense(2) ✓ | Dense(2) ✓ |
| **NLL loss** | Same ✓ | Same ✓ |
| **Penultimate layer** | Dropout before Dense(2) ✓ | Dropout before Dense(2) ✓ |
| **LLLA computation** | Identical ✓ | Identical ✓ |
| **Feature extraction** | Same φ(x) ✓ | Same φ(x) ✓ |
| **Epistemic/Aleatoric** | Same approach ✓ | Same approach ✓ |

**Conclusion: HLLLA is completely independent of RNN type**

---

## 7. HLLLA Advantages & Challenges

### **Advantages:**
| **Advantage** | **Explanation** |
|---|---|
| **Clear Decomposition** | Separates aleatoric and epistemic uncertainty |
| **Interpretability** | Understand which uncertainty dominates |
| **Fast Inference** | Only 1 forward pass per prediction |
| **Learned Heteroscedasticity** | Per-sample uncertainty adaptation |
| **Bayesian Principled** | Based on Laplace approximation |
| **No Ensemble Overhead** | No need for 100 passes like MCD |

### **Challenges:**
| **Challenge** | **Explanation** |
|---|---|
| **Diagonal Hessian** | Diagonal approximation loses correlations |
| **Final Layer Only** | Doesn't capture early layer uncertainty |
| **Gaussian Assumption** | LLLA assumes Gaussian posterior |
| **Hyperparameter (λ)** | Prior precision needs tuning |
| **NLL Training** | Different loss than standard MSE; harder tuning |
| **Noise Variance Est.** | Estimated from training residuals (may be biased) |

---

## Summary

**HLLLA in GRU/LSTM applies through:**

1. **Heteroscedastic Output** (Dense(2)) - Learn mean AND variance
2. **Heteroscedastic NLL Loss** - Joint optimization of μ and σ
3. **Optuna HPO** (50 trials) - Find optimal hyperparameters
4. **Training** - Standard fit with NLL loss
5. **Feature Extraction** - Get φ(x) from penultimate layer
6. **MAP Estimation** - Extract mean predictions and noise variance
7. **Laplace Approximation** - Diagonal Hessian of final layer
8. **Epistemic Uncertainty** - Posterior variance of weight distribution
9. **Aleatoric Uncertainty** - Learned per-sample variance from model
10. **Combined Uncertainty** - σ_total = √(σ_ale² + σ_epi²)
11. **Prediction Intervals** - [μ - 1.96σ, μ + 1.96σ]
12. **UQ Evaluation** - PICP, MPIW, Winkler Score

The RNN architecture (GRU vs LSTM) is **completely irrelevant** to HLLLA implementation. Both follow identical uncertainty quantification procedures. The choice between GRU and LSTM should be based on modeling performance, not on UQ methodology.

HLLLA provides **explicit uncertainty decomposition** and **fast inference** compared to MCD, making it ideal for production systems requiring both interpretability and speed.
