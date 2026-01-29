# Conformal Quantile Regression (CQR) in Transformer Pipeline: Detailed Analysis

## Overview
The Transformer_CQR_42.ipynb implements **Conformal Quantile Regression (CQR)** for uncertainty quantification in time-series forecasting using Transformer architecture. This method combines:
- **Training phase:** Ensemble of M independent Transformer models learning three quantiles (lower, median, upper) via pinball loss
- **Validation phase:** Split conformal calibration to compute a non-conformity threshold
- **Inference phase:** Apply the threshold to ensemble quantile predictions for formal coverage guarantees

CQR provides **distribution-free uncertainty quantification** with guaranteed marginal coverage at target significance level α (e.g., 95% PI for α=0.05).

---

## 1. WHERE CQR IS APPLIED IN THE TRANSFORMER PIPELINE

### **Stage 1: Quantile Targets Definition (Lines ~25-26)**
**File Location:** Configuration section

```python
ALPHA = 0.05                                    # Significance level
TAUS  = [ALPHA/2, 0.5, 1-ALPHA/2]             # [0.025, 0.5, 0.975]
                                                # ↓ 3 quantiles for 95% PI
```

**Mathematical Interpretation:**
```
τ₁ = α/2 = 0.025      → Lower quantile (2.5th percentile)
τ₂ = 0.5              → Median (50th percentile)
τ₃ = 1-α/2 = 0.975    → Upper quantile (97.5th percentile)

These 3 quantiles define the prediction interval [Q₀.₀₂₅, Q₀.₉₇₅]
with nominal coverage 1-α = 95%
```

---

### **Stage 2: Quantile Output Head Architecture (Lines ~54-75)**
**File Location:** `build_transformer_quantile()` function

```python
def build_transformer_quantile(lookback, n_features, d_model, num_heads, 
                                dff, num_layers, dropout, lr):
    """
    Transformer with quantile regression output head
    """
    inp = Input(shape=(lookback, n_features))
    
    # Standard Transformer encoder blocks
    x = Dense(d_model)(inp)                           # Project to d_model
    x = SinusoidalPositionalEncoding(d_model)(x)    # Position info
    
    for _ in range(num_layers):
        x = encoder_block(
            x,
            num_heads=num_heads,
            d_model=d_model,
            dff=dff,
            dropout_rate=dropout
        )
    
    # Extract last timestep: (batch, lookback, d_model) → (batch, d_model)
    x = Lambda(lambda t: t[:, -1, :])(x)
    
    # ═══════════════════════════════════════════════════════════════════
    # CRITICAL: Quantile output head
    # Instead of Dense(1) or Dense(2), use Dense(len(TAUS)) = Dense(3)
    # Output 3 values: [Q₀.₀₂₅, Q₀.₅, Q₀.₉₇₅] in SCALED space
    # ═══════════════════════════════════════════════════════════════════
    out = Dense(len(TAUS))(x)  # len(TAUS) = 3
    
    # → Shape: (batch, 3) for 3 quantile predictions
    
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss=None)  # Loss will be set per compilation
    
    return model
```

**Output Structure:**
```
For each sample:
  out[i, 0] = Q̂₀.₀₂₅(x_i) in scaled space (predicted lower quantile)
  out[i, 1] = Q̂₀.₅(x_i)   in scaled space (predicted median)
  out[i, 2] = Q̂₀.₉₇₅(x_i) in scaled space (predicted upper quantile)
```

---

### **Stage 3: Pinball Loss for Multi-Quantile Learning (Lines ~77-88)**
**File Location:** `pinball_multi()` function

```python
def pinball_multi(taus):
    """
    Multi-quantile pinball loss (asymmetric loss)
    
    Mathematical formulation:
    Loss = (1/N) Σ Σ_j [ρ_τⱼ(y - Q̂_τⱼ)]
    
    where ρ_τ(e) = max(τ × e, (τ-1) × e)
                 = τ × e if e ≥ 0 (prediction too low)
                 = (τ-1) × e if e < 0 (prediction too high)
    
    This loss encourages:
    - Q̂_τ ≤ y with low weight τ    (lower quantile penalties low)
    - Q̂_τ ≥ y with high weight 1-τ (upper quantile penalties high)
    """
    taus = tf.constant(taus, dtype=tf.float32)  # [0.025, 0.5, 0.975]
    
    def loss(y_true, y_pred):
        # y_true: (batch,)
        # y_pred: (batch, 3) - the 3 quantile predictions
        
        y_t = tf.expand_dims(y_true, axis=-1)   # (batch, 1)
        e   = y_t - y_pred                       # (batch, 3) - errors per quantile
        
        # Pinball loss per quantile
        l   = tf.maximum(taus * e, (taus - 1.0) * e)  # (batch, 3)
        
        # Average over batch and quantiles
        return tf.reduce_mean(tf.reduce_sum(l, axis=-1))
    
    return loss

def compile_member():
    """
    Build and compile a single ensemble member
    """
    m = build_transformer_quantile(
        lookback=BEST["lookback"],
        n_features=len(feature_cols),
        d_model=BEST["d_model"],
        num_heads=BEST["num_heads"],
        dff=BEST["dff"],
        num_layers=BEST["num_layers"],
        dropout=BEST["dropout"],
        lr=BEST["lr"]
    )
    
    # Compile with pinball loss
    m.compile(
        optimizer=optimizers.Adam(learning_rate=BEST["lr"]),
        loss=pinball_multi(TAUS)  # TAUS = [0.025, 0.5, 0.975]
    )
    
    return m
```

**Pinball Loss Intuition:**
```
Quantile τ = 0.025 (lower):
  Error > 0 (prediction too low):  Loss = 0.025 × error    [small penalty]
  Error < 0 (prediction too high): Loss = -0.975 × error   [large penalty]
  → Model learns to predict low with acceptable underprediction

Quantile τ = 0.5 (median):
  Error > 0: Loss = 0.5 × error
  Error < 0: Loss = -0.5 × error
  → Symmetric; standard median regression

Quantile τ = 0.975 (upper):
  Error > 0: Loss = 0.975 × error   [large penalty]
  Error < 0: Loss = -0.025 × error  [small penalty]
  → Model learns to predict high with acceptable overprediction
```

---

### **Stage 4: Ensemble Training via Bootstrap (Lines ~90-107)** ⭐ **ENSEMBLE CREATION**
**File Location:** Ensemble training loop

```python
def bootstrap_idx(n, seed):
    """
    Generate bootstrap indices for sampling with replacement
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=n)  # Sample n indices from [0, n) with replacement

def train_member(seed):
    """
    Train a single ensemble member with bootstrap data
    
    seed: RNG seed for reproducibility and diversity
    """
    print(f"\nTraining member with seed {seed} ...")
    
    # Set random seeds for reproducibility within this member
    tf.keras.utils.set_random_seed(seed)
    
    # Build and compile model
    member = compile_member()
    
    if BOOTSTRAP:
        # ═════════════════════════════════════════════════════════════
        # BOOTSTRAP SAMPLING: Create diversity via different data subsets
        # ═════════════════════════════════════════════════════════════
        idx_bs = bootstrap_idx(len(X_train_w), seed + 777)  # Bootstrap indices
        Xb, yb = X_train_w[idx_bs], y_train_w[idx_bs]       # Bootstrapped training data
        
        # Note: Some samples will be repeated, some omitted
        # → Creates different training data for each member
    else:
        # Use full training data (no bootstrap)
        Xb, yb = X_train_w, y_train_w
    
    # Train with early stopping and checkpointing
    cbs = [
        EarlyStopping(monitor="val_loss", patience=BEST["patience"], 
                     restore_best_weights=True),
        ModelCheckpoint(
            f"Model Checkpoints/transformer_cqr_member_{seed}.keras",
            monitor="val_loss",
            save_best_only=True
        )
    ]
    
    member.fit(
        Xb, yb,
        validation_data=(X_val_w, y_val_w),
        epochs=BEST["epochs"],
        batch_size=BEST["batch_size"],
        verbose=1,
        callbacks=cbs
    )
    
    return member

# ═══════════════════════════════════════════════════════════════════
# TRAIN ENSEMBLE: M independent members
# ═══════════════════════════════════════════════════════════════════
M_ENSEMBLE = 5
SEED_BASE = 42

print(f"\nTraining Transformer+CQR ensemble with M={M_ENSEMBLE} ...")
t0_train = time.time()

members = [train_member(SEED_BASE + 137*i) for i in range(M_ENSEMBLE)]
#           ↑ seed = 42, 179, 316, 453, 590 (different for each member)

t1_train = time.time()
print(f"Training time: {t1_train - t0_train:.4f} seconds")
```

**Key Points:**
```
Ensemble Diversity Mechanisms:
  1. Different random seeds → Different weight initialization
  2. Bootstrap data → Different training samples
  3. Stochastic optimization → Different convergence paths
  
Result: M = 5 diverse models, each predicting 3 quantiles
```

---

### **Stage 5: Ensemble Quantile Prediction (Lines ~109-145)** ⭐ **ENSEMBLE AGGREGATION**
**File Location:** `predict_all()` function

```python
def predict_quantiles_member(m, X, idx):
    """
    Get quantile predictions from a single member and inverse-scale
    """
    # Forward pass: (N, 3) standardized quantiles
    qz = m.predict(X, verbose=1)  # Shape: (N, 3)
    
    # Inverse scale from standardized to original space
    # q_orig = q_scaled × y_scale + y_mean
    qy = qz * y_scale + y_mean    # Shape: (N, 3)
    
    # Enforce monotonicity: L ≤ M ≤ U
    # (Quantile regression might violate ordering due to optimization)
    qy = np.sort(qy, axis=1)      # Sort each row independently
    
    # Return individual quantile series
    return (
        pd.Series(qy[:, 0], index=idx),  # Lower quantile (0.025)
        pd.Series(qy[:, 1], index=idx),  # Median (0.5)
        pd.Series(qy[:, 2], index=idx),  # Upper quantile (0.975)
    )

def predict_all(members, X, idx):
    """
    Aggregate predictions from all M ensemble members
    
    members: List of M trained Transformer models
    X: Input data (N, lookback, n_features)
    idx: Time indices for output alignment
    """
    Ls, Ms, Us = [], [], []  # Lists to collect predictions from all members
    
    # ═════════════════════════════════════════════════════════════
    # STEP 1: Get predictions from each member
    # ═════════════════════════════════════════════════════════════
    for m in members:
        qL, qM, qU = predict_quantiles_member(m, X, idx)
        Ls.append(qL.values)    # (N,) array
        Ms.append(qM.values)    # (N,) array
        Us.append(qU.values)    # (N,) array
    
    # Stack into matrices: shape (N, M)
    Ls = np.stack(Ls, axis=1)  # (N_samples, M_ensemble)
    Ms = np.stack(Ms, axis=1)  # (N_samples, M_ensemble)
    Us = np.stack(Us, axis=1)  # (N_samples, M_ensemble)
    
    # ═════════════════════════════════════════════════════════════
    # STEP 2: Average quantiles across members
    # ═════════════════════════════════════════════════════════════
    L_bar = pd.Series(Ls.mean(axis=1), index=idx, name="qL_bar")      # Average lower
    M_bar = pd.Series(Ms.mean(axis=1), index=idx, name="qM_bar")      # Average median
    U_bar = pd.Series(Us.mean(axis=1), index=idx, name="qU_bar")      # Average upper
    
    # ═════════════════════════════════════════════════════════════
    # STEP 3: Estimate epistemic uncertainty (across-member variance)
    # ═════════════════════════════════════════════════════════════
    # Epistemic: How much do ensemble members disagree on the median?
    var_epi = pd.Series(
        Ms.var(axis=1, ddof=1) if Ms.shape[1] > 1 else np.zeros(Ms.shape[0]),
        index=idx,
        name="var_epistemic"
    )
    
    # ═════════════════════════════════════════════════════════════
    # STEP 4: Estimate aleatoric uncertainty (from interval width)
    # ═════════════════════════════════════════════════════════════
    # Aleatoric: Interquartile range → approximate as ≈3.92 × σ for Gaussian
    # IQR = Q0.975 - Q0.025 ≈ 1.96 × σ + 1.96 × σ = 3.92 × σ
    IQR = U_bar.values - L_bar.values
    sigma_alea = IQR / 3.92                           # Approximate σ from IQR
    var_alea = pd.Series(
        np.maximum(sigma_alea, 0) ** 2,              # Ensure non-negative
        index=idx,
        name="var_aleatoric"
    )
    
    return L_bar, M_bar, U_bar, var_epi, var_alea

# ═════════════════════════════════════════════════════════════
# Predict on all splits
# ═════════════════════════════════════════════════════════════
t0_test = time.time()

L_tr_bar, M_tr_bar, U_tr_bar, var_epi_tr, var_alea_tr = predict_all(
    members, X_train_w, idx_train
)
L_v_bar, M_v_bar, U_v_bar, var_epi_v, var_alea_v = predict_all(
    members, X_val_w, idx_val
)
L_te_bar, M_te_bar, U_te_bar, var_epi_te, var_alea_te = predict_all(
    members, X_test_w, idx_test
)

t1_test = time.time()
print(f"Testing time: {t1_test - t0_test:.4f} seconds")
```

---

### **Stage 6: Split Conformal Calibration (Lines ~147-160)** ⭐ **FORMAL COVERAGE GUARANTEE**
**File Location:** Conformal threshold computation

```python
# ═════════════════════════════════════════════════════════════════════════════
# CONFORMAL CALIBRATION: Compute non-conformity threshold on validation set
# ═════════════════════════════════════════════════════════════════════════════

# Step 1: Compute non-conformity scores on validation set
# Non-conformity = How far actual is from the predicted interval

E_val = np.maximum(
    L_v_bar.values - actual_val.values,           # How much below lower bound?
    actual_val.values - U_v_bar.values            # How much above upper bound?
)
E_val = np.maximum(E_val, 0.0)                    # Non-negative distances

print(f"Non-conformity scores (validation):")
print(f"  Min: {E_val.min():.4f}")
print(f"  Max: {E_val.max():.4f}")
print(f"  Mean: {E_val.mean():.4f}")

# Step 2: Compute threshold q̂ such that coverage ≥ 1 - α
# q̂ = ⌈(n+1)(1-α)/n⌉-th smallest non-conformity score
# For asymptotic purposes: q̂ ≈ (1-α)-quantile of E_val

q_hat = np.quantile(E_val, 1 - ALPHA, method="higher")
# Alternative: np.percentile(E_val, (1-ALPHA)*100, interpolation="higher")

print(f"\nConformal threshold q̂ = {q_hat:.6f}")
print(f"  (ensures ≥ {1-ALPHA:.0%} coverage with high probability)")

# ═════════════════════════════════════════════════════════════════════════════
# Step 3: Conformalize predictions on all splits
# Conformalized interval = [L̂ - q̂, Û + q̂]
# ═════════════════════════════════════════════════════════════════════════════

# Training set: Conformalize
L_train = L_tr_bar - q_hat      # Expand lower bound downward
U_train = U_tr_bar + q_hat      # Expand upper bound upward

# Validation set: Conformalize
L_val = L_v_bar - q_hat
U_val = U_v_bar + q_hat

# Test set: Conformalize (this is where we evaluate)
L_test = L_te_bar - q_hat
U_test = U_te_bar + q_hat

# Point forecasts = ensemble median
mean_train = M_tr_bar.rename("point_pred")
mean_val   = M_v_bar.rename("point_pred")
mean_test  = M_te_bar.rename("point_pred")

print(f"\nConformal intervals:")
print(f"  Width on test (before adjustment): {(U_te_bar - L_te_bar).mean():.4f}")
print(f"  Width on test (after adjustment):  {(U_test - L_test).mean():.4f}")
```

**Conformal Guarantee (Distribution-Free):**

$$P(y_{n+1} \in [L_{n+1} - \hat{q}, U_{n+1} + \hat{q}]) \geq 1 - \frac{\lceil (n+1)(1-\alpha) \rceil}{n+1}$$

where:
- $n$ = validation set size
- $\hat{q}$ = $(1-\alpha)$-quantile of non-conformity scores
- Guarantee holds for **any** distribution (distribution-free)
- Only assumes **exchangeability** of data

---

### **Stage 7: UQ Metrics Computation (Lines ~162-188)**
**File Location:** Metrics evaluation

```python
def uq_metrics(y_true, L, U, alpha=ALPHA):
    """
    Compute uncertainty quantification metrics
    """
    y = np.asarray(y_true); L = np.asarray(L); U = np.asarray(U)
    
    # 1) PICP: Prediction Interval Coverage Probability
    cover = (y >= L) & (y <= U)
    picp = cover.mean()  # Fraction of observations within interval
    
    # 2) MPIW: Mean Prediction Interval Width
    mpiw = np.mean(U - L)  # Average width
    
    # 3) Winkler Score: Width + penalty for misses
    width = U - L
    penalty = np.zeros_like(y)
    
    below = y < L
    above = y > U
    
    # Penalize misses: (2/α) × distance outside interval
    penalty[below] = (L[below] - y[below])
    penalty[above] = (y[above] - U[above])
    
    winkler = np.mean(width + (2.0 / alpha) * penalty)
    
    return dict(PICP=picp, MPIW=mpiw, Winkler=winkler)

# Compute metrics for all splits
print("\n=== UQ Metrics (95% PI) — CQR (conformalized) ===")
print("Train:", uq_metrics(actual_train.values, L_train.values, U_train.values, ALPHA))
print("Val:  ", uq_metrics(actual_val.values,   L_val.values,   U_val.values,   ALPHA))
print("Test: ", uq_metrics(actual_test.values,  L_test.values,  U_test.values,  ALPHA))
```

---

## 2. SUMMARY TABLE: WHERE CQR APPLIES IN TRANSFORMER

| **Stage** | **Component** | **CQR Role** | **Code Location** |
|-----------|---------------|---|---|
| **Quantile Targets** | TAUS = [0.025, 0.5, 0.975] | Define 3-quantile outputs | Lines ~25-26 |
| **Output Head** | Dense(3) instead of Dense(1/2) | Predict 3 quantiles | Lines ~70-73 |
| **Pinball Loss** | Asymmetric loss per quantile | Learn quantiles directly | Lines ~77-88 |
| **Ensemble Creation** | M=5 independent members | Diversity via bootstrap + seeds | Lines ~90-107 |
| **Bootstrap Sampling** | Sample with replacement | Create diverse training sets | Lines ~101-103 |
| **Training** | fit() with pinball loss | Learn quantile predictions | Lines ~104-118 |
| **Single Member Prediction** | predict_quantiles_member() | Get 3 quantiles per member | Lines ~120-132 |
| **Ensemble Aggregation** | Average across M members | L̄, M̄, Ū ensemble predictions | Lines ~146-152 |
| **Uncertainty Estimation** | Epistemic & aleatoric proxies | Decompose uncertainty | Lines ~154-165 |
| **Conformal Calibration** | Compute q̂ on validation | Non-conformity threshold | Lines ~167-180 |
| **Conformalization** | L - q̂, U + q̂ | Expand intervals for coverage | Lines ~182-191 |
| **UQ Metrics** | PICP, MPIW, Winkler | Evaluate calibration & sharpness | Lines ~193-220 |

---

## 3. CQR FLOWCHART FOR TRANSFORMER

```
┌─────────────────────────────────────────────────────────────────────────┐
│  TRANSFORMER + CONFORMAL QUANTILE REGRESSION (CQR) WITH ENSEMBLE        │
└─────────────────────────────────────────────────────────────────────────┘

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
                    │  Define Quantile Targets     │
                    │  ├─ ALPHA = 0.05 (5%)       │
                    │  └─ TAUS = [0.025, 0.5,     │
                    │            0.975]           │
                    │     (2.5th, 50th, 97.5th)   │
                    │                              │
                    │  For 95% Prediction Interval │
                    └──────────────────────────────┘
                                      │
                    ══════════════════════════════════════════════
                   ║  BUILD TRANSFORMER + QUANTILE HEAD           ║
                    ══════════════════════════════════════════════
                                      │
                                      ▼
                    ┌───────────────────────────────────────────────┐
                    │  Transformer Architecture                     │
                    │  ┌─────────────────────────────────────────┐ │
                    │  │ INPUT (batch, lookback, n_features)    │ │
                    │  │           ↓                             │ │
                    │  │ Dense(d_model) → Positional Encoding   │ │
                    │  │           ↓                             │ │
                    │  │ FOR each encoder layer (num_layers):   │ │
                    │  │   MultiHeadAttention + Dropout + LN    │ │
                    │  │   Feed-Forward + Dropout + LN          │ │
                    │  │           ↓                             │ │
                    │  │ Extract last timestep (batch, d_model) │ │
                    │  │           ↓                             │ │
                    │  │ Dense(3) OUTPUT ◄─ QUANTILE HEAD       │ │
                    │  │ ├─ out[:, 0] = Q̂₀.₀₂₅ (scaled)        │ │
                    │  │ ├─ out[:, 1] = Q̂₀.₅ (scaled)          │ │
                    │  │ └─ out[:, 2] = Q̂₀.₉₇₅ (scaled)        │ │
                    │  │                                         │ │
                    │  │ Loss: Pinball (multi-quantile)          │ │
                    │  │ Optimizer: Adam                         │ │
                    │  └─────────────────────────────────────────┘
                    └───────────────────────────────────────────────┘
                                      │
                    ══════════════════════════════════════════════════════
                   ║  ENSEMBLE TRAINING (M MEMBERS WITH BOOTSTRAP)      ║
                    ══════════════════════════════════════════════════════
                                      │
                                      ▼
                    ┌──────────────────────────────────────────┐
                    │  FOR each ensemble member i=1 to M:     │
                    │                                          │
                    │  ├─ Set seed = SEED_BASE + 137×i        │
                    │  │   (ensures reproducibility + diversity)│
                    │  │                                        │
                    │  ├─ IF BOOTSTRAP:                        │
                    │  │    Generate bootstrap indices         │
                    │  │    Xb, yb = resample(X_train, y_train)│
                    │  │    (sample with replacement)          │
                    │  │ ELSE:                                 │
                    │  │    Xb, yb = X_train, y_train          │
                    │  │                                        │
                    │  ├─ Build Transformer (same arch)        │
                    │  │                                        │
                    │  ├─ Compile with pinball_multi(TAUS)     │
                    │  │                                        │
                    │  ├─ Train:                               │
                    │  │   fit(Xb, yb,                         │
                    │  │       validation=(X_val, y_val),      │
                    │  │       epochs, callbacks=[ES, MC])     │
                    │  │                                        │
                    │  └─ Save trained member                  │
                    │                                          │
                    │  Result: M diverse models, each          │
                    │  predicting 3 quantiles                  │
                    └──────────────────────────────────────────┘
                                      │
                    ══════════════════════════════════════════════════════
                   ║  ENSEMBLE QUANTILE PREDICTION (ALL SPLITS)         ║
                    ══════════════════════════════════════════════════════
                                      │
                                      ▼
                    ┌──────────────────────────────────────────┐
                    │  FOR each split (train/val/test):       │
                    │                                          │
                    │  ┌──────────────────────────────────┐   │
                    │  │ STEP 1: Member-level prediction │   │
                    │  │                                  │   │
                    │  │ FOR each member m:               │   │
                    │  │   ŷ_m = m.predict(X)  (N, 3)   │   │
                    │  │   ├─ Inverse scale              │   │
                    │  │   ├─ Enforce monotonicity       │   │
                    │  │   │   (sort each row)           │   │
                    │  │   └─ Store [L_m, M_m, U_m]     │   │
                    │  │                                  │   │
                    │  │ Collect from all M members:     │   │
                    │  │   L_matrix (N, M)               │   │
                    │  │   M_matrix (N, M)               │   │
                    │  │   U_matrix (N, M)               │   │
                    │  └──────────────────────────────────┘   │
                    │                                          │
                    │  ┌──────────────────────────────────┐   │
                    │  │ STEP 2: Ensemble aggregation     │   │
                    │  │                                  │   │
                    │  │ L̄ = mean(L_matrix, axis=1)     │   │
                    │  │ M̄ = mean(M_matrix, axis=1)     │   │
                    │  │ Ū = mean(U_matrix, axis=1)     │   │
                    │  │                                  │   │
                    │  │ (Take mean across all members)  │   │
                    │  └──────────────────────────────────┘   │
                    │                                          │
                    │  ┌──────────────────────────────────┐   │
                    │  │ STEP 3: Uncertainty estimation  │   │
                    │  │                                  │   │
                    │  │ var_epi = var(M_matrix, axis=1) │   │
                    │  │   (agreement on median)          │   │
                    │  │                                  │   │
                    │  │ IQR = Ū - L̄                     │   │
                    │  │ σ_alea ≈ IQR / 3.92              │   │
                    │  │ var_alea = σ_alea²               │   │
                    │  │   (from interval width)          │   │
                    │  └──────────────────────────────────┘   │
                    └──────────────────────────────────────────┘
                                      │
                    ══════════════════════════════════════════════════════
                   ║  SPLIT CONFORMAL CALIBRATION (VALIDATION)          ║
                    ══════════════════════════════════════════════════════
                                      │
                                      ▼
                    ┌──────────────────────────────────────────┐
                    │  Compute non-conformity on VALIDATION:  │
                    │                                          │
                    │  E_i = max(L̄_i - y_i, y_i - Ū_i, 0)   │
                    │                                          │
                    │  (How far is y from interval [L̄, Ū]?)  │
                    │                                          │
                    │  Collect: E_val = [E_1, E_2, ..., E_n] │
                    │           Length n = |validation set|   │
                    │                                          │
                    │  ├─ E_i = 0 if y_i ∈ [L̄_i, Ū_i]        │
                    │  └─ E_i > 0 if y_i outside interval    │
                    └──────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────────┐
                    │  Compute threshold q̂:                  │
                    │                                          │
                    │  q̂ = quantile(E_val, 1 - ALPHA)        │
                    │     = (1-α)-quantile of E_val          │
                    │     = (1-0.05)-quantile                │
                    │     = 0.95-quantile                    │
                    │                                          │
                    │  For α=0.05: q̂ ≈ 95th percentile      │
                    │                                          │
                    │  ← CONFORMAL THRESHOLD                 │
                    │  Guarantees coverage ≥ 1-α on test     │
                    └──────────────────────────────────────────┘
                                      │
                    ══════════════════════════════════════════════════════
                   ║  CONFORMALIZE PREDICTIONS (ALL SPLITS)              ║
                    ══════════════════════════════════════════════════════
                                      │
                                      ▼
                    ┌──────────────────────────────────────────┐
                    │  Expand intervals by threshold q̂:      │
                    │                                          │
                    │  L_conform = L̄ - q̂  ← Shift down      │
                    │  U_conform = Ū + q̂  ← Shift up        │
                    │                                          │
                    │  Conformalized interval:                │
                    │  [L_conform, U_conform]                 │
                    │                                          │
                    │  Applies to:                            │
                    │  ├─ Training set                        │
                    │  ├─ Validation set                      │
                    │  └─ Test set ← MAIN EVALUATION         │
                    │                                          │
                    │  This guarantees:                       │
                    │  P(y ∈ [L, U]) ≥ 1 - α   (coverage)    │
                    └──────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────────┐
                    │  Point forecasts:                       │
                    │  μ = M̄ (ensemble median)               │
                    │                                          │
                    │  Outputs for each sample:              │
                    │  ├─ μ: Point prediction                │
                    │  ├─ L_conform: Lower PI bound         │
                    │  ├─ U_conform: Upper PI bound         │
                    │  ├─ var_epi: Ensemble disagreement    │
                    │  └─ var_alea: Interval width          │
                    └──────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────────┐
                    │  Compute UQ Metrics:                    │
                    │                                          │
                    │  PICP = P(y ∈ [L, U]) on test         │
                    │    → Should be ≥ 95% for α=0.05       │
                    │                                          │
                    │  MPIW = E[U - L] on test              │
                    │    → Measure sharpness (lower better)  │
                    │                                          │
                    │  Winkler = E[width + penalty]          │
                    │    → Combined calibration metric       │
                    │                                          │
                    │  Coverage plot, heatmap, rolling PICP  │
                    └──────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────────┐
                    │  Save Results:                          │
                    │  ├─ ALL_UQ_PREDICTED.csv               │
                    │  │  (μ, L, U per timestamp)            │
                    │  └─ ALL_UQ_METRICS.csv                 │
                    │     (PICP, MPIW, Winkler, times)      │
                    └──────────────────────────────────────────┘
                                      │
                                      ▼
                                   OUTPUT:
                          Predictions + UQ Bounds
                        with Formal Coverage Guarantee
```

---

## 4. KEY CQR CONCEPTS FOR TRANSFORMER

### **Quantile Regression vs Standard Regression**

```
Standard (MSE/MAE):
  ŷ = argmin E[(y - ŷ)²]
  → Minimizes MSE → optimal for mean prediction
  → Single point output

Quantile Regression (Pinball):
  Q̂_τ = argmin E[ρ_τ(y - Q̂_τ)]
  → Predicts specific quantiles
  → Multiple outputs (one per quantile τ)
  → Asymmetric loss reflects quantile probability

For our CQR:
  Q̂₀.₀₂₅ = lower 2.5th percentile
  Q̂₀.₅₀₀ = median 50th percentile
  Q̂₀.₉₇₅ = upper 97.5th percentile
```

### **Ensemble Diversity via Bootstrap**

```
Member 1 (seed=42):
  Data: resample with replacement from training set
  Initialize: Random weights (seed=42)
  Train: Converge with different dynamics
  → Model A

Member 2 (seed=179):
  Data: DIFFERENT bootstrap sample
  Initialize: Random weights (seed=179)
  Train: Different convergence path
  → Model B

...Member 5 (seed=590)
  → Model E

Ensemble = [Model A, Model B, C, D, E]

Diversity sources:
  1. Different training data (bootstrap)
  2. Different weight initialization (seeds)
  3. Stochastic gradient descent (different paths)
  
Result: 5 diverse quantile regression models
```

### **Conformal Prediction Guarantee**

```
Distribution-Free Property:
  ├─ NO assumption on data distribution
  ├─ NO model assumptions (non-parametric)
  └─ Works for ANY supervised learning model

Only requirement: EXCHANGEABILITY of data

Coverage Guarantee (Finite-Sample):
  P(y ∈ [L - q̂, U + q̂]) ≥ 1 - α - δ
  
  where:
  ├─ q̂ = (1-α)-quantile of validation non-conformity
  ├─ α = target significance (0.05 for 95% PI)
  ├─ δ = ⌈(n+1)(1-α)⌉/(n+1) ≈ 0 for large n
  └─ n = validation set size

Practical:
  For n=200, α=0.05:
  ├─ Target coverage: 95%
  ├─ Guaranteed coverage: ≥ 95% (high probability)
  └─ In practice: Coverage ≈ 95.5-97% on test
```

---

## 5. CQR VS MCD VS HLLLA IN TRANSFORMER

| **Aspect** | **MCD** | **HLLLA** | **CQR** |
|---|---|---|---|
| **Output** | Dense(1) | Dense(2) | Dense(3) |
| **Architecture** | Single model | Single model | M ensemble models |
| **Training Data** | Full training set | Full training set | Bootstrap samples |
| **Loss Function** | MSE | Gaussian NLL | Pinball (quantile) |
| **Uncertainty Learning** | Via dropout ensemble | Heteroscedastic + LLLA | Direct quantile prediction |
| **Inference Cost** | 100× passes | 1 pass + LLLA | M forward passes (M=5) |
| **Distribution Assumption** | Assumes dropout = Bayes | Assumes Gaussian | Distribution-free |
| **Coverage Type** | Empirical (from ensemble) | From Gaussian CDF | Formal conformal guarantee |
| **Theoretical Basis** | Bayesian approximation | Laplace approximation | Distribution-free inference |
| **Best For** | General-purpose UQ | Explicit decomposition | Guaranteed coverage |
| **Failure Mode** | Miscalibrated intervals | Gaussian assumption violation | None (coverage guaranteed) |

---

## 6. KEY ADVANTAGES OF CQR

```
✓ Distribution-Free Coverage:
  ├─ No parametric assumptions
  ├─ Works for ANY data distribution
  └─ Provable guarantee on actual coverage

✓ Formal Statistical Guarantee:
  ├─ Coverage ≥ 1-α with high probability
  ├─ Finite-sample guarantee (not asymptotic)
  └─ Accounts for estimation error

✓ Direct Quantile Learning:
  ├─ Pinball loss encourages correct quantiles
  ├─ No need to invert CDF
  └─ Natural for asymmetric intervals

✓ Flexible Ensemble:
  ├─ Any number of models (M=5, 10, 100)
  ├─ Bootstrap creates diversity
  └─ Robust to individual model failures

✓ Post-Hoc Calibration:
  ├─ No retraining needed
  ├─ Can adjust α on new data
  └─ Simple threshold q̂
```

---

## 7. KEY CHALLENGES OF CQR

```
✗ Computational Cost:
  ├─ M ensemble members to train
  ├─ M forward passes for inference
  ├─ Higher cost than single-model methods

✗ Quantile Ordering:
  ├─ Individual members might violate Q_L ≤ Q_M ≤ Q_U
  ├─ Need to sort predictions
  └─ Averaging can create issues (solution: sort first)

✗ Exchangeability Assumption:
  ├─ Data must be exchangeable (no temporal drift)
  ├─ Time series violate this assumption
  ├─ Can use sliding-window calibration
  └─ Still less robust than MCD/HLLLA for time series

✗ Ensemble Diversity Trade-off:
  ├─ More diverse → better coverage but wider PI
  ├─ Bootstrap with replacement reduces diversity
  └─ Need balance between diversity and sharpness
```

---

## Summary

**CQR in Transformer applies through:**

1. **Quantile Output Head** (Dense(3)) - Predict 3 quantiles
2. **Pinball Loss** - Learn asymmetric quantile regression
3. **Ensemble Training** (M=5 members) - Bootstrap + different seeds
4. **Ensemble Prediction** - Average quantiles across members
5. **Uncertainty Estimation** - Epistemic + aleatoric proxies
6. **Conformal Calibration** - Compute threshold on validation
7. **Conformalization** - Expand intervals by threshold
8. **Coverage Guarantee** - Formal statistical guarantee ≥ 95%

The method provides **distribution-free uncertainty quantification** with **guaranteed coverage** - no assumptions needed about data distribution, only exchangeability. Ideal when you need theoretical guarantees, though at higher computational cost than single-model alternatives.
