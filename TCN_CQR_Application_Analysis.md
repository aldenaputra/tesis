# Conformal Quantile Regression (CQR) in TCN Pipeline: Detailed Analysis

## Overview
The TCN_CQR_42.ipynb implements **Conformal Quantile Regression (CQR)** for uncertainty quantification in time-series forecasting. This method combines:
1. **Quantile Regression**: Model learns multiple quantiles (lower, median, upper) instead of point estimate
2. **Ensemble Learning**: Train M independent models with bootstrap sampling
3. **Split Conformal Calibration**: Use validation data to calibrate intervals for guaranteed coverage

---

## 1. WHERE CQR IS APPLIED IN THE TCN PIPELINE

### **Stage 1: Configuration & Quantile Setup (Lines ~17-34)**
**File Location:** Hyperparameters and quantile definitions

```python
# Quantile levels for regression
ALPHA = 0.05                  # Significance level (95% coverage target)
TAUS = [ALPHA/2, 0.5, 1-ALPHA/2]  # [0.025, 0.5, 0.975]
                              # Lower, Median, Upper quantiles

# Ensemble settings
M_ENSEMBLE = 5                # Number of ensemble members
BOOTSTRAP = True              # Bootstrap resampling for each member
SEED_BASE = 42

# Visualization settings
ROLL_LEN   = 30
HEAT_WIN   = 30
HEAT_STRIDE = 10
```

**Key Point:** CQR requires predicting **3 quantiles** (or more) instead of a single point estimate.

---

### **Stage 2: TCN Architecture with Quantile Head (Lines ~65-86)**
**File Location:** `build_tcn_quantile()` function

```python
def build_tcn_quantile(lookback, n_features, filters, kernel_size, dropout, 
                       dilations, num_stacks, lr, use_layernorm=True):
    inp = Input(shape=(lookback, n_features))
    x = inp
    
    # Standard TCN blocks
    for _ in range(num_stacks):
        for d in dilations:
            x = tcn_block(x, filters, kernel_size, d, dropout, use_layernorm)

    # Compress to intermediate features
    x = Conv1D(32, 1, padding="same")(x)
    x = Activation("relu")(x)
    x = Lambda(lambda t: t[:, -1, :])(x)           # (batch, 32)

    # ⭐ QUANTILE HEAD: Output len(TAUS) channels ⭐
    # Instead of Dense(1), output Dense(len(TAUS)) = 3 quantiles
    out = Dense(len(TAUS))(x)                      # ← OUTPUTS 3 VALUES
                                                    # [q_0.025, q_0.5, q_0.975]

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss=None)  # set later
    return model
```

**Architecture Flow:**
```
Input → TCN Blocks → Conv1D(1,1) → Lambda(last_step) → Dense(32) → Dense(3)
                                                             ├─ [0]: q_0.025 (lower)
                                                             ├─ [1]: q_0.5 (median)
                                                             └─ [2]: q_0.975 (upper)
```

---

### **Stage 3: Pinball Loss for Multi-Quantile Regression (Lines ~88-97)**
**File Location:** `pinball_multi()` loss function

```python
def pinball_multi(taus):
    """
    Pinball Loss: asymmetric loss for quantile regression
    
    taus = [0.025, 0.5, 0.975]
    For each quantile τ and residual e = y - ŷ_τ:
      L_τ = max(τ*e, (τ-1)*e)
    
    This loss encourages:
      - τ < 0.5 (lower quantile): penalizes overshooting (prediction too high)
      - τ = 0.5 (median): symmetric (like L1/MAE)
      - τ > 0.5 (upper quantile): penalizes undershooting (prediction too low)
    """
    taus = tf.constant(taus, dtype=tf.float32)
    
    def loss(y_true, y_pred):
        y_t = tf.expand_dims(y_true, axis=-1)      # (B,) → (B,1)
        e   = y_t - y_pred                          # (B,3) residuals
        l   = tf.maximum(taus*e, (taus-1.0)*e)     # Apply pinball
        return tf.reduce_mean(tf.reduce_sum(l, axis=-1))
    
    return loss
```

**Pinball Loss Visualization:**
```
For τ=0.025 (lower quantile):
  If ŷ > y: large penalty (want to be below)
  If ŷ < y: small penalty

For τ=0.5 (median):
  Symmetric: same penalty above/below

For τ=0.975 (upper quantile):
  If ŷ < y: large penalty (want to be above)
  If ŷ > y: small penalty
```

---

### **Stage 4: Ensemble Training with Bootstrap (Lines ~99-121)** ⭐ **CRITICAL CQR COMPONENT**
**File Location:** `compile_member()` and `train_member()` functions

#### **4a: Bootstrap Sampling (Lines ~110-114)**
```python
def bootstrap_idx(n, seed):
    """
    Generate bootstrap indices (sample with replacement)
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=n)

def train_member(seed):
    print(f"\nTraining ensemble member with seed={seed} ...")
    tf.keras.utils.set_random_seed(seed)
    member = compile_member()
    
    if BOOTSTRAP:
        # Sample with replacement for diversity
        idx_bs = bootstrap_idx(len(X_train_w), seed+777)
        Xb, yb = X_train_w[idx_bs], y_train_w[idx_bs]  # ← Bootstrap sample
    else:
        Xb, yb = X_train_w, y_train_w  # Full dataset
    
    # Train on (possibly bootstrap) sample
    member.fit(Xb, yb,
               validation_data=(X_val_w, y_val_w),
               epochs=BEST["epochs"],
               batch_size=BEST["batch_size"],
               verbose=1,
               callbacks=[EarlyStopping(...), ModelCheckpoint(...)]
    )
    return member
```

**Why Ensemble?**
- Creates diversity: different members fit different parts of the data distribution
- Allows better uncertainty estimation across the feature space
- Enables conformal prediction calibration

#### **4b: Train M Members (Lines ~122-126)**
```python
print(f"\nTraining TCN+CQR ensemble with M={M_ENSEMBLE} ...")
t0_train = time.time()

# Train 5 independent models with different bootstrap samples + random seeds
members = [train_member(SEED_BASE + 137*i) for i in range(M_ENSEMBLE)]

t1_train = time.time()
print(f"\nTraining completed in {(t1_train - t0_train):.4f} seconds.")
```

---

### **Stage 5: Per-Member Quantile Predictions (Lines ~128-158)**
**File Location:** `predict_quantiles_member()` and `predict_all()` functions

#### **5a: Single Member Prediction (Lines ~129-140)**
```python
def predict_quantiles_member(m, X, idx):
    """
    Get quantile predictions from one model member
    """
    qz = m.predict(X, verbose=0)           # (N, 3) standardized space
    qy = qz * y_scale + y_mean             # Inverse scale to original space
    qy = np.sort(qy, axis=1)               # Enforce monotonicity: L ≤ M ≤ U
    
    return (
        pd.Series(qy[:,0], index=idx),     # Lower quantile (q_0.025)
        pd.Series(qy[:,1], index=idx),     # Median quantile (q_0.5)
        pd.Series(qy[:,2], index=idx)      # Upper quantile (q_0.975)
    )
```

#### **5b: Aggregate All Members (Lines ~142-158)**
```python
def predict_all(members, X, idx):
    """
    Aggregate predictions across all ensemble members
    """
    Ls, Ms, Us = [], [], []
    
    # Collect predictions from each member
    for m in members:
        qL, qM, qU = predict_quantiles_member(m, X, idx)
        Ls.append(qL.values); Ms.append(qM.values); Us.append(qU.values)
    
    # Stack: (N, M) where N=samples, M=members
    Ls = np.stack(Ls, axis=1)  # (N, 5)
    Ms = np.stack(Ms, axis=1)  # (N, 5)
    Us = np.stack(Us, axis=1)  # (N, 5)
    
    # Ensemble aggregation (averaging)
    L_bar = pd.Series(Ls.mean(axis=1), index=idx, name="qL_bar")      # Average lower
    M_bar = pd.Series(Ms.mean(axis=1), index=idx, name="qM_bar")      # Average median
    U_bar = pd.Series(Us.mean(axis=1), index=idx, name="qU_bar")      # Average upper
    
    # Epistemic uncertainty = across-member variance
    var_epi = pd.Series(
        Ms.var(axis=1, ddof=1) if Ms.shape[1]>1 else np.zeros(Ms.shape[0]),
        index=idx, name="var_epistemic"
    )
    
    # Aleatoric uncertainty ≈ from interval width (IQR proxy)
    IQR = U_bar.values - L_bar.values
    sigma_alea = IQR / 3.92  # Gaussian approximation
    var_alea = pd.Series(np.maximum(sigma_alea, 0)**2, index=idx, name="var_aleatoric")
    
    return L_bar, M_bar, U_bar, var_epi, var_alea
```

**Key Points:**
- Each member outputs 3 quantiles independently
- Average across members for robustness
- Variance across members estimates epistemic uncertainty

---

### **Stage 6: Split Conformal Calibration (Lines ~159-171)** ⭐ **CORE CQR STEP**
**File Location:** Calibration using validation set

```python
# ──────────────────────────────────────────────────────
# SPLIT CONFORMAL CALIBRATION (on validation set)
# ──────────────────────────────────────────────────────

# Step 1: Get non-conformity scores on VALIDATION set
# Non-conformity = how far actual is from predicted interval
E_val = np.maximum(
    L_v_bar.values - actual_val.values,      # How much below lower?
    actual_val.values - U_v_bar.values       # How much above upper?
)
E_val = np.maximum(E_val, 0.0)               # (N_val,)

# Step 2: Compute quantile of non-conformity scores
# This guarantees (1-α) coverage with high probability
q_hat = np.quantile(E_val, 1 - ALPHA, method="higher")

# q_hat = maximum non-conformity we tolerate

# Step 3: Expand intervals by q_hat on both sides
# This is the "conformal" correction step
L_train = L_tr_bar - q_hat;   U_train = U_tr_bar + q_hat
L_val   = L_v_bar  - q_hat;   U_val   = U_v_bar  + q_hat
L_test  = L_te_bar - q_hat;   U_test  = U_te_bar + q_hat
```

**Mathematical Foundation:**
```
Non-conformity score:
  R_i = max(0, L_i - y_i, y_i - U_i)
  
Conformal threshold (computed from validation):
  q_hat = ⌈(n+1)(1-α)/n⌉-th smallest R_i
  
Final conformalized interval:
  [L_final, U_final] = [L - q_hat, U + q_hat]
  
Guarantee: P(y ∈ [L_final, U_final]) ≥ 1-α (asymptotically)
```

**Why Validation Set?**
- Train set: used to fit all 5 ensemble members
- Validation set: used to calibrate the conformal threshold q_hat
- Test set: evaluate on truly unseen data
- This split ensures the guarantee is valid

---

### **Stage 7: Point Forecast & Uncertainty Decomposition (Lines ~172-180)**
**File Location:** Extract median and variance estimates

```python
# Point forecast = ensemble median
mean_train = M_tr_bar.rename("point_pred")
mean_val   = M_v_bar.rename("point_pred")
mean_test  = M_te_bar.rename("point_pred")

# Epistemic & Aleatoric from earlier computation
var_epi_tr, var_alea_tr  # From predict_all()
var_epi_te, var_alea_te

# Total uncertainty
var_total_te = var_epi_te + var_alea_te
```

---

### **Stage 8: UQ Metrics Computation (Lines ~182-210)**
**File Location:** `base_metrics()` and `uq_metrics()` functions

```python
def uq_metrics(y_true, L, U, alpha=ALPHA):
    """
    Compute UQ metrics for conformalized intervals
    """
    y = np.asarray(y_true); L = np.asarray(L); U = np.asarray(U)
    
    # PICP: What fraction of actuals are inside the interval?
    cover = (y >= L) & (y <= U)
    picp = cover.mean()
    
    # MPIW: Average interval width (sharpness)
    mpiw = np.mean(U - L)
    
    # Winkler Score: width + penalty for miscoverage
    penalty = np.where(
        y < L, (2/alpha)*(L - y),
        np.where(y > U, (2/alpha)*(y - U), 0.0)
    )
    winkler = np.mean((U - L) + penalty)
    
    return dict(PICP=picp, MPIW=mpiw, Winkler=winkler)
```

**Key Metrics:**
- **PICP**: Should be ≥ 1-α (ideally ~0.95 for α=0.05)
- **MPIW**: Smaller is sharper (but must maintain coverage)
- **Winkler**: Lower is better (balances width and coverage)

---

## 2. SUMMARY TABLE: WHERE CQR APPLIES

| **Stage** | **Component** | **CQR Role** | **Code Location** |
|-----------|---------------|---|---|
| **Config** | TAUS = [0.025, 0.5, 0.975] | Define quantile levels | Lines ~17-34 |
| **Architecture** | Quantile head `Dense(len(TAUS))` | Output 3 quantiles | Lines ~75-76 |
| **Loss Function** | `pinball_multi(TAUS)` | Asymmetric loss per quantile | Lines ~88-97 |
| **Ensemble Setup** | M=5 members, bootstrap sampling | Create diversity | Lines ~99-121 |
| **Per-Member Pred** | `predict_quantiles_member()` | Get 3 quantiles per model | Lines ~129-140 |
| **Aggregation** | `predict_all()`, ensemble averaging | Combine across members | Lines ~142-158 |
| **Conformal Cal** | Split conformal on validation | Compute q_hat threshold | Lines ~159-171 |
| **Calibrated Int** | [L - q_hat, U + q_hat] | Guarantee coverage | Lines ~159-171 |
| **Decomposition** | var_epi, var_alea from ensemble | Uncertainty analysis | Lines ~172-180 |
| **Evaluation** | PICP, MPIW, Winkler | Measure UQ quality | Lines ~182-210 |

---

## 3. CQR FLOWCHART

```
┌────────────────────────────────────────────────────────────────────────────┐
│     TCN + CONFORMAL QUANTILE REGRESSION (CQR) WITH ENSEMBLE LEARNING       │
└────────────────────────────────────────────────────────────────────────────┘

                                  INPUT DATA
                                      │
                                      ▼
                    ┌──────────────────────────────┐
                    │  Train/Val/Test Windowing    │
                    │  (lookback from Optuna)      │
                    └──────────────────────────────┘
                                      │
                    ══════════════════════════════════════
                   ║     NO OPTUNA HPO (use best from    ║
                   ║     prior TCN_Baseline)              ║
                    ══════════════════════════════════════
                                      │
                                      ▼
        ┌───────────────────────────────────────────────────┐
        │  Build TCN with Quantile Head                     │
        │  ┌─────────────────────────────────────────────┐ │
        │  │  Conv1D blocks                              │ │
        │  ├─ SpatialDropout1D (regularization)         │ │
        │  ├─ LayerNorm, Residual connections          │ │
        │  │                                             │ │
        │  ├─ Conv1D(1,1) → Conv1D(32) → Dense(3)      │ │
        │  │                                    ↑        │ │
        │  │                           QUANTILE HEAD    │ │
        │  │  Outputs 3 quantiles:                      │ │
        │  │  ├─ [0]: q_0.025 (lower)                  │ │
        │  │  ├─ [1]: q_0.5 (median)                   │ │
        │  │  └─ [2]: q_0.975 (upper)                  │ │
        │  │                                             │ │
        │  │  Loss: pinball_multi(TAUS)                 │ │
        │  │  Asymmetric penalties per quantile         │ │
        │  └─────────────────────────────────────────────┘
        │                                                 │
        │  Compiled with Adam optimizer + pinball loss   │
        └───────────────────────────────────────────────┘
                                      │
                    ══════════════════════════════════════
                   ║      ENSEMBLE TRAINING (M=5)          ║
                    ══════════════════════════════════════
                                      │
                    ┌──────────────────────────────────────┐
                    │  FOR i = 1 TO 5:                      │
                    │                                       │
                    │  ├─ Generate unique random seed      │
                    │  │  seed_i = SEED_BASE + 137*i       │
                    │  │                                    │
                    │  ├─ Bootstrap sampling:              │
                    │  │  idx_bs = sample(1..N,            │
                    │  │           size=N,                 │
                    │  │           replace=True,           │
                    │  │           seed=seed_i)            │
                    │  │  X_bootstrap, y_bootstrap         │
                    │  │  = data[idx_bs]                   │
                    │  │                                    │
                    │  ├─ Create new model                 │
                    │  │  (build_tcn_quantile)             │
                    │  │                                    │
                    │  ├─ Train on bootstrap sample        │
                    │  │  member_i.fit(X_bootstrap,        │
                    │  │              y_bootstrap,         │
                    │  │              val_data=(X_val,     │
                    │  │              y_val),              │
                    │  │              EarlyStopping)       │
                    │  │                                    │
                    │  └─ Save trained member              │
                    │     members[i] = member_i            │
                    │                                       │
                    └──────────────────────────────────────┘
                                      │
                    ══════════════════════════════════════
                   ║        ENSEMBLE PREDICTION            ║
                    ══════════════════════════════════════
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  1️⃣  PREDICT WITH EACH MEMBER         │
                    │  for member in members:              │
                    │    pred = member.predict(X)          │
                    │    qL, qM, qU = inverse_scale(pred)  │
                    │    enforce_monotonicity(qL≤qM≤qU)    │
                    │    collect quantiles                 │
                    │                                       │
                    │  Result: 3 arrays (N, 5)             │
                    │    Ls, Ms, Us (per sample, per member)
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  2️⃣  ENSEMBLE AGGREGATION            │
                    │                                      │
                    │  L_bar = mean(Ls, axis=1)           │
                    │  M_bar = mean(Ms, axis=1)           │
                    │  U_bar = mean(Us, axis=1)           │
                    │                                      │
                    │  Ensemble mean estimates:           │
                    │  Point forecast = M_bar             │
                    │  Initial interval = [L_bar, U_bar]  │
                    │  (NOT yet conformalized)            │
                    └──────────────────────────────────────┘
                                      │
                    ══════════════════════════════════════
                   ║   SPLIT CONFORMAL CALIBRATION        ║
                   ║   (on VALIDATION set only!)           ║
                    ══════════════════════════════════════
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  3️⃣  COMPUTE NON-CONFORMITY SCORES   │
                    │  (on validation set)                 │
                    │                                      │
                    │  For each validation sample:         │
                    │    e_i = max(0,                      │
                    │      L_bar_i - y_i,                 │
                    │      y_i - U_bar_i)                 │
                    │                                      │
                    │  E_val = [e_1, e_2, ..., e_n]       │
                    │  Shape: (N_val,)                     │
                    │                                      │
                    │  (How far each point is outside      │
                    │   the uncalibrated interval)         │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  4️⃣  CALIBRATION THRESHOLD            │
                    │                                      │
                    │  q_hat = quantile(E_val,             │
                    │           q = 1 - ALPHA,             │
                    │           method='higher')           │
                    │                                      │
                    │  ALPHA = 0.05  →  q = 0.95          │
                    │  q_hat ≈ 95th percentile of E_val   │
                    │                                      │
                    │  This value guarantees coverage!     │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  5️⃣  CONFORMALIZE ALL INTERVALS       │
                    │  (apply q_hat to all splits)        │
                    │                                      │
                    │  L_final = L_bar - q_hat            │
                    │  U_final = U_bar + q_hat            │
                    │                                      │
                    │  Expand intervals symmetrically      │
                    │  This is the "conformal correction" │
                    │  that guarantees coverage!          │
                    └──────────────────────────────────────┘
                                      │
                    ══════════════════════════════════════
                   ║   UNCERTAINTY DECOMPOSITION          ║
                    ══════════════════════════════════════
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  6️⃣  EPISTEMIC UNCERTAINTY            │
                    │  (across-member variance)           │
                    │                                      │
                    │  For each sample:                    │
                    │    σ_epi = var(M_bar_i across       │
                    │              members)                │
                    │                                      │
                    │  High variance → uncertain           │
                    │  Low variance → confident            │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  7️⃣  ALEATORIC UNCERTAINTY            │
                    │  (interval width proxy)             │
                    │                                      │
                    │  IQR = U_bar - L_bar                │
                    │  σ_alea = IQR / 3.92                │
                    │           (Gaussian approx)         │
                    │                                      │
                    │  Wide intervals → noisy data        │
                    │  Narrow intervals → clean data      │
                    └──────────────────────────────────────┘
                                      │
                    ══════════════════════════════════════
                   ║    FINAL PREDICTIONS & METRICS       ║
                    ══════════════════════════════════════
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  8️⃣  POINT FORECAST                   │
                    │  point_pred = M_bar (median)        │
                    │              (ensemble average)      │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  9️⃣  PREDICTION INTERVALS             │
                    │  [L_final, U_final]                 │
                    │  = [L_bar - q_hat, U_bar + q_hat]   │
                    │                                      │
                    │  GUARANTEED coverage ≥ (1-α)        │
                    │  (asymptotically)                    │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  🔟  COMPUTE UQ METRICS               │
                    │  ├─ PICP: P(y ∈ [L,U])              │
                    │  │   Target ≥ 0.95                  │
                    │  ├─ MPIW: E[U - L]                  │
                    │  │   Smaller is sharper             │
                    │  ├─ Winkler: width + penalty        │
                    │  │   Lower is better                │
                    │  └─ Coverage heatmap & rolling      │
                    │     PICP/MPIW plots                 │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  SAVE RESULTS                        │
                    │  ├─ ALL_UQ_PREDICTED.csv            │
                    │  │  (median, L, U, actual)          │
                    │  └─ ALL_UQ_METRICS.csv              │
                    │     (PICP, MPIW, Winkler, etc.)    │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                                  OUTPUT: 
                    Median Forecast + Conformalized PI
                    (Calibrated for ≥95% coverage)
                                  & Metrics
```

---

## 4. KEY CQR CONCEPTS IN YOUR IMPLEMENTATION

### **The Core Insight**

**Standard Point Regression:**
```
Model → single output → single prediction
Loss: MSE (mean squared error)
```

**Quantile Regression:**
```
Model → 3 outputs (q_0.025, q_0.5, q_0.975)
Loss: Pinball (asymmetric per quantile)
∼ Directly predicts intervals without post-hoc methods
```

**CQR (Quantile Regression + Conformal):**
```
Quantile Regression model (3 ensemble members)
         ↓
    Per-member quantile predictions
         ↓
    Ensemble averaging (create L_bar, U_bar)
         ↓
    Split Conformal Calibration (validation set)
         ↓
    Compute non-conformity threshold q_hat
         ↓
    Expand intervals: [L_bar - q_hat, U_bar + q_hat]
         ↓
    GUARANTEED coverage ≥ (1-α) on test set!
```

---

### **Why Three Quantiles?**

| **Quantile** | **Level** | **Purpose** |
|-------------|----------|-----------|
| **Lower** | 0.025 | 2.5th percentile (lower bound) |
| **Median** | 0.5 | 50th percentile (point forecast) |
| **Upper** | 0.975 | 97.5th percentile (upper bound) |

Together they form a 95% prediction interval before conformal correction.

---

### **Pinball Loss vs MSE**

**MSE Loss:**
```
L = (y - ŷ)²
→ Encourages mean prediction
→ Symmetric: equally penalizes over/underestimation
```

**Pinball Loss for Quantile τ:**
```
L_τ = max(τ*e, (τ-1)*e)  where e = y - ŷ

For τ=0.025:
  If ŷ > y:  penalty = 0.025 * (ŷ - y)  [high]
  If ŷ < y:  penalty = 0.975 * (y - ŷ)  [very high]
  → Encourages predictions ABOVE y (lower quantile)

For τ=0.5:
  Symmetric: same penalty above/below
  → Encourages median prediction

For τ=0.975:
  If ŷ > y:  penalty = 0.975 * (ŷ - y)  [very high]
  If ŷ < y:  penalty = 0.025 * (y - ŷ)  [low]
  → Encourages predictions BELOW y (upper quantile)
```

---

### **Ensemble & Bootstrap**

**Why Ensemble (M=5)?**
- Each member trains on different bootstrap sample
- Creates diversity in parameter estimates
- Enables better uncertainty quantification
- Across-member variance = epistemic uncertainty

**Why Bootstrap?**
- Resampling from training data WITH replacement
- Different member sees slightly different distribution
- Computationally cheaper than Bayesian inference
- Mimics posterior samples in a practical way

---

### **Split Conformal Prediction**

**Traditional Approach (Problem):**
- Train model on training data
- Apply to test set
- NO guarantee that coverage ≥ (1-α)

**Conformal Prediction (Solution):**
```
1. Train on training split
2. Compute non-conformity on VALIDATION split
3. Calibrate threshold q_hat from validation
4. Apply q_hat to test split
5. GUARANTEED: P(y ∈ interval) ≥ 1-α (asymptotically)
```

**Mathematical Guarantee:**
```
Non-conformity scores on validation:
  R_i = max(0, L_i - y_i, y_i - U_i)

Threshold (for coverage 1-α):
  q_hat = ⌈(n+1)(1-α)/n⌉-th smallest R_i

Conformalized interval on test:
  [L - q_hat, U + q_hat]

Proof:
  P(y_test ∈ [L_test - q_hat, U_test + q_hat])
  = P(R_test ≤ q_hat)
  ≥ 1 - α (by exchangeability)
```

---

## 5. CONFIGURATION PARAMETERS FOR CQR

| **Parameter** | **Default** | **Purpose** |
|---------------|-------------|-----------|
| `ALPHA` | 0.05 | Significance level (95% coverage target) |
| `TAUS` | [0.025, 0.5, 0.975] | Quantile levels (3 quantiles) |
| `M_ENSEMBLE` | 5 | Number of ensemble members |
| `BOOTSTRAP` | True | Use bootstrap resampling |
| `SEED_BASE` | 42 | Base random seed for ensemble diversity |

---

## 6. MATHEMATICAL FOUNDATION

### **Pinball Loss for Multiple Quantiles**
```
L(y, ŷ) = Σ_{τ∈TAUS} L_τ(y, ŷ_τ)

Where:
  L_τ(y, ŷ) = max(τ*(y - ŷ), (τ-1)*(y - ŷ))
             = ρ_τ(y - ŷ)  (quantile loss)
```

### **Ensemble Aggregation**
```
Averaged quantile:
  q̄_τ = (1/M) Σ_{m=1}^M q_τ^{(m)}

Epistemic variance:
  σ_epi² = (1/M) Σ_{m=1}^M (q_0.5^{(m)} - q̄_0.5)²

Aleatoric variance (from interval):
  σ_alea² = (IQR / 3.92)²  [Gaussian approximation]
```

### **Conformal Threshold (Theory)**
```
For any dataset of size n and any α ∈ (0,1):

E[Coverage] = P(y_{n+1} ∈ Interval_{n+1})
            ≥ 1 - α - O(1/n)

The threshold:
  q_hat = ⌈(n+1)(1-α)/n⌉-th order statistic of R_val

Ensures coverage even with finite samples!
```

---

## 7. COMPARISON: MCD vs HLLLA vs CQR

| **Aspect** | **MCD** | **HLLLA** | **CQR** |
|-----------|---------|----------|--------|
| **Point Pred** | MC mean (100 passes) | Deterministic μ | Ensemble median |
| **Mechanism** | MC dropout stochasticity | Heteroscedastic NLL + LLLA | Quantile regression + conformal |
| **Aleatoric Est.** | Implicit (MC variance) | Explicit (learned σ²) | From interval width (IQR) |
| **Epistemic Est.** | From MC ensemble | From LLLA posterior | Across-member variance |
| **Calibration** | Implicit (dropout %) | Implicit (loss function) | Explicit conformal (validation) |
| **Coverage Guar.** | Approximate | Approximate | Formal (asymptotic) |
| **Computation** | Slow (100 forward passes) | Fast (single pass + Hessian) | Medium (5 forward passes) |
| **Ensemble** | 100 stochastic samples | Single deterministic model | 5 independent trained models |
| **Key Advantage** | Simple, well-studied | Efficient, interpretable | **Formal coverage guarantee** |
| **Key Limitation** | Slow inference | Approximation quality varies | Requires validation split |

---

## 8. OUTPUT INTERPRETATION

### **Example Metrics Summary:**
```
Split    MSE     MAE     RMSE    MAPE    R²      PICP    MPIW    Winkler
────────────────────────────────────────────────────────────────────────
Train    0.0048  0.0542  0.0693  0.0162  0.9809  0.9480  0.2406  0.3128
Val      0.0055  0.0608  0.0742  0.0182  0.9754  0.9420  0.2598  0.3367
Test     0.0070  0.0734  0.0837  0.0219  0.9668  0.9524  0.2945  0.3791
```

**Interpretation:**
- **PICP ≈ 0.95** → Conformal calibration achieved target coverage!
- **MPIW** → Interval width (CQR typically sharper than MCD/HLLLA)
- **Winkler** → Lower is better
- **Coverage guarantee**: Even if PICP slightly below 0.95 in finite samples, asymptotically ≥ 0.95

### **Uncertainty Decomposition:**
```
At any time t:
  ├─ σ_epi(t) = √var(median across 5 members)
  │             (model uncertainty)
  ├─ σ_alea(t) = IQR(t) / 3.92
  │              (data noise estimate)
  └─ Total interval = [L_bar - q_hat, U_bar + q_hat]
                      (uses both components implicitly)
```

---

## Summary

**CQR applies in 8 key stages:**

1. **Configuration**: Define TAUS = [0.025, 0.5, 0.975]
2. **Architecture**: Quantile head with Dense(3)
3. **Loss Function**: Pinball loss (asymmetric, per-quantile)
4. **Ensemble**: Train M=5 models with bootstrap
5. **Per-Member**: Each model predicts 3 quantiles
6. **Aggregation**: Average across members
7. **Conformal Calibration**: Use validation set to compute q_hat
8. **Conformalization**: Expand intervals by q_hat (guarantees coverage)

The result is a **provably calibrated prediction interval** with:
- **Formal coverage guarantee** ≥ (1-α) asymptotically
- **Explicit uncertainty decomposition** (epistemic + aleatoric)
- **Sharper intervals** than methods without calibration
- **Practical efficiency** (5 models, not 100+ passes)

This method is particularly valuable when **coverage guarantee** is critical (e.g., financial forecasting, safety-critical applications).
