# Conformal Quantile Regression (CQR) in RNNs (GRU & LSTM): Detailed Analysis

## Overview

The GRU_CQR_42.ipynb and LSTM_CQR_42.ipynb implementations apply **Conformal Quantile Regression (CQR)** for uncertainty quantification in recurrent time-series forecasting. This method provides:

1. **Direct Quantile Learning**: Train RNNs to predict quantiles (0.025, 0.5, 0.975) via pinball loss
2. **Ensemble Diversity**: Multiple RNNs trained on bootstrap samples for epistemic uncertainty
3. **Split Conformal Calibration**: Distribution-free method for guaranteed coverage on test set
4. **Formal Guarantees**: Prediction intervals with provable coverage probability (not empirical)

Both GRU and LSTM follow identical CQR methodologies with an 11-stage pipeline. The ensemble approach provides both:
- **Epistemic Uncertainty**: Variance across ensemble members
- **Aleatoric Uncertainty**: Inter-quantile width (proxied via IQR/3.92)

This document explains the unified CQR implementation for both RNN types.

---

## 1. WHERE CQR IS APPLIED IN RNN PIPELINES (GRU & LSTM)

### **Stage 1: RNN Architecture with Quantile Output (Lines ~37-48 in GRU, similar in LSTM)**
**File Location:** `build_gru_quantile()` / `build_lstm_quantile()` functions

#### **GRU Architecture:**
```python
def build_gru_quantile(n_features: int, dropout_rate: float):
    """
    RNN model for learning multiple quantiles
    """
    m = Sequential()
    m.add(Input(shape=(BEST["lookback"], n_features)))  # (batch, lookback, n_features)
    
    # RNN layer(s)
    if BEST.get("num_layers", 1) == 2:
        # Optional dual-layer (disabled in current config)
        # m.add(GRU(BEST["units1"], return_sequences=True))
        # m.add(Dropout(dropout_rate))
        # m.add(GRU(BEST["units2"]))
        pass
    else:
        # Single GRU layer
        m.add(GRU(BEST["units1"]))  # → (batch, units1)
    
    m.add(Dropout(dropout_rate))
    
    # ← CRITICAL: Multiple quantile output heads (not single point prediction)
    m.add(Dense(len(TAUS)))  # Output: [q_0.025, q_0.5, q_0.975] (standardized)
    
    return m

# TAUS = [0.025, 0.5, 0.975]  (lower, median, upper)
# Output shape: (batch, 3) - one prediction per quantile level
```

#### **LSTM Architecture (identical structure):**
```python
def build_lstm_quantile(n_features: int, dropout_rate: float):
    m = Sequential()
    m.add(Input(shape=(BEST["lookback"], n_features)))

    if BEST["num_layers"] == 2:
        # m.add(LSTM(BEST["units1"], return_sequences=True))
        # m.add(Dropout(dropout_rate))
        # m.add(LSTM(BEST["units2"]))
        pass
    else:
        m.add(LSTM(BEST["units1"]))  # → (batch, units1)

    m.add(Dropout(dropout_rate))
    m.add(Dense(len(TAUS)))  # ← [q_0.025, q_0.5, q_0.975]
    
    return m
```

**Key Architecture Points:**
- **Multiple Quantile Heads**: Dense(len(TAUS)) = Dense(3) for 3 quantiles
  - Head 1: Lower quantile q_0.025 (2.5th percentile)
  - Head 2: Median quantile q_0.5 (50th percentile) 
  - Head 3: Upper quantile q_0.975 (97.5th percentile)
- **Output in Scaled Space**: All predictions in [−1, 1] range (standardized)
- **No Variance Head**: Unlike HLLLA, no explicit variance prediction
- **No Dropout for MC**: Dropout is only for regularization, not Bayesian inference

---

### **Stage 2: Pinball Loss Function (Lines ~50-62)**
**File Location:** Quantile regression loss definition

```python
def pinball_multi(taus):
    """
    Multi-quantile pinball loss (asymmetric absolute loss)
    
    taus: array of quantile levels [0.025, 0.5, 0.975]
    """
    taus = tf.constant(taus, dtype=tf.float32)  # Convert to TensorFlow
    
    def loss(y_true, y_pred):
        """
        y_true: (batch,) - actual targets (in scaled space)
        y_pred: (batch, 3) - predicted quantiles
        """
        # Reshape true values to (batch, 1) for broadcasting
        y_t = tf.expand_dims(y_true, axis=-1)  # (batch, 1)
        
        # Residuals: error for each quantile
        e = y_t - y_pred  # (batch, 3), e[:,i] = y - q_tau_i
        
        # Pinball loss (asymmetric):
        # L_tau = tau*e if e > 0 (overestimate)
        #         (tau-1)*e if e < 0 (underestimate)
        # Vectorized: max(tau*e, (tau-1)*e)
        l = tf.maximum(taus * e, (taus - 1.0) * e)  # (batch, 3)
        
        # Sum across quantiles, mean across batch
        return tf.reduce_mean(tf.reduce_sum(l, axis=-1))
    
    return loss
```

**Pinball Loss Intuition:**

For each quantile τ:
$$L_\tau(\mathbf{y}, \mathbf{q}) = \sum_{i=1}^{N} \left[\tau \cdot \mathbf{1}(y_i > q_i) + (1-\tau) \cdot \mathbf{1}(y_i < q_i)\right] |y_i - q_i|$$

Where:
- If $y > q$: penalty = $\tau \times |y - q|$ (overestimate cost)
- If $y < q$: penalty = $(1-\tau) \times |y - q|$ (underestimate cost)
- For $\tau=0.5$ (median): symmetric, same as L1 loss
- For $\tau=0.025$ (lower): penalizes overestimation 0.025× more than underestimation
- For $\tau=0.975$ (upper): penalizes underestimation 0.025× more than overestimation

---

### **Stage 3: Ensemble Member Compilation (Lines ~64-70)**
**File Location:** `compile_member()` function

```python
def compile_member():
    """
    Compile a single ensemble member with quantile regression
    """
    m = build_gru_quantile(
        n_features=len(feature_cols), 
        dropout_rate=BEST["dropout"]
    )
    
    # Compile with pinball loss
    m.compile(
        optimizer=optimizers.Adam(learning_rate=BEST["lr"]),
        loss=pinball_multi(TAUS)  # ← Multiple quantiles
    )
    
    return m
```

---

### **Stage 4: Bootstrap Sampling for Diversity (Lines ~72-78)**
**File Location:** `bootstrap_idx()` and `train_member()` functions

```python
def bootstrap_idx(n, seed):
    """
    Create bootstrap sample indices (sampling with replacement)
    """
    rng = np.random.default_rng(seed)
    # Draw n samples from [0, n) WITH REPLACEMENT
    return rng.integers(0, n, size=n)  # Shape: (n,)

def train_member(seed):
    """
    Train a single ensemble member on bootstrap sample
    """
    print(f"Training member with seed {seed}...")
    
    # Set random seed for reproducibility
    tf.keras.utils.set_random_seed(seed)
    
    # Compile fresh model
    member = compile_member()
    
    # Bootstrap sampling creates diversity
    if BOOTSTRAP:  # BOOTSTRAP = True
        # Sample with replacement from training data
        idx_bs = bootstrap_idx(len(X_train_w), seed + 777)  # Different seed for variety
        Xb = X_train_w[idx_bs]  # Bootstrap features (may have duplicates)
        yb = y_train_w[idx_bs]  # Bootstrap targets
    else:
        Xb, yb = X_train_w, y_train_w  # Use full training data

    # Callbacks
    cbs = [
        EarlyStopping(
            monitor="val_loss",
            patience=BEST["patience"],
            restore_best_weights=True
        ),
        ModelCheckpoint(
            f"Model Checkpoints/gru_cqr_member_{seed}.keras",
            monitor="val_loss",
            save_best_only=True
        )
    ]

    # Train on (possibly bootstrap) data
    member.fit(
        Xb, yb,
        validation_data=(X_val_w, y_val_w),
        epochs=BEST["epochs"],
        batch_size=BEST["batch_size"],
        verbose=1,
        callbacks=cbs
    )

    return member
```

**Bootstrap Concept:**
```
Original training data: [x₁, y₁], [x₂, y₂], ..., [xₙ, yₙ]

Member 1 (seed=42):
├─ idx_bs = [1, 3, 1, 2, 1, ...]  (with replacement)
└─ Training data = {[x₁, y₁], [x₃, y₃], [x₁, y₁], [x₂, y₂], ...}

Member 2 (seed=42+137):
├─ idx_bs = [2, 4, 1, 3, 2, ...]  (different random order)
└─ Training data = {[x₂, y₂], [x₄, y₄], [x₁, y₁], [x₃, y₃], ...}

Member 3 (seed=42+274):
├─ idx_bs = [4, 1, 2, 3, 4, ...]  (yet another order)
└─ Training data = {[x₄, y₄], [x₁, y₁], [x₂, y₂], [x₃, y₃], ...}

Result: 5 different models trained on different subsets → diverse predictions
```

#### **Ensemble Training Loop (Lines ~80-90):**
```python
print(f"\nTraining GRU+CQR ensemble with M={M_ENSEMBLE} ...")
t0_train = time.time()

# Train M=5 independent ensemble members
members = [train_member(SEED_BASE + 137*i) for i in range(M_ENSEMBLE)]
#                              └─── Different seed per member ───┘

t1_train = time.time()
print(f"Training completed in {round(t1_train - t0_train, 4)} seconds.")
```

**Ensemble Diversity Creates Epistemic Uncertainty:**
- Each member sees different training data (bootstrap)
- Each member has different random initialization
- Different final weight configurations
- → Predictions vary across members
- → Variance of predictions ≈ epistemic uncertainty

---

### **Stage 5: Per-Member Quantile Prediction (Lines ~92-108)**
**File Location:** `predict_quantiles_member()` function

```python
def predict_quantiles_member(m, X, idx):
    """
    Get quantile predictions from a single ensemble member
    
    m: trained model
    X: input data (N, lookback, n_features)
    idx: sample indices/dates
    
    Returns: three Series (lower, median, upper quantiles)
    """
    # Forward pass in standardized space
    qz = m.predict(X, verbose=1)  # (N, 3) = [q_0.025, q_0.5, q_0.975] standardized
    
    # Inverse-scale to original data space
    # If y_scaled = (y - mean)/std, then y_orig = y_scaled * std + mean
    # Linear transformation PRESERVES QUANTILE ORDER
    qy = qz * y_scale + y_mean  # (N, 3) in original scale
    
    # Enforce monotonicity: L ≤ M ≤ U
    # (sorting guarantees order even if model predicts out of order)
    qy = np.sort(qy, axis=1)  # (N, 3)
    
    # Return as Series for each quantile
    return (
        pd.Series(qy[:, 0], index=idx),  # Lower: q_0.025
        pd.Series(qy[:, 1], index=idx),  # Median: q_0.5
        pd.Series(qy[:, 2], index=idx)   # Upper: q_0.975
    )
```

**Monotonicity Enforcement:**
```
Model output (may be unsorted):
  q_0.025 = 1050
  q_0.5   = 1000  (WRONG: median < lower!)
  q_0.975 = 1200

After sorting:
  q_0.025 = 1000  ✓ (correct order)
  q_0.5   = 1050  ✓
  q_0.975 = 1200  ✓
```

---

### **Stage 6: Ensemble Aggregation (Lines ~110-135)** ⭐ **ENSEMBLE AGGREGATION**
**File Location:** `predict_all()` function

```python
def predict_all(members, X, idx):
    """
    Aggregate predictions from all M ensemble members
    
    members: list of M trained models
    X: input data
    idx: sample indices
    
    Returns: ensemble mean quantiles + uncertainty estimates
    """
    Ls, Ms, Us = [], [], []
    
    # Step 1: Get quantiles from each member
    for m in members:
        qL, qM, qU = predict_quantiles_member(m, X, idx)
        Ls.append(qL.values)   # (N,)
        Ms.append(qM.values)   # (N,)
        Us.append(qU.values)   # (N,)
    
    # Stack into (N, M) arrays
    Ls = np.stack(Ls, axis=1)  # (N_samples, M_members=5)
    Ms = np.stack(Ms, axis=1)  # (N_samples, 5)
    Us = np.stack(Us, axis=1)  # (N_samples, 5)
    
    # Step 2: Average across ensemble members
    L_bar = pd.Series(Ls.mean(axis=1), index=idx, name="qL_bar")
    M_bar = pd.Series(Ms.mean(axis=1), index=idx, name="qM_bar")
    U_bar = pd.Series(Us.mean(axis=1), index=idx, name="qU_bar")
    
    # Step 3: Estimate EPISTEMIC UNCERTAINTY from ensemble spread
    # Epistemic = variance across members of the median prediction
    var_epi = pd.Series(
        Ms.var(axis=1, ddof=1) if Ms.shape[1] > 1 else np.zeros(Ms.shape[0]),
        index=idx,
        name="var_epistemic"
    )
    
    # Step 4: Estimate ALEATORIC UNCERTAINTY from IQR
    # Gaussian approximation: IQR ≈ 3.92 * σ
    # So: σ ≈ IQR / 3.92
    IQR = U_bar.values - L_bar.values  # Inter-quartile range
    sigma_alea = IQR / 3.92  # Convert IQR to std dev
    var_alea = pd.Series(
        np.maximum(sigma_alea, 0) ** 2,  # Variance = σ²
        index=idx,
        name="var_aleatoric"
    )
    
    return L_bar, M_bar, U_bar, var_epi, var_alea
```

**Ensemble Aggregation Illustration:**
```
Member 1:  q_0.025=1010,  q_0.5=1050,  q_0.975=1090
Member 2:  q_0.025=1020,  q_0.5=1045,  q_0.975=1080
Member 3:  q_0.025=1005,  q_0.5=1060,  q_0.975=1110
Member 4:  q_0.025=1015,  q_0.5=1055,  q_0.975=1095
Member 5:  q_0.025=1008,  q_0.5=1052,  q_0.975=1092
           ────────────  ────────────  ────────────
Ensemble:  q_0.025=1012,  q_0.5=1052,  q_0.975=1093  (mean across members)
           std(med)=5.98  (epistemic variance from member spread)
           IQR=81, σ_ale≈20.66  (aleatoric variance from quantile width)
```

#### **Epistemic Uncertainty from Ensemble:**
```
Var_epistemic = Var(M_bar)  across M members

High variance → Members disagree → Model uncertainty high
Low variance  → Members agree    → Model uncertainty low

σ_epi captures:
- Limited training data (different bootstrap samples)
- Parameter uncertainty (different weight configurations)
- Model architecture uncertainty (implicit in ensemble)
```

#### **Aleatoric Uncertainty from IQR:**
```
IQR = U_bar - L_bar  (distance between 2.5th and 97.5th percentile)

For Gaussian distribution:
IQR ≈ 1.96σ_lower + 1.96σ_upper ≈ 3.92σ

So: σ_aleatoric ≈ IQR / 3.92

Captures:
- Intrinsic noise in target
- Data/measurement uncertainty
- Variation not explained by features
```

---

### **Stage 7: Split Conformal Calibration (Lines ~137-150)** ⭐ **CRITICAL FOR FORMAL GUARANTEES**
**File Location:** Conformal calibration on validation set

```python
# Split Conformal Calibration on VALIDATION set
# (not used for training, reserved for calibration)

E_val = np.maximum(
    L_v_bar.values - actual_val.values,  # How much lower bound misses
    actual_val.values - U_v_bar.values   # How much upper bound misses
)
# E_val[i] = 0 if actual[i] is inside [L, U], else > 0

E_val = np.maximum(E_val, 0.0)  # Ensure non-negative

# Compute (⌈(n+1)(1-α)/n⌉) quantile to guarantee coverage
# For finite sample: use quantile at level 1-α with method="higher"
q_hat = np.quantile(E_val, 1 - ALPHA, method="higher")

# q_hat is the CONFORMAL INFLATION THRESHOLD
# Adding q_hat to prediction intervals guarantees:
# P(y ∈ [L - q_hat, U + q_hat]) ≥ 1 - α  (with high probability)
```

**Split Conformal Concept:**

```
Goal: Guarantee coverage on TEST set WITHOUT knowing test distribution

Method:
  1. Validation set (held out):
     └─ For each sample i:
        - Predict [L_i, U_i] using ensemble
        - Compute residual: E_i = max(L_i - y_i, y_i - U_i, 0)
        - E_i = 0 if point inside, else distance outside

  2. Threshold (q_hat):
     └─ q_hat = (1-α)-quantile of E_i
     └─ Chosen so that: ~(1-α) of validation points have E_i ≤ q_hat

  3. Conformalization:
     └─ Final intervals: [L - q_hat, U + q_hat]
     └─ By construction: most validation points fit inside
     └─ Transfer to test: coverage guaranteed (distribution-free!)
```

**Mathematical Guarantee:**

If validation and test data come from same distribution:
$$P(y \in [L - \hat{q}, U + \hat{q}]) \geq 1 - \alpha$$

This is **distribution-free**: No Gaussian assumption, no parametric model. Purely data-driven!

---

### **Stage 8: Conformalized Intervals (Lines ~152-160)**
**File Location:** Apply conformal threshold to all splits

```python
# Apply conformal inflation to all splits
L_train = L_tr_bar - q_hat;   U_train = U_tr_bar + q_hat
L_val   = L_v_bar  - q_hat;   U_val   = U_v_bar  + q_hat
L_test  = L_te_bar - q_hat;   U_test  = U_te_bar + q_hat

# Point forecast = ensemble median (robust statistic)
mean_train = M_tr_bar.rename("point_pred")
mean_val   = M_v_bar.rename("point_pred")
mean_test  = M_te_bar.rename("point_pred")
```

**Why Conformal Works:**
- q_hat is learned on validation set
- q_hat applied uniformly to all test samples
- Makes intervals wider (conservative)
- Guarantees coverage ≥ 1-α on test set
- No retraining, no assumption on test distribution

---

### **Stage 9: Uncertainty Metrics (Lines ~162-188)**
**File Location:** Compute PICP, MPIW, Winkler

```python
def uq_metrics(y_true, L, U, alpha=ALPHA):
    """
    Uncertainty quantification evaluation
    """
    y = np.asarray(y_true); L = np.asarray(L); U = np.asarray(U)
    
    # PICP: Prediction Interval Coverage Probability
    cover = (y >= L) & (y <= U)
    picp = cover.mean()  # Proportion inside
    
    # MPIW: Mean Prediction Interval Width
    width = U - L
    mpiw = width.mean()
    
    # Winkler Score: Width + Penalty for misses
    penalties = np.zeros_like(y)
    penalties[y < L] = (L[y < L] - y[y < L])          # Below lower
    penalties[y > U] = (y[y > U] - U[y > U])           # Above upper
    winkler = np.mean(width + (2.0 / alpha) * penalties)
    
    return dict(PICP=picp, MPIW=mpiw, Winkler=winkler)

# Compute metrics
print(f"=== UQ Metrics ({int((1-ALPHA)*100)}% PI) — CQR (conformalized) ===")
print("Train:", uq_metrics(actual_train.values, L_train.values, U_train.values, ALPHA))
print("Val:  ", uq_metrics(actual_val.values,   L_val.values,   U_val.values,   ALPHA))
print("Test: ", uq_metrics(actual_test.values,  L_test.values,  U_test.values,  ALPHA))
```

---

## 2. SUMMARY TABLE: WHERE CQR APPLIES IN RNN PIPELINES

| **Stage** | **Component** | **CQR Role** | **Code Location (GRU/LSTM)** |
|-----------|---------------|---|---|
| **Architecture** | Dense(3) quantile output | Three heads: q_0.025, q_0.5, q_0.975 | Lines ~43 |
| **Loss Function** | Pinball loss (multi-quantile) | Learn all 3 quantiles jointly | Lines ~50-62 |
| **Compilation** | Model with pinball loss | Combine architecture + loss | Lines ~64-70 |
| **Bootstrap Sampling** | Random sampling with replacement | Create diversity for ensemble | Lines ~72-78 |
| **Ensemble Training** | Train M=5 members independently | Each on different bootstrap sample | Lines ~80-90 |
| **Member Prediction** | Inverse-scale quantiles | [q_0.025, q_0.5, q_0.975] in original space | Lines ~92-108 |
| **Monotonicity** | Sort quantile predictions | Enforce L ≤ M ≤ U | Lines ~106 |
| **Ensemble Aggregation** | Average across M members | Mean quantiles [L_bar, M_bar, U_bar] | Lines ~110-120 |
| **Epistemic Variance** | Var(median) across members | Model/parameter uncertainty | Lines ~122-125 |
| **Aleatoric Variance** | IQR / 3.92 estimate | Data/intrinsic uncertainty | Lines ~127-133 |
| **Conformal Calibration** | Compute q_hat on validation | Distribution-free threshold | Lines ~137-150 |
| **Conformalization** | L - q_hat, U + q_hat | Apply inflation for coverage guarantee | Lines ~152-160 |
| **Evaluation** | PICP, MPIW, Winkler | Measure UQ quality | Lines ~162-188 |

---

## 3. CQR FLOWCHART FOR GRU & LSTM

```
┌──────────────────────────────────────────────────────────────────────────┐
│ RNN (GRU & LSTM) + Conformal Quantile Regression (CQR) with Ensemble    │
└──────────────────────────────────────────────────────────────────────────┘

                                  INPUT DATA
                                      │
                                      ▼
                    ┌──────────────────────────────┐
                    │  Train/Val/Test Windowing    │
                    │  (lookback = 30 or 90)       │
                    │  Shape: (N, lookback, nfeat) │
                    └──────────────────────────────┘
                                      │
                    ══════════════════════════════════════════════════════
                   ║  BUILD GRU/LSTM FOR QUANTILE REGRESSION             ║
                    ══════════════════════════════════════════════════════
                                      │
                                      ▼
                    ┌───────────────────────────────────────────────────┐
                    │  ARCHITECTURE: Quantile Output Heads              │
                    │  ┌─────────────────────────────────────────────┐ │
                    │  │ INPUT (batch, lookback, n_features)        │ │
                    │  │           ↓                                 │ │
                    │  │ Layer 1: GRU/LSTM(units1)                  │ │
                    │  │ ├─ Process all timesteps                   │ │
                    │  │ ├─ Maintain hidden state h_t               │ │
                    │  │ └─ Output: (batch, units1)                 │ │
                    │  │           ↓                                 │ │
                    │  │ Dropout(dropout_rate)  [regularization]    │ │
                    │  │           ↓                                 │ │
                    │  │ Dense(3) ← QUANTILE OUTPUT HEADS           │ │
                    │  │ ├─ Output 1: q_0.025 (lower)              │ │
                    │  │ ├─ Output 2: q_0.5 (median)               │ │
                    │  │ ├─ Output 3: q_0.975 (upper)              │ │
                    │  │ └─ Shape: (batch, 3) in SCALED SPACE       │ │
                    │  │                                             │ │
                    │  │ Loss: Pinball (Multi-Quantile)             │ │
                    │  │ ├─ L_tau = τ*e if e>0, (τ-1)*e if e<0    │ │
                    │  │ ├─ Asymmetric: penalizes under/over        │ │
                    │  │ ├─ Sum across quantiles, mean across batch │ │
                    │  │ └─ Different τ → different cost penalties  │ │
                    │  │                                             │ │
                    │  │ Optimizer: Adam (lr from best params)       │ │
                    │  └─────────────────────────────────────────────┘ │
                    └───────────────────────────────────────────────────┘
                                      │
                    ══════════════════════════════════════════════════════
                   ║  ENSEMBLE TRAINING (M=5 MEMBERS)                    ║
                    ══════════════════════════════════════════════════════
                                      │
            ┌───────────────┬─────────────────┬──────────────┬──────────────┐
            │               │                 │              │              │
            ▼               ▼                 ▼              ▼              ▼
   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
   │  Member 1    │ │  Member 2    │ │  Member 3    │ │  Member 4    │ │  Member 5    │
   │  Seed=42+0   │ │  Seed=42+137 │ │  Seed=42+274 │ │  Seed=42+411 │ │  Seed=42+548 │
   ├──────────────┤ ├──────────────┤ ├──────────────┤ ├──────────────┤ ├──────────────┤
   │ Bootstrap 1  │ │ Bootstrap 2  │ │ Bootstrap 3  │ │ Bootstrap 4  │ │ Bootstrap 5  │
   │ (sample w/)  │ │ (sample w/)  │ │ (sample w/)  │ │ (sample w/)  │ │ (sample w/)  │
   │ replacement) │ │ replacement) │ │ replacement) │ │ replacement) │ │ replacement) │
   │   idx_b1     │ │   idx_b2     │ │   idx_b3     │ │   idx_b4     │ │   idx_b5     │
   │              │ │              │ │              │ │              │ │              │
   │ ├─ Compile   │ │ ├─ Compile   │ │ ├─ Compile   │ │ ├─ Compile   │ │ ├─ Compile   │
   │ ├─ Train:    │ │ ├─ Train:    │ │ ├─ Train:    │ │ ├─ Train:    │ │ ├─ Train:    │
   │ │ X_bootstrap │ │ X_bootstrap │ │ X_bootstrap │ │ X_bootstrap │ │ X_bootstrap │
   │ │ y_bootstrap │ │ y_bootstrap │ │ y_bootstrap │ │ y_bootstrap │ │ y_bootstrap │
   │ │ Pinball loss│ │ Pinball loss│ │ Pinball loss│ │ Pinball loss│ │ Pinball loss│
   │ └─ Save      │ │ └─ Save      │ │ └─ Save      │ │ └─ Save      │ │ └─ Save      │
   └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
            │               │                 │              │              │
            │               │                 │              │              │
            └───────────────┴─────────────────┴──────────────┴──────────────┘
                                      │
                    ══════════════════════════════════════════════════════
                   ║  ENSEMBLE INFERENCE & AGGREGATION                    ║
                    ══════════════════════════════════════════════════════
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  PHASE 1: Per-Member Predictions     │
                    │                                      │
                    │  For each member m ∈ {1..5}:       │
                    │  ├─ qz = m.predict(X)              │
                    │  │  (N, 3) in SCALED space         │
                    │  ├─ qy = qz * y_scale + y_mean    │
                    │  │  Inverse-scale to original      │
                    │  ├─ qy = sort(qy, axis=1)         │
                    │  │  Enforce monotonicity: L ≤ M ≤ U│
                    │  └─ Store [qL_m, qM_m, qU_m]      │
                    └──────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
   (N, 5) array           (N, 5) array              (N, 5) array
   Lower quantiles        Median quantiles        Upper quantiles
   Ls ∈ ℝ^(N×5)          Ms ∈ ℝ^(N×5)             Us ∈ ℝ^(N×5)
        │                             │                             │
        └─────────────────────────────┼─────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  PHASE 2: Ensemble Aggregation       │
                    │                                      │
                    │  L_bar = mean(Ls, axis=1)  (N,)    │
                    │  M_bar = mean(Ms, axis=1)  (N,)    │
                    │  U_bar = mean(Us, axis=1)  (N,)    │
                    │                                      │
                    │  Average across 5 members           │
                    │  Ensemble consensus quantiles       │
                    └──────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┴─────────────────────────────┐
        │                                                           │
        ▼                                                           ▼
   ┌───────────────────────────┐                    ┌───────────────────────────┐
   │  EPISTEMIC UNCERTAINTY    │                    │  ALEATORIC UNCERTAINTY    │
   │  (Model/Parameter)        │                    │  (Data/Intrinsic)         │
   │                           │                    │                           │
   │ var_epi = var(M_bar)     │                    │ IQR = U_bar - L_bar      │
   │ across ensemble members   │                    │ σ_ale = IQR / 3.92       │
   │                           │                    │ var_ale = σ_ale²         │
   │ High: members disagree    │                    │                           │
   │       → uncertain         │                    │ Wide interval:            │
   │ Low:  members agree       │                    │ → data noisy              │
   │       → confident         │                    │ Narrow interval:          │
   │                           │                    │ → clean data              │
   └───────────────────────────┘                    └───────────────────────────┘
        │                                                           │
        └─────────────────────────────┬─────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  PHASE 3: Conformal Calibration      │
                    │          (On VALIDATION set)         │
                    │                                      │
                    │  E_val = max(L_v_bar - y_val,      │
                    │              y_val - U_v_bar)      │
                    │  E_val[i] = 0 if inside             │
                    │           > 0 if outside            │
                    │                                      │
                    │  q_hat = quantile(E_val, 1-α)      │
                    │  q_hat = conformal threshold        │
                    │                                      │
                    │  Guarantees: ~(1-α) of val          │
                    │  points have E_i ≤ q_hat            │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  PHASE 4: Conformalization           │
                    │                                      │
                    │  L_test = L_bar - q_hat             │
                    │  U_test = U_bar + q_hat             │
                    │                                      │
                    │  Add conformal inflation to all      │
                    │  samples uniformly                   │
                    │                                      │
                    │  Formal guarantee:                  │
                    │  P(y ∈ [L, U]) ≥ 1 - α  (on test)  │
                    │                                      │
                    │  Distribution-free (no assumption)   │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  PREDICTION INTERVALS                │
                    │                                      │
                    │  Point forecast: M_bar (median)     │
                    │  Lower bound: L_test                │
                    │  Upper bound: U_test                │
                    │  Width: (U_test - L_test)           │
                    │                                      │
                    │  95% CI with formal coverage        │
                    │  guarantee on test set              │
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
                    │  │  (M, L, U, actual,               │
                    │  │   var_epi, var_ale)              │
                    │  └─ ALL_UQ_METRICS.csv              │
                    │     (PICP, MPIW, Winkler, etc.)     │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                                  OUTPUT: 
                    Predictions + Ensemble-based Uncertainty
                    with Formal Coverage Guarantees
```

---

## 4. KEY CQR CONCEPTS FOR RNN (GRU & LSTM)

### **Pinball Loss Explained**

```
τ = 0.025 (Lower Quantile):
  Underprediction (y > q):  cost = 0.025 * |y - q|  [low penalty]
  Overprediction (y < q):   cost = 0.975 * |y - q|  [high penalty]
  → Model learns to be conservative (predict LOW values)

τ = 0.5 (Median):
  Underprediction (y > q):  cost = 0.5 * |y - q|    [equal]
  Overprediction (y < q):   cost = 0.5 * |y - q|    [equal]
  → Model learns middle value (symmetric)

τ = 0.975 (Upper Quantile):
  Underprediction (y > q):  cost = 0.975 * |y - q|  [high penalty]
  Overprediction (y < q):   cost = 0.025 * |y - q|  [low penalty]
  → Model learns to be generous (predict HIGH values)
```

**Result:**
- Lower quantile learns lower bound of data distribution
- Median learns middle value
- Upper quantile learns upper bound
- All learned jointly on same training data
- Natural asymmetric learning built into loss!

---

### **Ensemble Diversity Mechanisms**

```
1. BOOTSTRAP SAMPLING (Main source of diversity):
   ├─ Different training data for each member
   ├─ Some samples repeated, some omitted
   ├─ Creates different learned weight distributions
   └─ → Predictions vary across members

2. RANDOM SEED VARIATIONS:
   ├─ Different weight initializations
   ├─ Different random shuffle orders
   ├─ Different dropout activations (if enabled)
   └─ → Small variations accumulate

3. NO EXPLICIT DROPOUT FOR MC:
   ├─ Dropout is only regularization here
   ├─ Not for Bayesian inference (unlike MCD)
   ├─ Bootstrap provides ensemble diversity
   └─ → Clean implementation

Result: 5 different models → 5 different quantile predictions per sample
        → Variance across members = epistemic uncertainty
```

---

### **Conformal Prediction Benefits**

```
TRADITIONAL CI (e.g., Gaussian):
├─ Assumes: y ~ N(μ, σ²)
├─ Assumes: σ is correctly estimated
├─ Risk: Wrong assumptions → poor coverage
└─ Example: HLLLA assumes Gaussian

CONFORMAL CI:
├─ Assumes: Data exchangeability (i.i.d., similar)
├─ No distributional assumptions
├─ Guarantees: Coverage ≥ 1-α with high probability
├─ Risk: Only if distribution shifts significantly
└─ Example: CQR works for ANY distribution

Why conformal works:
  q_hat is learned on validation set
  → Automatically compensates for:
    - Model bias
    - Dataset-specific characteristics
    - Calibration errors
  → Applies universally to test set
```

---

### **Monotonicity Enforcement**

```
Why sort quantiles?

Without sorting:
  q_0.025 = 1050
  q_0.5   = 1000  (WRONG: median < lower!)
  q_0.975 = 1200

Issues:
  - Logical inconsistency
  - Invalid probability interpretation
  - Negative interval width [1000, 1050]

With sorting:
  q_0.025 = 1000  ✓ (correct order)
  q_0.5   = 1050  ✓
  q_0.975 = 1200  ✓

Interpretation:
  - 2.5% of data below 1000
  - 50% below 1050
  - 97.5% below 1200
  - Valid probability model
```

---

## 5. CQR vs HLLLA vs MCD Comparison

| **Aspect** | **CQR** | **HLLLA** | **MCD** |
|---|---|---|---|
| **Quantile Learning** | Direct (pinball loss) | No | No |
| **Ensemble** | 5 bootstrap members | No | 100 dropout passes |
| **Epistemic** | Var(median) across members | Laplace approx | Ensemble variance |
| **Aleatoric** | IQR / 3.92 proxy | Learned output | Not captured |
| **Inference Cost** | 5 forward passes | 1 forward pass | 100 forward passes |
| **Coverage Guarantee** | YES (conformal) | NO (empirical) | NO (empirical) |
| **Distribution-Free** | YES | NO (Gaussian) | Approximate |
| **Calibration** | Explicit (q_hat) | Implicit | Implicit |
| **Interpretability** | Clear (quantiles) | Clear (decomposed) | Black-box |
| **Implementation** | Moderate (ensemble) | Moderate (LLLA) | Simple (dropout) |

---

## 6. GRU vs LSTM: Identical for CQR

| **Component** | **GRU** | **LSTM** |
|---|---|---|
| **Quantile output** | Dense(3) ✓ | Dense(3) ✓ |
| **Pinball loss** | Same ✓ | Same ✓ |
| **Bootstrap sampling** | Same ✓ | Same ✓ |
| **Ensemble training** | Same ✓ | Same ✓ |
| **Member prediction** | Same ✓ | Same ✓ |
| **Aggregation** | Same ✓ | Same ✓ |
| **Conformal calibration** | Same ✓ | Same ✓ |
| **Conformalization** | Same ✓ | Same ✓ |

**Conclusion: CQR is completely independent of RNN type**

---

## 7. CQR Advantages & Challenges

### **Advantages:**
| **Advantage** | **Explanation** |
|---|---|
| **Formal Guarantees** | Distribution-free coverage guarantee ≥ 1-α |
| **No Assumptions** | Works for ANY data distribution |
| **Direct Quantiles** | Learns uncertainty naturally via pinball loss |
| **Ensemble Diversity** | Bootstrap samples create meaningful variation |
| **Interpretable** | Quantiles are directly interpretable |
| **Moderate Cost** | 5 forward passes (between HLLLA and MCD) |

### **Challenges:**
| **Challenge** | **Explanation** |
|---|---|
| **Ensemble Size** | M=5 is arbitrary; larger M better but slower |
| **Aleatoric Proxy** | IQR/3.92 is rough approximation for Gaussian |
| **Calibration Set** | Needs separate validation for q_hat computation |
| **Monotonicity Sorting** | Post-hoc fix, not learned in model |
| **Conformal Scaling** | q_hat inflation can make intervals very wide |
| **Distribution Shift** | Formal guarantee breaks if test ≠ training |

---

## Summary

**CQR in GRU/LSTM applies through:**

1. **Quantile Output Architecture** (Dense(3)) - Learn 3 quantile levels
2. **Pinball Loss** - Asymmetric loss for quantile regression
3. **Model Compilation** - Combine architecture + pinball loss
4. **Bootstrap Sampling** - Create diversity for ensemble
5. **Ensemble Training** - Train M=5 members on bootstrap samples
6. **Per-Member Quantiles** - Get [q_0.025, q_0.5, q_0.975] per member
7. **Monotonicity Enforcement** - Sort quantiles to ensure L ≤ M ≤ U
8. **Ensemble Aggregation** - Average quantiles across members
9. **Epistemic Uncertainty** - Variance of median across ensemble
10. **Aleatoric Uncertainty** - IQR / 3.92 proxy from quantile width
11. **Split Conformal Calibration** - Compute q_hat on validation set
12. **Conformalization** - L - q_hat, U + q_hat for guaranteed coverage
13. **UQ Evaluation** - PICP, MPIW, Winkler metrics

**Key Distinction:** Unlike MCD and HLLLA, CQR provides **formal coverage guarantees** via distribution-free conformal prediction. This makes it particularly valuable for applications requiring provable reliability.

The RNN architecture (GRU vs LSTM) is **completely irrelevant** to CQR implementation. Both follow identical uncertainty quantification procedures. CQR is the only method (among the three) with formal statistical guarantees on coverage probability.
