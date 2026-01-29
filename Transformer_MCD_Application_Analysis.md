# Monte Carlo Dropout (MCD) in Transformer Pipeline: Detailed Analysis

## Overview
The Transformer_MCD_42.ipynb implements **Monte Carlo Dropout (MCD)** for uncertainty quantification in time-series forecasting using a Transformer-based architecture. This applies the same Bayesian approximation principles as TCN+MCD but leverages self-attention mechanisms instead of temporal convolutions.

---

## 1. WHERE MCD IS APPLIED IN THE TRANSFORMER PIPELINE

### **Stage 1: Transformer Architecture with Dropout (Lines ~113-144)**
**File Location:** `encoder_block()` and `build_transformer()` functions

```python
def encoder_block(x, num_heads, d_model, dff, dropout_rate):
    """
    Transformer encoder block with multi-head attention + feed-forward network
    Both with dropout layers for MCD capability
    """
    # Multi-head self-attention
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)(
        x, x, use_causal_mask=True  # ← Causal masking (autoregressive)
    )
    x = Add()([x, Dropout(dropout_rate)(attn)])      # ← DROPOUT LAYER 1
    x = LayerNormalization(epsilon=1e-6)(x)

    # Feed-forward network
    ff = Dense(dff, activation="relu")(x)
    ff = Dropout(dropout_rate)(ff)                    # ← DROPOUT LAYER 2
    ff = Dense(d_model)(ff)
    x = Add()([x, Dropout(dropout_rate)(ff)])        # ← DROPOUT LAYER 3
    x = LayerNormalization(epsilon=1e-6)(x)
    
    return x

def build_transformer(input_shape, params):
    """
    Build complete Transformer architecture
    """
    lb = params["lookback"]
    d_model = params["d_model"]
    num_heads = params["num_heads"]
    dff = params["dff"]
    dropout = params["dropout"]
    num_layers = params["num_layers"]

    inp = Input(shape=input_shape)
    
    # Project input to d_model dimension
    x = Dense(d_model)(inp)
    
    # Add positional encoding (critical for sequence position awareness)
    x = AddPE(lb, d_model)(x)
    
    # Stack encoder blocks (each with dropout)
    for _ in range(num_layers):
        x = encoder_block(
            x,
            num_heads=num_heads,
            d_model=d_model,
            dff=dff,
            dropout_rate=dropout  # ← Dropout rate applied per layer
        )

    # Extract last timestep and predict
    x = Lambda(lambda t: t[:, -1, :])(x)  # Take position -1: (batch, d_model)
    out = Dense(1)(x)                      # Single output (point estimate)

    model = Model(inputs=inp, outputs=out)
    opt = Adam(learning_rate=params["lr"])
    model.compile(optimizer=opt, loss="mse")
    
    return model
```

**Key Architecture Differences from TCN:**
```
TCN+MCD:
Input → Conv1D blocks + SpatialDropout1D → Dense(1)

Transformer+MCD:
Input → Dense(d_model) → Positional Encoding
  → Encoder blocks (with MultiHeadAttention + Dropout)
    → Extract last timestep → Dense(1)
```

**Dropout in Transformer Layers:**
- After attention: `Dropout(dropout_rate)(attn)`
- Within FF-network: `Dropout(dropout_rate)(ff)`
- After FF-output: `Dropout(dropout_rate)(ff)`
- Total: 3 dropout points per encoder block × num_layers

---

### **Stage 2: Hyperparameter Search (Random Search, not Optuna) (Lines ~146-199)**
**File Location:** `sample_params()` and random search loop

```python
def sample_params():
    """
    Sample random hyperparameters for Transformer
    """
    lookback = random.choice([30, 45, 60, 90])
    d_model = random.choice([32, 64, 96, 128])
    
    # Ensure d_model is divisible by num_heads
    valid_heads = [h for h in (2, 4, 8) if d_model % h == 0 and d_model // h >= 8]
    num_heads = random.choice(valid_heads)
    
    # Feed-forward hidden dimension (typically 2-4× d_model)
    dff = random.choice([2*d_model, 3*d_model, 4*d_model])

    params = {
        "lookback": lookback,
        "d_model": d_model,              # Model dimension
        "num_heads": num_heads,          # Number of attention heads
        "dff": dff,                      # FF hidden dimension
        "num_layers": random.choice([1, 2, 3]),  # Number of encoder blocks
        "dropout": np.random.uniform(0.0, 0.3),  # ← DROPOUT RATE (0-30%)
        "optimizer": "adam",
        "lr": 10 ** np.random.uniform(-4, math.log10(5e-3)),
        "batch_size": random.choice([32, 64, 128]),
        "epochs": random.choice([30, 40, 50, 60, 70, 80, 90, 100]),
        "patience": random.choice([5, 6, 7, 8, 9, 10])
    }
    
    return params

# Random search loop (lines 160-199)
best = {"val_loss": np.inf, "params": None}

for t in range(1, N_TRIALS + 1):  # N_TRIALS = 50
    params = sample_params()
    X_tr, y_tr, _ = make_windows(X_train_s, y_train_s, params["lookback"])
    X_vl, y_vl, _ = make_windows(X_val_s,   y_val_s,   params["lookback"])
    
    model = build_transformer((params["lookback"], len(feature_cols)), params)
    
    # Train with early stopping
    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_vl, y_vl),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        callbacks=[EarlyStopping(...)],
        verbose=0
    )
    
    val_loss = float(min(hist.history["val_loss"]))
    
    if val_loss < best["val_loss"]:
        best = {"val_loss": val_loss, "params": params}
```

**Key Point:** Dropout rate is optimized via random search (unlike Optuna which uses Bayesian optimization). This defines the **stochasticity level** for MCD.

---

### **Stage 3: Model Training (Lines ~200-220)**
**File Location:** Final training on combined train+val with best hyperparams

```python
# Combine train + val for final training
X_trainval_s = pd.concat([X_train_s, X_val_s], axis=0)
y_trainval_s = pd.concat([y_train_s, y_val_s], axis=0)

X_trv, y_trv, _ = make_windows(X_trainval_s, y_trainval_s, lb)

# Build model with best parameters
best_model = build_transformer((lb, len(feature_cols)), best["params"])

# Train with validation split and early stopping
hist_final = best_model.fit(
    X_trv, y_trv,
    validation_split=0.1,  # Use 10% of trainval as validation
    epochs=best["params"]["epochs"],
    batch_size=best["params"]["batch_size"],
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        ModelCheckpoint(MODEL_BEST, monitor="val_loss", save_best_only=True)
    ],
    verbose=VERBOSE_TRAIN
)
```

**During training:**
- Dropout is **active** (default Keras behavior with `training=True`)
- Model learns to use dropout as regularization
- Creates implicit ensemble during training

---

### **Stage 4: MCD Inference (Lines ~221-303)** ⭐ **CRITICAL MCD COMPONENT**
**File Location:** `predict_mc()` function

```python
# === MC DROPOUT INFERENCE ===

@tf.function
def mc_call(m, X, training=True):
    """
    Force dropout to stay ACTIVE during inference
    """
    return m(X, training=training)

def predict_mc(m, X_np, idx, n_mc=N_MC, alpha=ALPHA):
    """
    Monte Carlo Dropout forward passes
    
    m: trained Transformer model
    X_np: input data (N_samples, lookback, n_features)
    idx: time indices for output alignment
    n_mc: number of MC passes (typically 100)
    alpha: significance level (0.05 for 95% PI)
    """
    Ys_scaled = []
    X_tf = tf.convert_to_tensor(X_np, dtype=tf.float32)

    # ← MONTE CARLO LOOP: Run model N_MC times with dropout ACTIVE
    for _ in range(n_mc):
        # Forward pass through Transformer with all Dropout layers active
        y_s = mc_call(m, X_tf, training=True).numpy().squeeze()  # (N,)
        Ys_scaled.append(y_s)

    # Stack all predictions: (N_samples, N_MC)
    Ys_scaled = np.stack(Ys_scaled, axis=1)   # (N, 100)
    
    # Inverse scale to original space
    Ys = Ys_scaled * y_scale + y_mean        # (N, 100)

    # Compute statistics from 100 predictions per sample
    mean = Ys.mean(axis=1)                   # Point estimate (mean)
    std  = Ys.std(axis=1, ddof=1)            # Uncertainty (std dev)

    # Prediction intervals from empirical quantiles
    lower = np.quantile(Ys, q=alpha/2.0,     axis=1)      # 2.5th percentile
    upper = np.quantile(Ys, q=1-alpha/2.0,   axis=1)      # 97.5th percentile

    return (
        pd.Series(mean,  index=idx, name="mean"),
        pd.Series(lower, index=idx, name="lower"),
        pd.Series(upper, index=idx, name="upper"),
        pd.Series(std,   index=idx, name="mc_std"),
        Ys  # Full ensemble (N, 100) for uncertainty decomposition
    )

# Execute MCD on all splits (lines 277-282)
print("\n=== Running MC Dropout on Random-Search Best Transformer ===")
start_opt_mc = time.time()

# Run on train, val, test
mean_train, L_train, U_train, std_train, Ys_train = predict_mc(best_model, X_tr, idx_tr)
mean_val,   L_val,   U_val,   std_val,   Ys_val   = predict_mc(best_model, X_vl, idx_vl)
mean_test,  L_test,  U_test,  std_test,  Ys_test  = predict_mc(best_model, X_te, idx_te)

end_opt_mc = time.time()
print(f"[MC] Done in {end_opt_mc - start_opt_mc:.4f}s")
```

**Execution Flow:**
```
For each of 100 MC passes:
  1. Forward through Dense(d_model) → positional encoding
  2. Pass through encoder block 1:
     - MultiHeadAttention (no dropout in attention itself)
     - Dropout(0.1) → attention output
     - LayerNorm
     - Dense(dff) + Dropout(0.1)
     - Dense(d_model) + Dropout(0.1)
     - LayerNorm
  3. Repeat encoder block 2, 3, ... (num_layers times)
  4. Extract last timestep
  5. Dense(1) → single prediction
  6. Store this stochastic prediction

Result: 100 different predictions due to different dropout masks
```

---

### **Stage 5: Uncertainty Quantification (Lines ~284-330)**
**File Location:** UQ metrics computation

```python
def base_metrics(y_true, y_pred):
    """Point forecast metrics"""
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return dict(MSE=mse, MAE=mae, RMSE=rmse, MAPE=mape, R2=r2)

def uq_metrics(y_true, L, U, alpha=ALPHA):
    """Uncertainty quantification metrics"""
    y = np.asarray(y_true); L = np.asarray(L); U = np.asarray(U)
    
    # PICP: Prediction Interval Coverage Probability
    cover = (y >= L) & (y <= U)
    picp = cover.mean()
    
    # MPIW: Mean Prediction Interval Width
    mpiw = np.mean(U - L)
    
    # Winkler Score: penalizes width + miscoverage
    penalty = np.where(y < L, (2/alpha)*(L - y),
              np.where(y > U, (2/alpha)*(y - U), 0.0))
    winkler = np.mean((U - L) + penalty)
    
    return dict(PICP=picp, MPIW=mpiw, Winkler=winkler)

# Compute UQ metrics for all splits (lines 305-320)
uq_train = uq_metrics(actual_train.values, L_train.values, U_train.values)
uq_val   = uq_metrics(actual_val.values,   L_val.values,   U_val.values)
uq_test  = uq_metrics(actual_test.values,  L_test.values,  U_test.values)

print("\n=== MC Dropout UQ Metrics (95% PI) ===")
print("Train:", uq_train)
print("Val:  ", uq_val)
print("Test: ", uq_test)
```

---

### **Stage 6: Uncertainty Decomposition (Lines ~315-325)**
**File Location:** Epistemic vs Aleatoric estimation

```python
# Approximate aleatoric from validation residuals
resid_test = actual_test.values - mean_test.values
sigma2_aleatoric = np.var(resid_test, ddof=1)

# Total variance from 100 MC predictions per sample
var_total_test = np.var(Ys_test, axis=1, ddof=1)

# Epistemic = Total - Aleatoric
var_epistemic  = np.maximum(0.0, var_total_test - sigma2_aleatoric)
var_aleatoric  = np.full_like(var_total_test, sigma2_aleatoric)

epi_series = pd.Series(var_epistemic, index=idx_te, name="var_epistemic")
ale_series = pd.Series(var_aleatoric, index=idx_te, name="var_aleatoric")
tot_series = pd.Series(var_total_test, index=idx_te, name="var_total")
```

---

## 2. SUMMARY TABLE: WHERE MCD APPLIES IN TRANSFORMER

| **Stage** | **Component** | **MCD Role** | **Code Location** |
|-----------|---------------|---|---|
| **Architecture** | Dropout layers in encoder blocks | Stochastic regularization | Lines ~113-144 |
| **Encoder Blocks** | 3× Dropout per block (attention + FF) | Multiple dropout points | Lines ~118-125 |
| **Dropout Rate** | Sampled via random search | 0-30% per layer | Lines ~130-150 |
| **Training** | Standard fit() with dropout active | Dropout acts as regularizer | Lines ~200-220 |
| **HPO** | Random search (N_TRIALS=50) | Find best dropout rate | Lines ~160-199 |
| **Best Model Save** | ModelCheckpoint callback | Store trained weights | Lines ~220 |
| **Inference** | `predict_mc()` function | **Force training=True** | Lines ~221-303 |
| **MC Loop** | For i=1 to 100 iterations | Generate ensemble | Lines ~264-268 |
| **Forward Pass** | `mc_call(model, X, training=True)` | Dropout ACTIVE | Lines ~264-265 |
| **Ensemble** | Stack 100 predictions | (N_samples, 100) shape | Lines ~270 |
| **Statistics** | mean, std, quantiles | From 100 predictions | Lines ~273-280 |
| **Decomposition** | Epistemic/Aleatoric variance | Uncertainty analysis | Lines ~315-325 |
| **Evaluation** | PICP, MPIW, Winkler | Measure UQ quality | Lines ~305-320 |

---

## 3. MCD FLOWCHART FOR TRANSFORMER

```
┌────────────────────────────────────────────────────────────────────────────┐
│         TRANSFORMER + MONTE CARLO DROPOUT (MCD) FOR UNCERTAINTY             │
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
                    │  Random Search (N=50)        │
                    │  Sample hyperparameters:     │
                    │  - d_model                   │
                    │  - num_heads                 │
                    │  - num_layers                │
                    │  - dropout rate (0-30%)      │
                    │  - lr, batch_size, epochs    │
                    └──────────────────────────────┘
                                      │
                    ══════════════════════════════════════════
                   ║   BUILD TRANSFORMER WITH DROPOUT        ║
                    ══════════════════════════════════════════
                                      │
                                      ▼
                    ┌───────────────────────────────────────────────┐
                    │  Transformer Architecture                     │
                    │  ┌─────────────────────────────────────────┐ │
                    │  │ INPUT (batch, lookback, n_features)    │ │
                    │  │           ↓                             │ │
                    │  │ Dense(d_model)  → (batch, lb, d_model) │ │
                    │  │           ↓                             │ │
                    │  │ Positional Encoding (add position info) │ │
                    │  │           ↓                             │ │
                    │  │ FOR each encoder layer (num_layers):   │ │
                    │  │ ┌───────────────────────────────────┐ │ │
                    │  │ │ MultiHeadAttention (causal mask) │ │ │
                    │  │ │           ↓                       │ │ │
                    │  │ │ Dropout(dropout_rate) ◄─ DROPOUT1│ │ │
                    │  │ │           ↓                       │ │ │
                    │  │ │ Add (residual) + LayerNorm        │ │ │
                    │  │ │           ↓                       │ │ │
                    │  │ │ Dense(dff, relu)                 │ │ │
                    │  │ │           ↓                       │ │ │
                    │  │ │ Dropout(dropout_rate) ◄─ DROPOUT2│ │ │
                    │  │ │           ↓                       │ │ │
                    │  │ │ Dense(d_model)                   │ │ │
                    │  │ │           ↓                       │ │ │
                    │  │ │ Dropout(dropout_rate) ◄─ DROPOUT3│ │ │
                    │  │ │           ↓                       │ │ │
                    │  │ │ Add (residual) + LayerNorm        │ │ │
                    │  │ └───────────────────────────────────┘ │ │
                    │  │           ↓ (repeat num_layers times)  │ │
                    │  │ Lambda: Take last timestep             │ │
                    │  │           ↓                             │ │
                    │  │ Dense(1)  → OUTPUT (batch, 1)          │ │
                    │  │                                         │ │
                    │  │ Loss: MSE                               │ │
                    │  │ Optimizer: Adam (lr from search)        │ │
                    │  └─────────────────────────────────────────┘
                    └───────────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────┐
                    │  TRAIN PHASE                 │
                    │  ├─ fit(X_train, y_train)   │
                    │  ├─ Dropout active          │
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
                    ══════════════════════════════════════
                   ║  START MCD INFERENCE (NEW)          ║
                    ══════════════════════════════════════
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  MONTE CARLO LOOP (N_MC=100)        │
                    │  for i in range(100):               │
                    │    ├─ Input X to model              │
                    │    ├─ Pass through encoder:         │
                    │    │  FOR each encoder layer:       │
                    │    │   - Attention output           │
                    │    │   - Dropout(dropout_rate) ◄─╮  │
                    │    │     ACTIVE (training=True)    │  │
                    │    │   - Dense(dff)                │  │
                    │    │   - Dropout(dropout_rate) ◄─╮ │  │
                    │    │     DIFFERENT MASK each time  │  │
                    │    │   - Dense(d_model)            │  │
                    │    │   - Dropout(dropout_rate) ◄─╮ │  │
                    │    ├─ Extract last timestep       │  │
                    │    ├─ Dense(1) → ŷ_i(x, θ_i)     │  │
                    │    │   (stochastic prediction)     │  │
                    │    └─ Store in ensemble           │  │
                    │                                    │  │
                    │  Due to 3× Dropout per layer ×    │  │
                    │  num_layers → 3×num_layers        │  │
                    │  random masks per pass            │  │
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

## 4. KEY MCD CONCEPTS FOR TRANSFORMER

### **The Core Difference: Transformer vs TCN**

**TCN+MCD:**
```
SpatialDropout1D (temporal/feature-wise masking)
Default: 1 dropout point per Conv1D block
Multiple blocks in series
```

**Transformer+MCD:**
```
Standard Dropout (neuron-wise masking)
Multiple dropout points per encoder block:
  - After attention
  - Within FF network (2 places)
Default: 3 dropout per encoder × num_layers
Parallel attention heads (more noise tolerance)
```

### **Dropout Placement in Transformer**

```python
# Per encoder block: 3 dropout layers

Attention Output:
  y = attn(x, x)
  y = Dropout(p)(y)          # ← Dropout 1 of 3
  y = Residual(y) + x
  y = LayerNorm(y)

Feed-Forward:
  ff = Dense(dff)(y)
  ff = Dropout(p)(ff)        # ← Dropout 2 of 3
  ff = Dense(d_model)(ff)
  ff = Dropout(p)(ff)        # ← Dropout 3 of 3
  ff = Residual(ff) + y
  ff = LayerNorm(ff)
```

With `num_layers=2`: **6 dropout masks per forward pass**
With `num_layers=3`: **9 dropout masks per forward pass**

This creates **more diverse stochastic samples** than TCN (which typically has 1-2 dropout per block).

### **Positional Encoding (Unique to Transformer)**

```python
def positional_encoding(length, depth):
    """
    Sinusoidal positional encoding
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    positions = np.arange(length)[:, np.newaxis]
    dims = np.arange(depth)[np.newaxis, :]
    angle_rates = 1.0 / (10000 ** (2 * (dims//2) / depth))
    angle_rads = positions * angle_rates
    pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return pe
```

This encoding ensures the model knows the **temporal position** of each token, which is crucial since Transformer has no inherent sequence awareness like RNNs/CNNs.

### **Multi-Head Attention with Dropout**

```python
attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)(
    x, x, use_causal_mask=True  # Causal: can't attend to future
)
# attn shape: (batch, lookback, d_model)
x = Add()([x, Dropout(dropout_rate)(attn)])
```

Dropout is applied **after** attention, not within the attention heads. This affects the **entire attention output**, not individual heads.

---

## 5. RANDOM SEARCH VS OPTUNA

**TCN+MCD used Optuna:**
- Bayesian optimization with Parzen Estimators
- Early stopping of unpromising trials
- More efficient sampling
- Typically requires fewer trials for convergence

**Transformer+MCD uses Random Search:**
- Purely random hyperparameter sampling
- All trials run to completion
- Less efficient but simpler to implement
- Requires more trials (N=50) to find good configurations

### **Hyperparameter Space Comparison**

| **Aspect** | **TCN** | **Transformer** |
|-----------|--------|-----------------|
| **Lookback** | [30, 45, 60, 90] | [30, 45, 60, 90] |
| **Filters/d_model** | [32, 64, 96, 128] | [32, 64, 96, 128] |
| **Kernel size/Heads** | [3, 5, 7] | Computed from d_model |
| **Dropout** | [0, 0.5] | [0, 0.3] |
| **Depth** | num_stacks: [1, 2] | num_layers: [1, 2, 3] |
| **Dilation** | [(1,2,4), (1,2,4,8)] | dff: [2×, 3×, 4× d_model] |
| **Epochs** | [30-100] | [30-100] |
| **HPO Method** | Optuna (Bayesian) | Random search |

---

## 6. COMPUTATIONAL COMPARISON

### **Training Phase**

```
Transformer random search:
  50 trials × (variable epochs per trial) + early stopping
  Typical: 50 × 40 epochs average = 2000 epoch-equivalents

TCN Optuna:
  50 trials with pruning (stops bad trials early)
  Typical: 50 × 20-30 epochs average = 1000-1500 epoch-equivalents
```

### **Inference Phase (per split)**

```
Transformer + MCD:
  For each of N_MC=100 passes:
    - Pass through 3 encoder blocks (assuming num_layers=2-3)
    - Each block: attention + 2 FF layers
    - 3 dropout masks per block × 3 blocks = 9 stochastic points
  Total: 100 forward passes with different dropout masks

Computational cost: ~100× single inference
Memory: Store (N_samples, 100) matrix of predictions
```

---

## 7. KEY ADVANTAGES OF TRANSFORMER+MCD

| **Advantage** | **Explanation** |
|---|---|
| **Multi-Head Attention** | Parallel attention reduces sensitivity to single dropout mask |
| **Positional Encoding** | Explicit position awareness (vs TCN implicit via receptive field) |
| **Scalability** | Easier to add depth (more encoder layers) than TCN |
| **Flexibility** | Can adjust num_heads independently of hidden dimension |
| **Research Evidence** | Transformers well-studied in NLP with MCD UQ |

## 8. KEY CHALLENGES OF TRANSFORMER+MCD

| **Challenge** | **Explanation** |
|---|---|
| **Many Dropout Points** | 3 per encoder block → more randomness per pass (vs TCN) |
| **Longer Inference** | Attention is O(L²) where L=lookback (vs TCN's O(L)) |
| **Hyperparameter Tuning** | More interdependencies (d_model, num_heads, dff) |
| **Random Search** | Less efficient than Bayesian optimization (requires N=50 trials) |
| **Memory Usage** | Attention mechanisms require storing all token interactions |

---

## Summary

**MCD in Transformer applies through:**

1. **Dropout in Encoder Blocks** (3 per block) - Stochastic regularization
2. **Random Search HPO** (50 trials) - Find optimal dropout rate
3. **Training** - Dropout active (standard Keras behavior)
4. **Inference** - `training=True` forces dropout to stay active
5. **100 Forward Passes** - Generate diverse predictions
6. **Ensemble Statistics** - Mean, std, quantiles from 100 samples
7. **Uncertainty Decomposition** - Epistemic vs Aleatoric variance

The Transformer architecture with MCD provides **more stochastic diversity** (multiple dropout points per block) compared to TCN, potentially capturing uncertainty better at the cost of higher computational inference cost.
