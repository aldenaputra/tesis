# Methodology: Uncertainty Quantification in RNN-Based Time-Series Forecasting

## 1. Introduction to Uncertainty Quantification in Recurrent Neural Networks

Recurrent Neural Networks (RNNs), particularly GRU (Gated Recurrent Unit) and LSTM (Long Short-Term Memory), have become the standard architecture for time-series forecasting due to their ability to capture sequential dependencies and long-term patterns. However, point predictions alone are insufficient for decision-making in critical applications such as financial forecasting, energy demand prediction, and risk management. **Uncertainty quantification (UQ)** extends point predictions to provide prediction intervals and probabilistic statements about model confidence.

This section implements three complementary UQ methodologies for RNN-based forecasting:

1. **Monte Carlo Dropout (MCD)** — Bayesian approximation via stochastic inference
2. **Heteroscedastic Last-Layer Laplace Approximation (HLLLA)** — Parametric epistemic uncertainty with learned aleatoric uncertainty
3. **Conformal Quantile Regression (CQR)** — Distribution-free formal guarantees via quantile learning and conformal prediction

These methods address different aspects of uncertainty and make distinct assumptions, enabling comprehensive assessment of model reliability across multiple perspectives.

---

## 2. RNN Architectures for Uncertainty Quantification

### 2.1 GRU (Gated Recurrent Unit)

The GRU processes sequential input $(x_1, x_2, \ldots, x_T)$ of length $T$ (lookback window) through gating mechanisms:

$$\mathbf{r}_t = \sigma(\mathbf{W}_r \mathbf{x}_t + \mathbf{U}_r \mathbf{h}_{t-1} + \mathbf{b}_r) \quad \text{(reset gate)}$$

$$\mathbf{z}_t = \sigma(\mathbf{W}_z \mathbf{x}_t + \mathbf{U}_z \mathbf{h}_{t-1} + \mathbf{b}_z) \quad \text{(update gate)}$$

$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h \mathbf{x}_t + \mathbf{U}_h (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_h) \quad \text{(candidate state)}$$

$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \tilde{\mathbf{h}}_t + \mathbf{z}_t \odot \mathbf{h}_{t-1} \quad \text{(hidden state update)}$$

The final hidden state $\mathbf{h}_T$ is passed to task-specific output layers for point prediction or uncertainty quantification.

### 2.2 LSTM (Long Short-Term Memory)

The LSTM extends the GRU with an explicit cell state $\mathbf{c}_t$:

$$\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f) \quad \text{(forget gate)}$$

$$\mathbf{i}_t = \sigma(\mathbf{W}_i \mathbf{x}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i) \quad \text{(input gate)}$$

$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c \mathbf{x}_t + \mathbf{U}_c \mathbf{h}_{t-1} + \mathbf{b}_c) \quad \text{(cell candidate)}$$

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \quad \text{(cell state)}$$

$$\mathbf{o}_t = \sigma(\mathbf{W}_o \mathbf{x}_t + \mathbf{U}_o \mathbf{h}_{t-1} + \mathbf{b}_o) \quad \text{(output gate)}$$

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t) \quad \text{(hidden state)}$$

### 2.3 Architecture Equivalence for UQ

For uncertainty quantification purposes, **GRU and LSTM are functionally equivalent**. Both:
- Process sequences identically through gating/memory mechanisms
- Output a final hidden state for prediction
- Allow identical output layers for different UQ tasks
- Respond to UQ methodologies (dropout, uncertainty heads, quantile regression) identically

The choice between GRU and LSTM in this work is empirically driven by best hyperparameters discovered during optimization, not by fundamental UQ differences. All three UQ methodologies apply identically to both architectures.

---

## 3. Three Uncertainty Quantification Methodologies for RNNs

### 3.1 Monte Carlo Dropout (MCD)

**Core Concept:** MCD treats dropout as a Bayesian approximation (Gal & Ghahramani, 2016), where stochastic forward passes with active dropout approximate posterior inference over model parameters.

**Architecture:**
- Standard RNN (GRU/LSTM) with dropout applied at two locations:
  - After the RNN layer(s)
  - After intermediate dense layers
- Single output head (point prediction): $\hat{y} \in \mathbb{R}$
- Loss: Mean Squared Error (MSE)

**Inference Procedure:**
- During inference, set `training=True` to force dropout active
- Perform $T=100$ stochastic forward passes: $\{\hat{y}^{(1)}, \hat{y}^{(2)}, \ldots, \hat{y}^{(T)}\}$
- Each pass produces different prediction due to random dropout masking
- Aggregate via ensemble statistics:
  - **Point forecast:** $\mu = \frac{1}{T}\sum_{t=1}^{T} \hat{y}^{(t)}$ (mean)
  - **Standard deviation:** $\sigma_{\text{MCD}} = \sqrt{\frac{1}{T}\sum_{t=1}^{T}(\hat{y}^{(t)} - \mu)^2}$ (sample std)
  - **Quantiles:** Empirical quantiles from $T$ samples

**Uncertainty Decomposition:**
$$\sigma_{\text{total}}^2 = \sigma_{\text{epistemic}}^2 + \sigma_{\text{aleatoric}}^2$$

Where:
- **Epistemic:** $\sigma_{\text{epistemic}}^2 = \mathbb{E}[\sigma^2]$ (average variance)
- **Aleatoric:** $\sigma_{\text{aleatoric}}^2 = \mathbb{Var}[\mu]$ (variance of means)

---

### 3.2 Heteroscedastic Last-Layer Laplace Approximation (HLLLA)

**Core Concept:** HLLLA combines learned heteroscedastic uncertainty (aleatoric) with Bayesian posterior approximation on the final layer (epistemic), providing a principled decomposition without ensemble multiplicity.

**Architecture:**
- RNN (GRU/LSTM) followed by dense layer
- Dual output heads:
  - Head 1: Mean prediction $\mu_s$ (standardized)
  - Head 2: Log-variance $\log(\sigma_s^2)$ (standardized)
- Loss: Heteroscedastic Gaussian Negative Log-Likelihood (NLL):
$$\mathcal{L}_{\text{NLL}} = \frac{1}{N}\sum_{i=1}^{N} \left[\frac{1}{2}\log(\sigma_i^2) + \frac{(y_i - \mu_i)^2}{2\sigma_i^2}\right]$$

**Aleatoric Uncertainty (Learned):**
- Directly from model output: $\sigma_{\text{aleatoric}}^2 = \text{softplus}(\log(\sigma_s^2))$
- Per-sample, heteroscedastic (varies with input)

**Epistemic Uncertainty (Laplace Approximation on Last Layer):**

1. Extract penultimate layer features: $\boldsymbol{\phi}(\mathbf{x}) \in \mathbb{R}^{d}$ (RNN output)
2. Extend with bias term: $\boldsymbol{\Phi} = [\boldsymbol{\phi}(\mathbf{x}); 1] \in \mathbb{R}^{d+1}$
3. Approximate posterior Hessian (diagonal):
$$\mathbf{H}_{\text{diag}} = \frac{1}{\sigma_{\text{noise}}^2} \sum_{i=1}^{N} \boldsymbol{\Phi}_i \boldsymbol{\Phi}_i^\top + \lambda \mathbf{I}$$

4. Posterior covariance: $\boldsymbol{\Sigma} = \mathbf{H}^{-1}$
5. Epistemic variance (per-sample):
$$\sigma_{\text{epistemic}}^2 = \boldsymbol{\Phi} \boldsymbol{\Sigma} \boldsymbol{\Phi}^\top$$

**Total Uncertainty:**
$$\sigma_{\text{total}}^2 = \sigma_{\text{epistemic}}^2 + \sigma_{\text{aleatoric}}^2$$

---

### 3.3 Conformal Quantile Regression (CQR)

**Core Concept:** CQR learns prediction intervals directly via quantile regression on an ensemble of RNN models, then applies distribution-free conformal calibration to guarantee coverage.

**Architecture:**
- RNN (GRU/LSTM) with single output layer
- **Quantile output heads (3 heads):**
  - Head 1: Lower quantile $q_{0.025}$ (2.5th percentile)
  - Head 2: Median quantile $q_{0.5}$ (50th percentile)
  - Head 3: Upper quantile $q_{0.975}$ (97.5th percentile)
- Loss: Multi-quantile pinball loss

**Pinball Loss (Asymmetric Absolute Loss):**
$$\mathcal{L}_{\text{pinball}}(\boldsymbol{\tau}) = \frac{1}{N}\sum_{i=1}^{N} \sum_{k=1}^{3} \max(\tau_k \cdot e_i^{(k)}, (\tau_k - 1) \cdot e_i^{(k)})$$

Where $e_i^{(k)} = y_i - q_{\tau_k}(x_i)$ and $\boldsymbol{\tau} = [0.025, 0.5, 0.975]$.

**Ensemble Training:**
- Train $M=5$ independent RNN models
- Each member trained on bootstrap sample (sampling with replacement)
- Different random seeds → different weight initialization and gradient flow
- **Diversity mechanism:** Bootstrap sampling + seed variation

**Ensemble Inference:**
1. Predict quantiles from each member: $m \in \{1, 2, \ldots, 5\}$
2. Aggregate via averaging:
   - $\bar{q}_{0.025} = \frac{1}{M}\sum_{m=1}^{M} q_{0.025}^{(m)}$
   - $\bar{q}_{0.5} = \frac{1}{M}\sum_{m=1}^{M} q_{0.5}^{(m)}$
   - $\bar{q}_{0.975} = \frac{1}{M}\sum_{m=1}^{M} q_{0.975}^{(m)}$

**Uncertainty Decomposition:**
- **Epistemic:** $\sigma_{\text{epistemic}}^2 = \text{Var}[\bar{q}_{0.5}]$ (variance of ensemble median across members)
- **Aleatoric:** $\sigma_{\text{aleatoric}}^2 = \left(\frac{\text{IQR}}{3.92}\right)^2$ (Gaussian approximation from inter-quantile range)

Where $\text{IQR} = \bar{q}_{0.975} - \bar{q}_{0.025}$.

**Split Conformal Calibration (Distribution-Free Coverage Guarantee):**

1. On **validation set**, compute non-conformity scores:
$$E_{\text{val},i} = \max(\bar{q}_{0.025,i} - y_i, y_i - \bar{q}_{0.975,i}, 0)$$

2. Compute conformal threshold:
$$\hat{q} = \text{quantile}\left(\{E_{\text{val},i}\}_{i=1}^{n_{\text{val}}}, \left\lceil\frac{(n_{\text{val}}+1)(1-\alpha)}{n_{\text{val}}}\right\rceil\right)$$

3. **Conformalize intervals on all splits:**
$$[\bar{q}_{0.025} - \hat{q}, \bar{q}_{0.975} + \hat{q}]$$

**Formal Guarantee:** With high probability over validation set draws:
$$P(y \in [\bar{q}_{0.025} - \hat{q}, \bar{q}_{0.975} + \hat{q}]) \geq 1 - \alpha$$

This guarantee is **distribution-free** (no parametric assumptions) and **finite-sample valid** (provable with finite data).

---

## 4. Comprehensive Comparison Table: MCD vs HLLLA vs CQR for RNN-Based Forecasting

| **Dimension** | **MCD (Monte Carlo Dropout)** | **HLLLA (Heteroscedastic NLL + Laplace)** | **CQR (Conformal Quantile Regression)** |
|---|---|---|---|
| **UQ Category** | Bayesian Approximation | Hybrid (Parametric + Bayesian) | Distribution-Free |
| **Core Assumption** | Dropout ≈ Bayesian posterior | Gaussian likelihood + Laplace posterior | Data exchangeability (i.i.d.-like) |
| **RNN Architecture** | Standard (GRU/LSTM) | Standard (GRU/LSTM) | Standard (GRU/LSTM) |
| **Output Heads** | 1 (point: $\hat{y}$) | 2 (mean: $\mu$, logvar: $\log\sigma^2$) | 3 (quantiles: $q_{0.025}, q_{0.5}, q_{0.975}$) |
| **Training Loss** | MSE | Heteroscedastic Gaussian NLL | Multi-Quantile Pinball Loss |
| **Loss Equation** | $\frac{1}{N}\sum (y_i - \hat{y}_i)^2$ | $\frac{1}{N}\sum[\frac{1}{2}\log\sigma_i^2 + \frac{(y_i-\mu_i)^2}{2\sigma_i^2}]$ | $\frac{1}{N}\sum[\max(\tau \cdot e, (\tau-1) \cdot e)]$ |
| **Ensemble Method** | Stochastic inference (dropout) | Single forward pass | Bootstrap ensemble (M=5 members) |
| **Ensemble Size** | T=100 MC passes | N/A | M=5 models |
| **Dropout Configuration** | Active during inference | Inference-time dropout OFF | Training-time dropout only |
| **Aleatoric Uncertainty** | Aggregated from ensemble | Learned from output head | Proxy from IQR |
| **Epistemic Uncertainty** | Var(ensemble) | Laplace on final layer weights | Var(ensemble median) |
| **Uncertainty Formula** | $\sigma_{\text{epi}}^2 = \mathbb{E}[\sigma^2]$, $\sigma_{\text{ale}}^2 = \mathbb{Var}[\mu]$ | $\sigma_{\text{ale}}^2 = \text{softplus}(\log\sigma_s^2)$, $\sigma_{\text{epi}}^2 = \boldsymbol{\Phi}\boldsymbol{\Sigma}\boldsymbol{\Phi}^\top$ | $\sigma_{\text{epi}}^2 = \text{Var}[\bar{q}_{0.5}]$, $\sigma_{\text{ale}}^2 = (\text{IQR}/3.92)^2$ |
| **Inference Cost** | 100 forward passes (slow) | 1 forward pass (fast) | 5 forward passes (moderate) |
| **Calibration** | Implicit (learned on training) | Implicit (learned on training) | Explicit (conformal on validation) |
| **Coverage Guarantee** | Empirical (no proof) | Empirical (no proof) | Formal (distribution-free proof) |
| **Confidence Interval** | $[\mu - 1.96\sigma, \mu + 1.96\sigma]$ (Gaussian approx) | Same as MCD | Direct quantiles $[q_{0.025}, q_{0.975}]$ |
| **Parametric Assumption** | Implicit Gaussian | Explicit Gaussian | None (distribution-free) |
| **Hyperparameters (Training)** | Dropout rate, learning rate, epochs | Dropout rate, learning rate, epochs, Laplace λ | Learning rate, epochs, bootstrap seed, M |
| **Hyperparameters (Inference)** | T (MC passes) | None | q̂ (computed on validation) |
| **Strengths** | Simple, interpretable, Bayesian-grounded | Fast inference, explicit aleatoric learning | Formal guarantees, interpretable quantiles |
| **Limitations** | Slow inference, Gaussian assumption, empirical coverage | Gaussian assumption, fixed epistemic/aleatoric split | Wider intervals from conformal, ensemble overhead |
| **Best For** | Understanding Bayesian approximation | Applications requiring fast inference | Critical applications needing formal guarantees |
| **Validation Set Usage** | No | No | YES (for conformal calibration) |
| **Monotonicity Constraint** | N/A | N/A | Post-hoc sorting (L ≤ M ≤ U) |
| **Implementation Complexity** | Low | Moderate (Laplace approximation) | Moderate (ensemble, conformal) |

---

## 5. Detailed Implementation Overview

### 5.1 Common Pipeline Steps (All Three Methods)

All three UQ methodologies follow identical data preparation steps:

1. **Time-Series Windowing:**
   - Lookback window: $T = 30$ (LSTM) or $T = 90$ (GRU)
   - Input shape: $(N, T, n_{\text{features}})$
   - Target shape: $(N,)$

2. **Data Standardization:**
   - Features: StandardScaler fitted on training set
   - Target: Separate StandardScaler (inverse-transformed for final predictions)

3. **Train/Val/Test Split:**
   - Training: 70% (used for model fitting)
   - Validation: 10% (used for conformal calibration in CQR, early stopping in all)
   - Test: 20% (evaluation set, unseen during training)

4. **Model Architecture Base:**
   - GRU: Single layer with units1=224, dropout=0.043
   - LSTM: Single layer with units1=224, dropout=0.259
   - Hyperparameters optimized via Optuna (50 trials)

### 5.2 Method-Specific Training Procedures

#### MCD Training:
```
For each RNN sample (X, y):
  1. Forward pass: ŷ = RNN(X) with dropout active
  2. Compute MSE loss: L = (y - ŷ)²
  3. Backpropagate and update weights
  4. Apply early stopping on validation set
```

#### HLLLA Training:
```
For each RNN sample (X, y):
  1. Forward pass: [μ, log(σ²)] = RNN(X) with 2 output heads
  2. Compute NLL loss: L = 0.5*[log(σ²) + (y-μ)²/σ²]
  3. Backpropagate and update weights
  4. Apply early stopping on validation set
  5. After training, compute Laplace approximation:
     - Extract penultimate features φ(x)
     - Compute diagonal Hessian H
     - Posterior covariance Σ = H⁻¹
```

#### CQR Training (Ensemble):
```
For m in {1, 2, 3, 4, 5}:
  1. Sample bootstrap indices: idx_b ~ sample(n, n, replace=True, seed=42+137*m)
  2. For each bootstrap sample (X_b, y_b):
     a. Forward pass: [q₀.₀₂₅, q₀.₅, q₀.₉₇₅] = RNN(X_b) with 3 output heads
     b. Compute pinball loss: L = Σ max(τ*e, (τ-1)*e) for τ ∈ {0.025, 0.5, 0.975}
     c. Backpropagate and update weights
     d. Apply early stopping on validation set
     e. Save trained model
```

### 5.3 Inference Procedures

#### MCD Inference:
```
Input: X_test (N, lookback, n_features)
Output: Prediction intervals [L, U] for each sample

For i in {1, 2, ..., N}:
  For t in {1, 2, ..., 100}:
    ŷ_t(i) = RNN(X_i) with training=True (dropout active)
  
  Aggregate:
    μ_i = mean({ŷ_t(i)})
    σ_i = std({ŷ_t(i)})
    q_α_i = quantile({ŷ_t(i)}, α)
  
  Output: [μ_i - 1.96*σ_i, μ_i + 1.96*σ_i]
```

#### HLLLA Inference:
```
Input: X_test (N, lookback, n_features)
Output: Prediction intervals [L, U] for each sample

For i in {1, 2, ..., N}:
  1. Forward pass: [μ_i, log(σ²_i)] = RNN(X_i)
  2. Aleatoric: σ_ale,i = softplus(log(σ²_i))
  3. Extract features: φ_i = RNN_penultimate(X_i)
  4. Epistemic: σ_epi,i = sqrt(φ_i Σ φ_i^T)  [Laplace approximation]
  5. Total: σ_total,i = sqrt(σ_epi,i² + σ_ale,i²)
  
  Output: [μ_i - 1.96*σ_total,i, μ_i + 1.96*σ_total,i]
```

#### CQR Inference:
```
Input: X_test (N, lookback, n_features), X_val, y_val (for conformal)
Output: Prediction intervals [L, U] with formal coverage guarantee

Phase 1: Validation (offline, done once):
  For i in {1, 2, ..., N_val}:
    For m in {1, 2, 3, 4, 5}:
      [q₀.₀₂₅^(m), q₀.₅^(m), q₀.₉₇₅^(m)] = Model_m(X_val,i)
    
    q̄₀.₀₂₅,i = mean({q₀.₀₂₅^(m)})
    q̄₀.₉₇₅,i = mean({q₀.₉₇₅^(m)})
    E_i = max(q̄₀.₀₂₅,i - y_val,i, y_val,i - q̄₀.₉₇₅,i, 0)
  
  q̂ = quantile({E_i}, 1-α)  [conformal threshold]

Phase 2: Test (online, per sample):
  For i in {1, 2, ..., N_test}:
    For m in {1, 2, 3, 4, 5}:
      [q₀.₀₂₅^(m), q₀.₅^(m), q₀.₉₇₅^(m)] = Model_m(X_test,i)
    
    q̄₀.₀₂₅,i = mean({q₀.₀₂₅^(m)})
    q̄₀.₅,i = mean({q₀.₅^(m)})
    q̄₀.₉₇₅,i = mean({q₀.₉₇₅^(m)})
    
    Conformalize: L_i = q̄₀.₀₂₅,i - q̂, U_i = q̄₀.₉₇₅,i + q̂
  
  Output: [L_i, U_i] with P(y ∈ [L, U]) ≥ 0.95 (guaranteed)
```

---

## 6. Evaluation Framework

### 6.1 Point Forecast Metrics

All three methods provide point predictions used for regression quality assessment:

| **Metric** | **Formula** | **Interpretation** |
|---|---|---|
| **MAE** | $\frac{1}{N}\sum\|y_i - \hat{y}_i\|$ | Average absolute error |
| **RMSE** | $\sqrt{\frac{1}{N}\sum(y_i - \hat{y}_i)^2}$ | Root mean squared error |
| **MAPE** | $\frac{1}{N}\sum\frac{\|y_i - \hat{y}_i\|}{y_i}$ | Mean absolute percentage error |
| **R²** | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Coefficient of determination |

### 6.2 Uncertainty Quantification Metrics

#### **PICP (Prediction Interval Coverage Probability)**
$$\text{PICP} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}(y_i \in [L_i, U_i])$$

**Target:** PICP ≈ 0.95 (95% confidence level)

**Interpretation:**
- PICP > 0.95: Intervals too wide (over-conservative)
- PICP < 0.95: Intervals too narrow (under-confident)
- PICP ≈ 0.95: Well-calibrated predictions

#### **MPIW (Mean Prediction Interval Width)**
$$\text{MPIW} = \frac{1}{N}\sum_{i=1}^{N} (U_i - L_i)$$

**Interpretation:**
- Measures interval sharpness (narrow = better, if coverage maintained)
- Trade-off with PICP: wider → higher coverage, narrower → lower coverage
- Useful as secondary objective: maximize coverage, minimize width

#### **Winkler Score (Asymmetric Interval Loss)**
$$\text{Winkler} = \frac{1}{N}\sum_{i=1}^{N} \left[(U_i - L_i) + \frac{2}{\alpha}(L_i - y_i)\mathbf{1}(y_i < L_i) + \frac{2}{\alpha}(y_i - U_i)\mathbf{1}(y_i > U_i)\right]$$

**Interpretation:**
- Combines width (sharpness) and coverage (calibration)
- Penalizes misses more than width
- Lower is better
- Penalty proportional to 1/α (stricter for higher confidence)

### 6.3 Decomposition Metrics (MCD & HLLLA Only)

Both MCD and HLLLA separately quantify epistemic and aleatoric uncertainty:

| **Metric** | **Definition** | **Interpretation** |
|---|---|---|
| **Epistemic** | Model/parameter uncertainty | Reduces with more training data or longer training |
| **Aleatoric** | Data/measurement uncertainty | Inherent noise, difficult to reduce |
| **Total** | $\sqrt{\sigma_{\text{epi}}^2 + \sigma_{\text{ale}}^2}$ | Combined uncertainty |

**Analysis:**
- High epistemic, low aleatoric → more data helps
- Low epistemic, high aleatoric → accept data noise or improve sensors
- Balanced → both matter

### 6.4 Temporal Calibration (Rolling Metrics)

To assess if uncertainty estimates adapt over time:

$$\text{Rolling PICP}_{\text{window}}(t) = \frac{1}{W}\sum_{i=t-W/2}^{t+W/2} \mathbf{1}(y_i \in [L_i, U_i])$$

$$\text{Rolling MPIW}_{\text{window}}(t) = \frac{1}{W}\sum_{i=t-W/2}^{t+W/2} (U_i - L_i)$$

**Interpretation:**
- Stable rolling PICP near 0.95 → well-calibrated over time
- Increasing rolling PICP → model becoming more uncertain
- Decreasing rolling PICP → model gaining confidence

---

## 7. Comparative Analysis & Selection Guide

### 7.1 When to Use Each Method

| **Scenario** | **Recommended Method** | **Reason** |
|---|---|---|
| Fast inference required | **HLLLA** | 1 forward pass, no ensemble |
| Critical application (medical, finance) | **CQR** | Formal coverage guarantee |
| Interpretable uncertainty sources | **HLLLA** | Explicit epistemic/aleatoric separation |
| Budget-constrained training | **MCD** | No ensemble training overhead |
| Limited validation data | **MCD or HLLLA** | CQR needs separate validation set for q̂ |
| Unknown data distribution | **CQR** | Distribution-free, works for any data |
| Confidence in Gaussian assumption | **MCD or HLLLA** | Both assume Gaussian (implicit or explicit) |
| Maximal interpretability | **CQR** | Quantiles directly interpretable |

### 7.2 Computational Comparison

| **Aspect** | **MCD** | **HLLLA** | **CQR** |
|---|---|---|---|
| **Training Time** | Low | Low | High (5 ensemble members) |
| **Inference Time per Sample** | Slowest (100 passes) | Fastest (1 pass) | Medium (5 passes) |
| **Memory (Train)** | Low | Low | High (5 models) |
| **Memory (Inference)** | Low | Low | Medium (5 models) |
| **GPU-Friendly** | Yes | Yes | Yes (parallel members) |

---

## 8. Implementation Considerations

### 8.1 Hyperparameter Optimization

All three methods were optimized independently via **Optuna** (50 trials):

```
Search Space:
├─ Lookback: {30, 60, 90}
├─ Batch size: {16, 32, 64}
├─ Hidden units: {64, 128, 224, 256}
├─ Dropout rate: [0.01, 0.30]
├─ Learning rate: [0.0001, 0.01] (log scale)
├─ Epochs: {50, 60, 100}
└─ Patience (early stopping): {7, 9, 15}

Objective: Minimize validation loss (MSE for MCD, NLL for HLLLA, Pinball for CQR)
```

### 8.2 Data Leakage Prevention

Strict temporal ordering maintained:

```
Training set:     [t₁, t₂, ..., t_n₁]  
                       ↓ (StandardScaler fitted here)
Validation set:   [t_n₁+1, ..., t_n₂]  
                       ↓ (CQR conformal calibration here)
Test set:         [t_n₂+1, ..., t_n₃]  
                       ↓ (Evaluation here)

No information from validation/test used during training
CQR conformal threshold q̂ computed ONLY on validation, applied to test
```

### 8.3 Ensemble Member Independence (CQR)

```
Member 1: seed = 42 + 137*0 = 42
Member 2: seed = 42 + 137*1 = 179
Member 3: seed = 42 + 137*2 = 316
Member 4: seed = 42 + 137*3 = 453
Member 5: seed = 42 + 137*4 = 590

Each seed controls:
├─ NumPy random state
├─ TensorFlow random seed
├─ Random weight initialization
└─ Stochastic gradient descent trajectory

Bootstrap sampling indices:
├─ idx_bs = np.random.default_rng(seed+777).integers(0, n, n)
├─ Sample WITH REPLACEMENT (some duplicates, some omissions)
└─ Creates diverse training subsets → diverse learned models
```

---

## 9. Results Reporting Template

### 9.1 Point Forecast Performance

For each method on test set, report:

```
| Method | MAE    | RMSE   | MAPE   | R²     |
|--------|--------|--------|--------|--------|
| MCD    | 45.23  | 56.78  | 0.089  | 0.742  |
| HLLLA  | 43.12  | 54.32  | 0.085  | 0.756  |
| CQR    | 44.67  | 55.89  | 0.087  | 0.749  |
```

### 9.2 Uncertainty Performance

For each method on test set (α=0.05), report:

```
| Method | PICP   | MPIW   | Winkler | Epi Var | Ale Var |
|--------|--------|--------|---------|---------|---------|
| MCD    | 0.953  | 142.34 | 158.67  | 287.1   | 652.3   |
| HLLLA  | 0.947  | 128.45 | 142.89  | 156.2   | 489.7   |
| CQR    | 0.951  | 151.23 | 168.45  | 312.4   | 724.6   |
```

### 9.3 Temporal Calibration

Visualization: Plot rolling PICP over time (window=30)

```
Expected: Line hovering around y=0.95
Interpretation:
  - Stable line → consistent calibration
  - Upward trend → increasing uncertainty over time
  - Downward trend → decreasing uncertainty (gaining confidence)
```

---

## 10. Conclusion

This section implemented three complementary uncertainty quantification methodologies for RNN-based time-series forecasting:

1. **MCD:** Bayesian approximation via stochastic dropout, interpretable epistemic/aleatoric decomposition, empirical coverage
2. **HLLLA:** Fast inference, parametric approach, learned aleatoric uncertainty, empirical coverage
3. **CQR:** Distribution-free guarantees, quantile learning, ensemble-based, formal coverage guarantee

Each method addresses different requirements:
- **MCD:** Best for understanding Bayesian approximation and computational budgets
- **HLLLA:** Best for fast inference and explicit uncertainty decomposition
- **CQR:** Best for critical applications requiring formal, provable coverage guarantees

The choice among methods depends on application requirements (speed, formality, interpretability), computational constraints, and data characteristics. This work provides comprehensive implementation and evaluation frameworks for all three approaches, enabling practitioners to select and deploy appropriate UQ methods for their specific time-series forecasting scenarios.
