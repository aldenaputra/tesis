# Methodology: Uncertainty Quantification in Temporal Convolutional Networks

## Overview

This section details the implementation of three distinct uncertainty quantification (UQ) methodologies within Temporal Convolutional Network (TCN) architectures for time-series forecasting. Each method captures different aspects of uncertainty through distinct mechanisms and provides complementary insights into prediction reliability.

---

## 1. Monte Carlo Dropout (MCD)

### Conceptual Foundation

Monte Carlo Dropout is a Bayesian approximation technique that treats dropout as a form of stochastic sampling from an approximate posterior distribution. During training, dropout layers randomly mask features with probability $p$, acting as a regularization mechanism. Critically, in standard inference, dropout is disabled (training=False). MCD inverts this by keeping dropout **active during inference**, forcing the model to make multiple stochastic forward passes through identical architecture but different weight subsets.

### Implementation in TCN

In the TCN architecture, `SpatialDropout1D` layers are embedded within each TCN residual block after activation functions. These layers apply dropout along the temporal (feature) dimension, creating spatially coherent masking. During training, the model learns with these stochastic layers active, effectively training an ensemble of thinned networks. At inference time, instead of a single deterministic forward pass, the trained TCN is executed N times (typically 100) with `training=True`, forcing dropout to remain active.

Each forward pass produces a stochastically different prediction due to different weight dropout masks, generating an ensemble of predictions $\{\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_{N_{\text{MC}}}\}$. The point forecast is obtained as the mean of this ensemble:
$$\hat{y}_{\text{MC}} = \frac{1}{N_{\text{MC}}} \sum_{i=1}^{N_{\text{MC}}} \hat{y}_i$$

Uncertainty is quantified through the standard deviation across ensemble members:
$$\sigma_{\text{MC}} = \sqrt{\frac{1}{N_{\text{MC}}} \sum_{i=1}^{N_{\text{MC}}} (\hat{y}_i - \hat{y}_{\text{MC}})^2}$$

Prediction intervals are constructed using empirical quantiles (if `USE_QUANTILES=True`):
$$L = Q_{0.025}(\{\hat{y}_1, \ldots, \hat{y}_{N_{\text{MC}}}\}), \quad U = Q_{0.975}(\{\hat{y}_1, \ldots, \hat{y}_{N_{\text{MC}}}\})$$

Or using Gaussian approximation:
$$L = \hat{y}_{\text{MC}} - z \cdot \sigma_{\text{MC}}, \quad U = \hat{y}_{\text{MC}} + z \cdot \sigma_{\text{MC}}$$
where $z$ is the critical value (1.96 for 95% intervals).

### Uncertainty Decomposition

Epistemic (model) and aleatoric (data) uncertainties are approximated by partitioning the total variance:
$$\sigma_{\text{epi}}^2 = \text{Var}_{\text{MC}}(\hat{y}) - \sigma_{\text{alea}}^2$$

where $\sigma_{\text{alea}}^2$ is estimated as the variance of residuals on the validation set. This decomposition provides insights into whether prediction uncertainty stems from model ignorance (high epistemic) or inherent data noise (high aleatoric).

---

## 2. Heteroscedastic NLL with Last-Layer Laplace Approximation (HLLLA)

### Conceptual Foundation

HLLLA combines heteroscedastic predictive modeling (where the model learns to predict both mean and variance) with Last-Layer Laplace Approximation (LLLA), a Bayesian method for uncertainty quantification. The model is explicitly trained to output two parameters: the conditional mean $\mu(x)$ and the conditional log-variance $\log \sigma^2(x)$. This heteroscedastic approach captures aleatoric uncertainty directly through the learned variance. Epistemic uncertainty is then estimated via LLLA, which approximates the posterior distribution over only the final layer weights.

### Implementation in TCN

The TCN architecture is modified to have a **heteroscedastic output head** consisting of `Dense(2)` instead of `Dense(1)`. The first output channel predicts the mean in standardized space $\mu_s$, while the second channel predicts the log-variance $\log \sigma_s^2$. The loss function is the Gaussian Negative Log-Likelihood (NLL):
$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \left[ \frac{1}{2} \log \sigma_i^2 + \frac{1}{2\sigma_i^2}(y_i - \mu_i)^2 \right]$$

where log-variance is clipped for numerical stability: $\log \sigma^2 \in [-20, 5]$.

After training, the model weights are fixed, and the Last-Layer Laplace Approximation is applied. First, the penultimate layer (before the final Dense(2)) is identified, and a feature extractor model is created: $\phi(x) = \text{output of penultimate layer}$. Noise variance is estimated from training residuals:
$$\sigma_n^2 = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} (y_i - \mu_i)^2$$

The diagonal Hessian (Fisher Information Matrix) at the trained weights is approximated as:
$$H_{\text{diag}} = \frac{1}{\sigma_n^2} \sum_{i=1}^{N_{\text{train}}} \Phi_i^T \Phi_i + \lambda I$$

where $\Phi_i = [\phi(x_i), 1]$ (features with bias term) and $\lambda$ is a prior precision hyperparameter. The posterior covariance over last-layer weights is approximated as:
$$\text{Cov}(w) \approx H_{\text{diag}}^{-1}$$

Epistemic uncertainty is then computed as:
$$\sigma_{\text{epi}}(x) = \sqrt{\Phi(x)^T H_{\text{diag}}^{-1} \Phi(x)} \approx \sqrt{\sum_j \Phi_j(x)^2 \cdot [H_{\text{diag}}^{-1}]_{jj}}$$

### Uncertainty Decomposition

Aleatoric uncertainty is directly available from the model's heteroscedastic output:
$$\sigma_{\text{ale}}(x) = \sqrt{\exp(\log \sigma_s^2(x)) \cdot y_{\text{scale}}^2}$$

Total predictive uncertainty combines both components:
$$\sigma_{\text{total}}(x) = \sqrt{\sigma_{\text{ale}}(x)^2 + \sigma_{\text{epi}}(x)^2}$$

Prediction intervals are constructed using this combined uncertainty with Gaussian approximation:
$$L = \mu(x) - z \cdot \sigma_{\text{total}}(x), \quad U = \mu(x) + z \cdot \sigma_{\text{total}}(x)$$

The explicit separation of aleatoric and epistemic components provides interpretability regarding the source of uncertainty, making HLLLA particularly valuable when understanding prediction confidence is important.

---

## 3. Conformal Quantile Regression (CQR)

### Conceptual Foundation

Conformal Quantile Regression combines quantile regression (which directly predicts multiple quantiles of the conditional distribution) with conformal prediction (a distribution-free method providing finite-sample coverage guarantees). Rather than predicting a point estimate or learning to predict variance, the model is trained to output the quantiles themselves. Quantile regression uses the asymmetric pinball loss, which naturally encourages the lower quantile to lie below the true value and the upper quantile to lie above it. Conformal prediction then uses a held-out validation set to compute a data-dependent calibration threshold, guaranteeing that the final prediction intervals achieve the desired coverage level.

### Implementation in TCN

The TCN architecture is modified with a **quantile regression head** outputting `Dense(len(TAUS))` channels, where TAUS typically consists of three quantile levels: $\tau \in [0.025, 0.5, 0.975]$ corresponding to lower, median, and upper quantiles.

The training loss is the multi-quantile pinball loss:
$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \sum_{\tau \in \text{TAUS}} \rho_\tau(y_i - \hat{q}_\tau(x_i))$$

where the pinball loss for quantile $\tau$ is:
$$\rho_\tau(u) = \max(\tau \cdot u, (\tau - 1) \cdot u)$$

This loss is asymmetric: for $\tau = 0.025$, overshooting (prediction too high) incurs larger penalties than undershooting, encouraging the model to predict low. Conversely, for $\tau = 0.975$, the opposite holds.

To improve uncertainty quantification, an ensemble of $M$ independent TCN models is trained, each using bootstrap resampling of the training data. For each member $m \in \{1, \ldots, M\}$:
1. Sample indices with replacement: $\text{idx}_m \sim \text{Uniform}(\{1, \ldots, N\})^N$
2. Train on bootstrapped sample: $(X_m^{\text{boot}}, y_m^{\text{boot}})$
3. Store trained model member $m$

Each member predicts quantiles independently. Ensemble aggregation produces averaged quantile estimates:
$$\bar{q}_\tau = \frac{1}{M} \sum_{m=1}^{M} \hat{q}_\tau^{(m)}(x)$$

**Conformal Calibration** is performed using the validation set. Non-conformity scores quantify how far test points fall outside the predicted interval:
$$R_i = \max(0, \bar{q}_{0.025,i} - y_i, y_i - \bar{q}_{0.975,i})$$

The conformal threshold is computed as the empirical quantile of validation non-conformity scores:
$$\hat{q} = \left\lceil \frac{(n_{\text{val}}+1)(1-\alpha)}{n_{\text{val}}} \right\rceil \text{-th smallest } R_i$$

where $\alpha = 0.05$ for 95% coverage. This threshold is then applied symmetrically to all intervals:
$$L = \bar{q}_{0.025} - \hat{q}, \quad U = \bar{q}_{0.975} + \hat{q}$$

The point forecast is the ensemble median: $\hat{y}_{\text{CQR}} = \bar{q}_{0.5}$.

### Uncertainty Decomposition

Epistemic (model) uncertainty is estimated as the across-ensemble variance of the median quantile:
$$\sigma_{\text{epi}}^2 = \text{Var}_m(\hat{q}_{0.5}^{(m)}) \quad \text{if } M > 1$$

Aleatoric (data) uncertainty is estimated from the inter-quantile range under Gaussian approximation:
$$\text{IQR} = \bar{q}_{0.975} - \bar{q}_{0.025}$$
$$\sigma_{\text{ale}} = \frac{\text{IQR}}{3.92}$$

where the denominator 3.92 corresponds to the 95% interval width of a standard normal distribution.

### Coverage Guarantee

A key advantage of CQR is the **finite-sample coverage guarantee**. Due to the conformal framework, the final prediction intervals satisfy:
$$P(y_{n+1} \in [L, U]) \geq 1 - \alpha \quad \text{(asymptotically)}$$

This guarantee holds **regardless of the data distribution** and relies only on exchangeability, making it valuable for safety-critical applications where coverage probability is paramount.

---

## Comprehensive Comparison Table

| **Aspect** | **MCD** | **HLLLA** | **CQR** |
|:---|:---:|:---:|:---:|
| **Architecture Component** | SpatialDropout1D in TCN blocks | Dense(2) heteroscedastic head | Dense(3) quantile head |
| **Output Type** | Single point prediction | (μ, log σ²) pair | (q₀.₀₂₅, q₀.₅, q₀.₉₇₅) triplet |
| **Loss Function** | MSE (standard) | Gaussian NLL | Pinball loss (asymmetric) |
| **Training Procedure** | Standard TCN training | Standard with heteroscedastic loss | Quantile regression training |
| **Ensemble Strategy** | Stochastic (dropout masks) | Single deterministic model | M independent bootstrap-trained models |
| **Number of Forward Passes** | Nₘc ≈ 100 (at inference) | 1 (training); ~1 (inference) | M ≈ 5 (M members, 1 pass each) |
| **Point Forecast** | Mean of MC samples | μ from heteroscedastic head | Ensemble median |
| **Aleatoric Estimation** | Implicit (MC variance proxy) | Explicit (learned σ²) | From IQR (interval width) |
| **Epistemic Estimation** | Variance across MC samples | LLLA posterior over last layer | Variance across ensemble members |
| **Total Uncertainty** | σ_mc directly | √(σ_ale² + σ_epi²) | √(σ_ale² + σ_epi²) |
| **Interval Construction** | Quantiles or Gaussian ±z·σ | Gaussian ±z·σ_total | Quantiles ± q̂ (conformal) |
| **Calibration Method** | Implicit (dropout %) | Implicit (NLL training) | Explicit (validation split, conformal) |
| **Coverage Guarantee** | Approximate | Approximate | Formal (asymptotic, distribution-free) |
| **Computational Cost (Training)** | Standard | Standard | M × standard |
| **Computational Cost (Inference)** | Very High (100 passes) | Very Low (1 pass) | Low-Medium (5 passes) |
| **Memory Requirement** | High (store 100 predictions) | Low | Medium (store M models) |
| **Hyperparameters** | Dropout rate, Nₘc, α, z | λ_prior, log_var clip, α, z | M, Bootstrap flag, τ values, α |
| **Data Requirements** | Training + validation | Training + validation | Training + validation + calibration |
| **Robustness to Overfit** | Good (stochasticity acts as regularizer) | Moderate | Good (bootstrap + ensemble) |
| **Interpretability** | Black-box ensemble | Explicit uncertainty components | Explicit quantiles + conformal logic |
| **Theoretical Justification** | Bayesian approximation | Laplace approximation | Distribution-free conformal theory |
| **Best Use Case** | General uncertainty, simple implementation | When uncertainty decomposition needed | When coverage guarantee is critical |
| **Code Complexity** | Low (wrapper on trained model) | Medium (LLLA computation post-training) | High (ensemble training + conformal) |
| **Scalability** | Poor (Nₘc scales linearly with inference cost) | Excellent | Good |
| **Temperature Tuning** | No | No | No (distribution-free) |

---

## Implementation Details Summary

### Training Phase

**MCD & HLLLA:**
- Both use standard training procedure (MSE for MCD, Heteroscedastic NLL for HLLLA)
- Single model trained on training+validation data (or early stopping on validation)
- Training time: ~identical

**CQR:**
- Quantile regression training with pinball loss
- Requires THREE separate phases:
  1. Train M ensemble members (each on bootstrap sample)
  2. Predict on validation set for non-conformity scores
  3. Compute conformal threshold from validation residuals
- Training time: ~M× longer than single model

### Inference Phase

**MCD:**
```
for i in 1 to Nₘc:
    ŷᵢ ← model(x, training=True)  # Dropout active
predictions ← [ŷ₁, ..., ŷₙₘc]
point_pred ← mean(predictions)
intervals ← quantiles(predictions, [0.025, 0.975])
```

**HLLLA:**
```
μ_scaled, log_σ² ← model(x)
μ, σ_ale ← inverse_scale(μ_scaled, σ²_scaled)
σ_epi ← LLLA_posterior(x)
σ_total ← √(σ_ale² + σ_epi²)
point_pred ← μ
intervals ← [μ - z·σ_total, μ + z·σ_total]
```

**CQR:**
```
for member in members:
    q_025, q_50, q_975 ← member.predict(x)
    store quantiles
q̄_025, q̄_50, q̄_975 ← ensemble_mean(quantiles)
point_pred ← q̄_50
intervals ← [q̄_025 - q̂, q̄_975 + q̂]  # q̂ from conformal calibration
```

---

## Key Methodological Considerations

### Exchangeability Assumption
All three methods assume exchangeability of samples, though this is most critical for CQR's formal coverage guarantee. For time-series data, this assumption is relaxed through temporal windowing (making successive samples more independent).

### Computational Trade-offs
- **MCD**: Maximum uncertainty, maximum computation (100× inference cost)
- **HLLLA**: Minimum uncertainty, minimum computation (standard inference)
- **CQR**: Moderate computation (5× inference cost) with formal guarantees

### Coverage Calibration
- **MCD & HLLLA**: Coverage depends on model capacity and training data quality
- **CQR**: Coverage **guaranteed** by construction (distribution-free, finite-sample validity)

### Ensemble Diversity
- **MCD**: Diversity from stochastic dropout patterns (100 variations)
- **HLLLA**: No ensemble (single model) but LLLA captures weight uncertainty
- **CQR**: Diversity from bootstrap resampling and different random seeds (5 variations)

---

## Model Selection Criteria

| **Criterion** | **MCD** | **HLLLA** | **CQR** |
|:---|:---:|:---:|:---:|
| Need fast inference? | ✗ | ✓ | ✓ |
| Need interpretable uncertainty decomposition? | ✗ | ✓ | ✓ |
| Need formal coverage guarantee? | ✗ | ✗ | ✓ |
| Prefer single-model simplicity? | ✓ | ✓ | ✗ |
| Have limited computational resources? | ✗ | ✓ | ✓ |
| Need to handle small validation sets? | ✓ | ✓ | ✗ |
| Want theoretically-grounded approach? | ~ | ~ | ✓ |
| Prefer empirically well-studied method? | ✓ | ~ | ~ |

---

## Conclusion

Each UQ method offers distinct advantages within the TCN framework:

- **MCD** provides a practical, well-understood approach to uncertainty through stochastic dropout ensembling, suitable for applications where computational cost is secondary to comprehensive uncertainty quantification.

- **HLLLA** offers computational efficiency combined with explicit uncertainty decomposition, making it ideal when inference speed matters and interpretable epistemic/aleatoric separation is valuable.

- **CQR** delivers distribution-free coverage guarantees with moderate computational cost, making it the method of choice for applications where maintaining specific coverage levels is critical.

The choice among these methods should be guided by the specific requirements of the forecasting task, computational constraints, and the importance of different aspects of uncertainty quantification.
