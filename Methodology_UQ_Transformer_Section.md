# Uncertainty Quantification in Temporal Transformer Networks: Methodology

## 1. Overview

This section describes the implementation of three distinct uncertainty quantification (UQ) approaches integrated into Temporal Transformer Networks for probabilistic time-series forecasting. The three methods—Monte Carlo Dropout (MCD), Heteroscedastic Loss with Last-Layer Laplace Approximation (HLLLA), and Conformal Quantile Regression (CQR)—represent different philosophical approaches to capturing predictive uncertainty. MCD approximates Bayesian inference through stochastic dropout during inference, HLLLA decomposes uncertainty into aleatoric and epistemic components through heteroscedastic loss and Laplace approximation, and CQR provides formal distribution-free coverage guarantees through ensemble quantile regression and conformal calibration. Each method was implemented within an identical Temporal Transformer encoder architecture with multi-head attention mechanisms, positional encoding, and residual connections, ensuring fair comparison across UQ methodologies while controlling for architectural factors.

---

## 2. Transformer Encoder Architecture

All three UQ methods utilize the same base Transformer encoder architecture, consisting of:
- **Input projection**: Dense layer projecting raw input features to embedding dimension $d_{model} \in \{32, 64, 96, 128\}$
- **Positional encoding**: Sinusoidal positional encoding $PE_{pos,2i} = \sin(pos/10000^{2i/d_{model}})$ providing temporal position awareness
- **Encoder blocks**: $L$ stacked encoder blocks where $L \in \{1, 2, 3\}$, each containing:
  - Multi-head self-attention: $L = \text{MultiHeadAttention}(X, X, causal\_mask=True)$ with $h$ attention heads
  - Dropout regularization: Dropout$(p_{drop})$ applied after attention and within feed-forward networks, where $p_{drop} \in [0, 0.3]$
  - Feed-forward network: Two-layer dense network with hidden dimension $d_{ff} = \{2d_{model}, 3d_{model}, 4d_{model}\}$
  - Layer normalization: Applied after residual additions for numerical stability
- **Sequence pooling**: Lambda layer extracting the last timestep representation for single-step prediction
- **Output head**: Architecture-specific prediction layer detailed below for each UQ method

Hyperparameter optimization was performed via random search with $N_{trials} = 50$ trials, sampling: lookback window $L_{bw} \in \{30, 45, 60, 90\}$, $d_{model}$, number of heads $h$ (constrained such that $d_{model} \mod h = 0$), dropout rate, learning rate, batch size, and epochs.

---

## 3. Method 1: Monte Carlo Dropout (MCD)

### 3.1 Architecture and Training

MCD implements uncertainty quantification through stochastic inference via dropout, treating a neural network with dropout as a Bayesian approximation. The output head consists of a single Dense layer producing point predictions: $\hat{y}_s = f(x; w, \theta)_{scaled}$, where subscript $s$ denotes scaled-space predictions. Training employs standard mean squared error loss:

$$\mathcal{L}_{MSE}(y, \hat{y}) = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$

Dropout layers ($p_{drop} \in [0, 0.3]$) are embedded within encoder blocks (three dropout points per encoder block: after attention, within feed-forward) and serve as implicit regularization. During training, dropout is activated by default, creating an ensemble of sub-networks. Importantly, **training does not explicitly learn uncertainty**; the model learns point predictions, and uncertainty emerges from ensemble variance.

### 3.2 Inference and Uncertainty Quantification

At inference time, MCD forces dropout to remain active by specifying $training=True$ in forward passes, enabling stochastic inference. The procedure involves:

1. **Ensemble generation**: Execute $N_{MC} = 100$ forward passes through the trained model with different dropout masks, collecting predictions:
   $$\{\hat{y}_s^{(1)}, \hat{y}_s^{(2)}, \ldots, \hat{y}_s^{(N_{MC})}\}$$

2. **Inverse scaling**: Transform predictions from standardized space back to original scale:
   $$\hat{y}^{(j)} = \hat{y}_s^{(j)} \cdot \sigma_y + \mu_y$$

3. **Ensemble statistics**: Compute distributional properties from the $N_{MC}$ samples:
   - Point estimate: $\hat{\mu} = \frac{1}{N_{MC}}\sum_{j=1}^{N_{MC}} \hat{y}^{(j)}$ (ensemble mean)
   - Uncertainty: $\hat{\sigma} = \sqrt{\frac{1}{N_{MC}-1}\sum_{j=1}^{N_{MC}}(\hat{y}^{(j)} - \hat{\mu})^2}$ (ensemble standard deviation)
   - Quantiles: $\hat{Q}_{\tau} = \text{quantile}(\{\hat{y}^{(j)}\}, \tau)$ where $\tau \in \{0.025, 0.975\}$

4. **Uncertainty decomposition**: Under the assumption that ensemble variance comprises epistemic and aleatoric components:
   $$\sigma_{total}^2 = \sigma_{epistemic}^2 + \sigma_{aleatoric}^2$$
   
   Aleatoric uncertainty is approximated from validation residuals:
   $$\sigma_{aleatoric}^2 = \frac{1}{N_{val}}\sum_{i=1}^{N_{val}}(y_i - \hat{\mu}_i)^2$$
   
   Epistemic uncertainty is derived as the residual:
   $$\sigma_{epistemic}^2 = \text{max}(\sigma_{total}^2 - \sigma_{aleatoric}^2, 0)$$

5. **Prediction intervals**: Construct intervals using empirical quantiles or Gaussian assumption:
   $$[L, U] = [\hat{Q}_{0.025}, \hat{Q}_{0.975}] \quad \text{or} \quad [L, U] = [\hat{\mu} - 1.96\hat{\sigma}, \hat{\mu} + 1.96\hat{\sigma}]$$

### 3.3 Theoretical Foundation

MCD approximates Bayesian inference by interpreting dropout as approximate variational inference. Under this interpretation, each stochastic forward pass samples from an approximate posterior over network weights, and the ensemble of predictions approximates the Bayesian predictive distribution. This connection was formalized by Gal & Ghahramani (2016), establishing that performing MC dropout at test time approximates sampling from a dropout-based variational posterior.

---

## 4. Method 2: Heteroscedastic Loss with Last-Layer Laplace Approximation (HLLLA)

### 4.1 Architecture and Training

HLLLA employs heteroscedastic learning combined with post-training epistemic uncertainty estimation via Laplace approximation. The output head deviates from MCD by predicting **two values** per sample:

$$\hat{y}_{out} = [\hat{\mu}_s, \log\hat{\sigma}^2_s]$$

where $\hat{\mu}_s$ is the predicted mean in scaled space and $\log\hat{\sigma}^2_s$ is the log-variance (logged for numerical stability). Training uses Gaussian negative log-likelihood loss:

$$\mathcal{L}_{NLL}(y, \hat{\mu}, \hat{\sigma}^2) = \frac{1}{N}\sum_{i=1}^{N}\left[\frac{1}{2}\log(\hat{\sigma}_i^2) + \frac{(y_i - \hat{\mu}_i)^2}{2\hat{\sigma}_i^2}\right]$$

This loss function directly optimizes the Gaussian likelihood $p(y|x) = \mathcal{N}(y; \mu(x), \sigma^2(x))$, encouraging the model to:
- Minimize prediction error $(y - \mu)^2$ (accuracy term)
- Learn appropriate prediction variance $\sigma^2$ (uncertainty term)

The log-variance is clipped to $[-20, 5]$ during training for numerical stability, preventing extreme values. This learned variance represents **aleatoric uncertainty** (data-dependent noise that cannot be reduced through additional observations).

### 4.2 Last-Layer Laplace Approximation (LLLA)

Epistemicuncertainty is estimated post-training through Laplace approximation restricted to the final layer. The procedure involves:

1. **Feature extraction**: Extract features from the penultimate layer (the Lambda layer output):
   $$\phi(x) = \text{Penultimate}(x) \in \mathbb{R}^{d_{model}}$$

2. **Noise estimation**: Estimate the noise variance $\sigma_n^2$ from training residuals:
   $$\sigma_n^2 = \frac{1}{N_{train}}\sum_{i=1}^{N_{train}}(y_i - \hat{\mu}_i)^2$$

3. **Hessian diagonal approximation**: Compute the diagonal of the Hessian matrix at the MAP estimate. For heteroscedastic regression, the Fisher Information Matrix diagonal approximates the Hessian:
   $$H_{diag} = \frac{1}{\sigma_n^2}\sum_{i=1}^{N}\phi(x_i) \odot \phi(x_i) + \lambda I$$
   
   where $\odot$ denotes element-wise multiplication and $\lambda$ is an L2 regularization weight. The posterior variance (precision) is:
   $$\text{Var}(w) \approx H_{diag}^{-1}$$

4. **Epistemic variance computation**: At each test point, epistemic uncertainty arises from weight uncertainty:
   $$\sigma_{epistemic}^2(x) = \phi(x)^T \text{diag}(\text{Var}(w)) \phi(x) \approx \sum_{j} \phi_j(x)^2 \text{Var}(w_j)$$

5. **Total uncertainty combination**: The total predictive uncertainty combines both sources:
   $$\sigma_{total}^2 = \sigma_{aleatoric}^2 + \sigma_{epistemic}^2$$

This decomposition provides interpretable uncertainty: aleatoric uncertainty (what the model has learned from data) and epistemic uncertainty (what the model does not know due to insufficient training coverage).

### 4.3 Prediction Intervals

Prediction intervals are constructed assuming Gaussian predictive distribution:
$$[L, U] = [\hat{\mu} - z_{\alpha/2}\sigma_{total}, \hat{\mu} + z_{\alpha/2}\sigma_{total}]$$

where $z_{\alpha/2} = 1.96$ for 95% confidence level (corresponding to $\alpha = 0.05$).

---

## 5. Method 3: Conformal Quantile Regression (CQR)

### 5.1 Ensemble Quantile Regression Training

CQR employs an ensemble of $M = 5$ independent Transformer models, each trained to predict three quantiles: lower quantile $Q_{0.025}$, median $Q_{0.5}$, and upper quantile $Q_{0.975}$. The output head produces three predictions:

$$\hat{y}_{out} = [\hat{Q}_{0.025}, \hat{Q}_{0.5}, \hat{Q}_{0.975}]$$

Each ensemble member is trained using multi-quantile pinball loss:

$$\mathcal{L}_{pinball}(y, \hat{Q}_{\tau}) = \frac{1}{N}\sum_{i=1}^{N}\sum_{\tau \in \{0.025, 0.5, 0.975\}} \rho_\tau(y_i - \hat{Q}_{\tau,i})$$

where the pinball loss is defined as:

$$\rho_\tau(e) = \max(\tau \cdot e, (\tau - 1) \cdot e)$$

This asymmetric loss penalizes under- and over-prediction asymmetrically:
- For $\tau = 0.025$ (lower quantile): Underprediction (e < 0) is heavily penalized, encouraging conservative lower bounds
- For $\tau = 0.5$ (median): Symmetric loss, standard median regression
- For $\tau = 0.975$ (upper quantile): Overprediction (e > 0) is heavily penalized, encouraging conservative upper bounds

### 5.2 Ensemble Diversity via Bootstrap

To create diverse ensemble members, each of the $M = 5$ models is trained on a bootstrap sample of the training data:

$$\text{Bootstrap index set}: \mathcal{I}_m = \{\text{sample with replacement from } \{1, \ldots, N_{train}\}\}$$

Combined with different random seeds for weight initialization, this creates:
1. Different training data distributions per member (bootstrap variation)
2. Different initial weights (seed variation)
3. Different stochastic optimization trajectories (training variation)

Result: An ensemble of diverse quantile regression models, each capturing different aspects of the conditional quantile function $Q_\tau(x)$.

### 5.3 Ensemble Aggregation and Uncertainty Estimation

For a given input $x$ and all three splits (train/validation/test), the procedure is:

1. **Member-level predictions**: For each of $M$ ensemble members, obtain the three quantile predictions and inverse-scale to original space:
   $$\hat{Q}_{\tau}^{(m)}(x) = \text{quantile output}_{\tau}^{(m)} \cdot \sigma_y + \mu_y$$

2. **Enforcing monotonicity**: Sort each sample's quantile predictions to ensure $\hat{Q}_{0.025} \leq \hat{Q}_{0.5} \leq \hat{Q}_{0.975}$

3. **Ensemble averaging**: Compute the mean quantile across the ensemble:
   $$\bar{Q}_\tau = \frac{1}{M}\sum_{m=1}^{M}\hat{Q}_{\tau}^{(m)}(x)$$

4. **Epistemic uncertainty**: Measure disagreement among ensemble members on the median prediction:
   $$\sigma_{epistemic}^2 = \text{Var}(\{\hat{Q}_{0.5}^{(1)}, \ldots, \hat{Q}_{0.5}^{(M)}\})$$

5. **Aleatoric uncertainty proxy**: Approximate from the interquartile range (IQR) assuming Gaussian:
   $$\text{IQR} = \bar{Q}_{0.975} - \bar{Q}_{0.025}$$
   $$\sigma_{aleatoric} \approx \frac{\text{IQR}}{3.92}$$
   
   (since for Gaussian, $\text{IQR} \approx 1.96\sigma + 1.96\sigma = 3.92\sigma$)

### 5.4 Split Conformal Calibration

CQR employs split conformal prediction to ensure formal coverage guarantees. On the **validation set**, compute non-conformity scores:

$$E_i = \max(L_i - y_i, y_i - U_i, 0)$$

where $L_i = \bar{Q}_{0.025}(x_i)$ and $U_i = \bar{Q}_{0.975}(x_i)$. This measures how far actual $y_i$ lies outside the ensemble's predicted interval. Collect all scores: $\{E_1, E_2, \ldots, E_{n_{val}}\}$.

Compute the conformal threshold:

$$\hat{q} = \text{quantile}(\{E_i\}, 1 - \alpha, \text{method}=\text{higher})$$

This threshold ensures that the fraction of non-conformity scores exceeding $\hat{q}$ is at most $\alpha$ (approximately). Mathematically, with $n_{val}$ validation samples:

$$P(\hat{E}_{n+1} \leq \hat{q}) \geq 1 - \alpha$$

### 5.5 Conformalization and Formal Coverage Guarantee

Apply the threshold to all splits (train/validation/test) by expanding the intervals:

$$L_{conform} = L - \hat{q}, \quad U_{conform} = U + \hat{q}$$

This conformalized interval $[L_{conform}, U_{conform}]$ provides a **distribution-free coverage guarantee**:

$$P(y_{test} \in [L_{conform}, U_{conform}]) \geq 1 - \alpha - \frac{\lceil(n_{val}+1)(1-\alpha)\rceil}{n_{val}+1}$$

The second term is negligible for large $n_{val}$. Importantly, this guarantee:
- Requires **no distributional assumptions** (not even Gaussianity)
- Requires only **exchangeability** of data
- Is **finite-sample** (not asymptotic)
- Holds for **any** model class (quantile regression, neural networks, decision trees, etc.)

For test point predictions, the point forecast is the ensemble median:

$$\hat{\mu} = \bar{Q}_{0.5}$$

---

## 6. Comparative Methodology Summary

| **Dimension** | **MCD** | **HLLLA** | **CQR** |
|---|---|---|---|
| **Output Head** | Dense(1): point | Dense(2): μ, log σ² | Dense(3): 3 quantiles |
| **Base Model** | Single Transformer | Single Transformer | M=5 Transformer ensemble |
| **Training Data** | Full training set | Full training set | M bootstrap samples |
| **Primary Loss** | MSE | Gaussian NLL | Pinball loss |
| **Uncertainty Learning** | Implicit (dropout) | Explicit (heteroscedastic) | Explicit (quantile) |
| **Inference Cost** | N_MC=100 passes | 1 forward + LLLA | M=5 forward passes |
| **Aleatoric Estimation** | Validation residuals | Learned output σ² | IQR-based proxy |
| **Epistemic Estimation** | Ensemble variance | LLLA Hessian diagonal | Member disagreement |
| **Decomposition Method** | Variance subtraction | Principled via LLLA | Proxies (variance+IQR) |
| **Prediction Interval** | [μ̂ ± 1.96σ̂] or quantiles | [μ̂ ± z σ_total] | [L̄ - q̂, Ū + q̂] conformal |
| **Distribution Assumption** | Gaussian (for z-scores) | Gaussian | Distribution-free |
| **Coverage Type** | Empirical (from ensemble) | From Gaussian CDF | Formal conformal guarantee |
| **Coverage Guarantee** | Approximate | Model-dependent | **Provable (≥1-α)** |
| **Hyperparameter Tuning** | Dropout rate p_drop | Dropout rate p_drop | Ensemble size M, α |
| **Calibration Method** | Empirical quantiles | Gaussian scaling | Threshold q̂ |
| **Best for OOD Detection** | Moderate (ensemble spread) | Good (epistemic variance) | Excellent (wider intervals) |
| **Best for Computational Efficiency** | Poor (100× passes) | Good (1 pass) | Moderate (5× passes) |
| **Best for Formal Guarantees** | No | No | **Yes** |
| **Theoretical Basis** | Bayesian NN approximation | Laplace approximation | Distribution-free inference |

---

## 7. Implementation Details and Pseudocode

### 7.1 MCD Inference Procedure

```
Function MC_Dropout_Inference(model, X_test, N_MC=100, α=0.05):
    predictions ← []
    for i = 1 to N_MC:
        ŷ_s^(i) ← model.predict(X_test, training=True)  // Force dropout active
        ŷ^(i) ← inverse_scale(ŷ_s^(i))
        predictions.append(ŷ^(i))
    
    predictions ← stack(predictions, axis=1)  // Shape: (N, N_MC)
    
    μ̂ ← mean(predictions, axis=1)
    σ̂ ← std(predictions, axis=1, ddof=1)
    L ← quantile(predictions, α/2, axis=1)
    U ← quantile(predictions, 1-α/2, axis=1)
    
    return {μ̂, σ̂, L, U}
```

### 7.2 HLLLA Inference Procedure

```
Function HLLLA_Inference(model, φ_model, X_train, y_train, X_test, α=0.05):
    // Training time: compute Hessian diagonal
    ŷ_s_train ← model.predict(X_train)
    μ_s_train ← ŷ_s_train[:, 0]
    
    σ_n² ← mean((y_train - μ_s_train)²)
    
    Φ_train ← φ_model.predict(X_train)  // Penultimate features
    H_diag ← (1/σ_n²) × sum(Φ_train² + λI)
    var_w_diag ← 1 / H_diag
    
    // Test time: predict with uncertainty
    ŷ_s_test ← model.predict(X_test)
    μ_s_test ← ŷ_s_test[:, 0]
    log_var_s_test ← clip(ŷ_s_test[:, 1], -20, 5)
    
    μ_test ← inverse_scale(μ_s_test)
    σ_ale_test ← sqrt(inverse_scale_var(exp(log_var_s_test)))
    
    Φ_test ← φ_model.predict(X_test)
    σ_epi_test² ← sum(Φ_test² × var_w_diag, axis=1)
    σ_epi_test ← sqrt(inverse_scale_var(σ_epi_test²))
    
    σ_total_test ← sqrt(σ_ale_test² + σ_epi_test²)
    
    L ← μ_test - 1.96 × σ_total_test
    U ← μ_test + 1.96 × σ_total_test
    
    return {μ_test, σ_ale_test, σ_epi_test, σ_total_test, L, U}
```

### 7.3 CQR Inference Procedure

```
Function CQR_Inference(ensemble_members, X_train, y_train, X_val, y_val, X_test, α=0.05):
    // Training: already done (pinball loss on each member)
    
    // Validation: conformal calibration
    L_val, M_val, U_val ← empty lists
    for m in ensemble_members:
        Q_m ← m.predict(X_val)  // (N_val, 3) quantile outputs
        Q_m ← inverse_scale(Q_m)
        L_val.append(Q_m[:, 0])
        M_val.append(Q_m[:, 1])
        U_val.append(Q_m[:, 2])
    
    L_val ← mean(L_val, axis=0)
    U_val ← mean(U_val, axis=0)
    
    E_val ← max(L_val - y_val, y_val - U_val, 0)
    q̂ ← quantile(E_val, 1-α, method="higher")
    
    // Test: apply conformal threshold
    L_test, M_test, U_test ← empty lists
    for m in ensemble_members:
        Q_m ← m.predict(X_test)
        Q_m ← inverse_scale(Q_m)
        L_test.append(Q_m[:, 0])
        M_test.append(Q_m[:, 1])
        U_test.append(Q_m[:, 2])
    
    L_test ← mean(L_test, axis=0) - q̂
    U_test ← mean(U_test, axis=0) + q̂
    μ_test ← mean(M_test, axis=0)
    
    var_epi_test ← var(M_test, axis=0, ddof=1)
    IQR ← mean(U_test, axis=0) - mean(L_test, axis=0)
    σ_ale_test ← IQR / 3.92
    
    return {μ_test, L_test, U_test, var_epi_test, var_ale_test}
```

---

## 8. Model Selection Criteria

The selection of which UQ method to employ depends on several criteria:

| **Decision Criterion** | **MCD** | **HLLLA** | **CQR** |
|---|---|---|---|
| **Need formal coverage guarantee** | ✗ | ✗ | ✓ |
| **Interpretable uncertainty decomposition** | ✗ (implicit) | ✓ (explicit) | ≈ (proxy-based) |
| **Computational efficiency crucial** | ✗ (100× passes) | ✓ (1 pass) | ≈ (5× passes) |
| **Non-Gaussian data distribution** | ≈ (assumes Gaussian for z) | ✗ (assumes Gaussian) | ✓ (distribution-free) |
| **Out-of-distribution detection** | Moderate | Good | Excellent |
| **Small dataset (n < 500)** | ≈ | Moderate | Good |
| **Large-scale production** | ✗ | ✓ | ≈ |
| **Educational/interpretability priority** | Moderate | ✓ | Moderate |
| **Time-series with distributional shifts** | Moderate | ✗ | Good |

---

## 9. Key Methodological Considerations

### 9.1 Exchangeability and Conformal Prediction

CQR's formal coverage guarantee assumes **exchangeability** of the data: the assumption that $(x_i, y_i)$ pairs could appear in any order without changing the joint distribution. In strictly temporal settings with temporal drift or seasonality, this assumption may be violated. Practitioners can mitigate this through sliding-window calibration or online learning approaches.

### 9.2 Computational Trade-offs

MCD requires 100 forward passes per prediction, making it expensive for real-time systems. HLLLA requires only 1 forward pass and a single LLLA computation (offline), making it efficient. CQR represents a middle ground with 5 ensemble members. The choice depends on whether computational efficiency or methodological properties are prioritized.

### 9.3 Uncertainty Decomposition

All three methods provide uncertainty estimates, but the decomposition differs:
- **MCD**: Implicit decomposition via variance subtraction (model-dependent)
- **HLLLA**: Principled decomposition (aleatoric from heteroscedastic output, epistemic from LLLA)
- **CQR**: Proxy-based decomposition (epistemic from ensemble variance, aleatoric from IQR)

HLLLA provides the most theoretically grounded decomposition, while CQR is more robust to model misspecification.

### 9.4 Hyperparameter Selection

All methods include hyperparameters requiring selection:
- **MCD**: Dropout rate $p_{drop}$ (validated via random search)
- **HLLLA**: Dropout rate $p_{drop}$ (validated via random search)
- **CQR**: Ensemble size $M$ (typically 5-10), bootstrap fraction, conformal calibration set size

---

## 10. Evaluation Metrics

All methods are evaluated using the same metrics:

- **Point Forecast Metrics**: 
  - MSE, MAE, RMSE, MAPE, R² (computed on point predictions μ̂)

- **Uncertainty Quantification Metrics**:
  - **PICP (Prediction Interval Coverage Probability)**: $\text{PICP} = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}(y_i \in [L_i, U_i])$
    - Target: PICP ≥ 0.95 for 95% nominal level
  - **MPIW (Mean Prediction Interval Width)**: $\text{MPIW} = \frac{1}{N}\sum_{i=1}^{N}(U_i - L_i)$
    - Lower is better (sharpness)
  - **Winkler Score**: $\text{WS} = \frac{1}{N}\sum_{i=1}^{N}[(U_i - L_i) + \frac{2}{\alpha}(\mathbb{1}(y_i < L_i)(L_i - y_i) + \mathbb{1}(y_i > U_i)(y_i - U_i))]$
    - Combines width and miscoverage penalty

---

## Conclusion

The three UQ methodologies presented—MCD, HLLLA, and CQR—each offer distinct advantages for probabilistic forecasting with Transformer networks. MCD provides a practical Bayesian approximation with good uncertainty estimates at computational cost. HLLLA offers principled uncertainty decomposition with efficiency. CQR provides distribution-free formal guarantees at moderate computational expense. The selection among these methods should be guided by application-specific requirements: whether formal coverage guarantees are necessary, whether computational efficiency is critical, and whether the data satisfies the exchangeability assumption required for conformal prediction.
