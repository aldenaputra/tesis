# Complete Pseudocode: 9 Model-UQ Combinations

This document provides comprehensive pseudocode for all 9 combinations of 3 architectures (RNN, TCN, Transformer) and 3 uncertainty quantification methods (MCD, HLLLA, CQR).

---

## 1. RNN (GRU/LSTM) + Monte Carlo Dropout (MCD)

### Architecture & Training

```
Algorithm RNN_MCD_Train(X_train, y_train, X_val, y_val, config):
    Input: Training features X_train (N_train, lookback, n_features)
           Training targets y_train (N_train,)
           Validation data X_val, y_val
           Hyperparameters: units, dropout_rate, lr, epochs, patience

    // Step 1: Build RNN model with dropout
    model ← Sequential()
    model.add(RNN_Layer(units, return_sequences=False))  // GRU or LSTM
    model.add(Dropout(dropout_rate))                      // Critical for MC
    model.add(Dense(1))                                   // Point prediction
    model.compile(optimizer=Adam(lr=lr), loss='mse')

    // Step 2: Train with early stopping
    callbacks ← [EarlyStopping(patience=patience, restore_best_weights=True),
                 ModelCheckpoint(best_model_path)]
    history ← model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=1)

    return model
```

### Inference with MC Dropout

```
Algorithm RNN_MCD_Inference(model, X_test, y_scaler, n_mc=100, alpha=0.05):
    Input: Trained model with dropout
           Test features X_test (N_test, lookback, n_features)
           Target standardizer y_scaler
           Number of MC passes: n_mc
           Confidence level: alpha

    // Initialize storage
    MC_predictions ← empty (N_test, n_mc)

    // Step 1: Stochastic forward passes (CRITICAL: training=True)
    for i ← 1 to n_mc do
        ŷ_i ← model.predict(X_test, training=True)      // Keep dropout ACTIVE
        MC_predictions[:, i] ← ŷ_i.squeeze()

    // Step 2: Inverse-scale to original space
    MC_predictions_original ← MC_predictions * y_scale + y_mean

    // Step 3: Compute ensemble statistics (across MC passes)
    μ_test ← mean(MC_predictions_original, axis=1)       // Ensemble mean
    σ_test ← std(MC_predictions_original, axis=1)        // Ensemble std
    L_test ← quantile(MC_predictions_original, α/2, axis=1)      // Lower quantile
    U_test ← quantile(MC_predictions_original, 1-α/2, axis=1)    // Upper quantile

    // Step 4: Uncertainty decomposition
    // Estimate aleatoric from validation residuals
    σ²_aleatoric ← var(actual_val - μ_val)
    // Epistemic = total - aleatoric
    σ²_epistemic ← σ²_test - σ²_aleatoric

    return {
        mean: μ_test,
        std: σ_test,
        lower: L_test,
        upper: U_test,
        var_epistemic: σ²_epistemic,
        var_aleatoric: σ²_aleatoric
    }
```

---

## 2. RNN (GRU/LSTM) + Heteroscedastic NLL + Last-Layer Laplace (HLLLA)

### Architecture & Training

```
Algorithm RNN_HLLLA_Train(X_train, y_train, X_val, y_val, config):
    Input: Training features X_train (N_train, lookback, n_features)
           Training targets y_train (N_train,)
           Validation data X_val, y_val
           Hyperparameters: units, dropout_rate, lr, epochs, patience

    // Step 1: Build heteroscedastic RNN (dual output heads)
    model ← Sequential()
    model.add(RNN_Layer(units, return_sequences=False))  // GRU or LSTM
    model.add(Dropout(dropout_rate))
    model.add(Dense(2))  // [μ_scaled, log(σ²)_scaled]

    // Step 2: Define heteroscedastic Gaussian NLL loss
    function loss_nll(y_true, y_pred):
        μ ← y_pred[:, 0]
        log_var ← clip(y_pred[:, 1], -20, 5)  // Numerical stability
        σ² ← exp(log_var)
        nll ← 0.5 × (log_var + (y_true - μ)² / σ²)
        return mean(nll)

    model.compile(optimizer=Adam(lr=lr), loss=loss_nll)

    // Step 3: Train with early stopping
    callbacks ← [EarlyStopping(patience=patience),
                 ModelCheckpoint(best_model_path)]
    history ← model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks)

    return model
```

### Laplace Approximation (Last-Layer)

```
Algorithm LLLA_Approximation(model, X_train, y_train, lambda_prior=1.0):
    Input: Trained model with heteroscedastic output
           Training features and targets
           Prior precision λ

    // Step 1: Get MAP predictions and compute noise variance
    y_pred ← model.predict(X_train)
    μ_train ← y_pred[:, 0]  // Mean predictions
    residuals ← y_train - μ_train
    σ²_noise ← mean(residuals²)

    // Step 2: Extract penultimate layer features φ(x)
    penultimate_model ← Model(input=model.input, output=model.layers[-2].output)
    φ_train ← penultimate_model.predict(X_train)  // (N_train, H)

    // Step 3: Build extended features with bias term
    Φ_train ← [φ_train, ones(N_train, 1)]  // (N_train, H+1)

    // Step 4: Compute diagonal Hessian approximation
    H_diag ← (1/σ²_noise) × sum(Φ_train², axis=0) + λ_prior

    // Step 5: Posterior covariance (diagonal approximation)
    Σ_diag ← 1 / H_diag  // (H+1,)

    return {Σ_diag, H_diag}
```

### Inference with HLLLA

```
Algorithm RNN_HLLLA_Inference(model, X_test, penultimate_model, Σ_diag, y_scaler):
    Input: Trained heteroscedastic model
           Test features X_test
           Penultimate features extractor
           Posterior covariance diagonal Σ_diag
           Scaler for inverse-transform

    // Step 1: Get heteroscedastic predictions
    y_pred_scaled ← model.predict(X_test)
    μ_scaled ← y_pred_scaled[:, 0]
    log_var_scaled ← y_pred_scaled[:, 1]

    // Step 2: Inverse-scale to original space
    μ_original ← μ_scaled × y_scale + y_mean
    σ²_aleatoric_original ← exp(log_var_scaled) × y_scale²

    // Step 3: Extract penultimate features and compute epistemic uncertainty
    φ_test ← penultimate_model.predict(X_test)  // (N_test, H)
    Φ_test ← [φ_test, ones(N_test, 1)]          // (N_test, H+1)

    // Step 4: Epistemic variance per-sample
    σ²_epistemic_scaled ← sum((Φ_test² × Σ_diag), axis=1)
    σ²_epistemic_original ← σ²_epistemic_scaled × y_scale²

    // Step 5: Total uncertainty
    σ²_total ← σ²_aleatoric_original + σ²_epistemic_original
    σ_total ← sqrt(σ²_total)

    // Step 6: Construct prediction intervals (Gaussian assumption)
    z ← 1.96  // 95% confidence
    L_test ← μ_original - z × σ_total
    U_test ← μ_original + z × σ_total

    return {
        mean: μ_original,
        sigma_aleatoric: sqrt(σ²_aleatoric_original),
        sigma_epistemic: sqrt(σ²_epistemic_original),
        sigma_total: σ_total,
        lower: L_test,
        upper: U_test
    }
```

---

## 3. RNN (GRU/LSTM) + Conformal Quantile Regression (CQR)

### Ensemble Training

```
Algorithm RNN_CQR_Train_Ensemble(X_train, y_train, X_val, y_val, M=5, config):
    Input: Training features and targets
           Validation data
           Number of ensemble members M
           Quantile levels: TAUS = [0.025, 0.5, 0.975]

    ensemble_members ← empty list of size M

    for m ← 1 to M do
        // Step 1: Generate bootstrap sample
        seed_m ← SEED_BASE + 137 × m
        idx_bootstrap ← sample_without_replacement(range(N_train), N_train, seed=seed_m)
        X_bootstrap ← X_train[idx_bootstrap]
        y_bootstrap ← y_train[idx_bootstrap]

        // Step 2: Build quantile RNN (3 output heads for 3 quantiles)
        model_m ← Sequential()
        model_m.add(RNN_Layer(units, return_sequences=False))  // GRU or LSTM
        model_m.add(Dropout(dropout_rate))
        model_m.add(Dense(len(TAUS)))  // 3 quantiles

        // Step 3: Define pinball loss for multi-quantile regression
        function loss_pinball(y_true, y_pred):
            // y_pred shape: (batch, 3) for [q_0.025, q_0.5, q_0.975]
            errors ← y_true - y_pred  // (batch, 3)
            pinball ← zeros_like(errors)
            for j ← 0 to 2:
                tau_j ← TAUS[j]
                pinball[:, j] ← max(tau_j × errors[:, j],
                                    (tau_j - 1) × errors[:, j])
            return mean(sum(pinball, axis=1))

        model_m.compile(optimizer=Adam(lr=lr), loss=loss_pinball)

        // Step 4: Train on bootstrap sample
        callbacks ← [EarlyStopping(patience=patience),
                     ModelCheckpoint(f'model_{m}.keras')]
        model_m.fit(X_bootstrap, y_bootstrap,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks)

        ensemble_members.append(model_m)

    return ensemble_members
```

### Conformal Calibration & Inference

```
Algorithm RNN_CQR_Inference(ensemble_members, X_train, y_train,
                             X_val, y_val, X_test, y_test, y_scaler, alpha=0.05):
    Input: List of M trained ensemble members
           Training/validation/test data
           Target scaler
           Significance level alpha

    TAUS ← [alpha/2, 0.5, 1-alpha/2]  // [0.025, 0.5, 0.975]

    // ============ PHASE 1: Predict on all splits ============
    // For training, validation, test: get quantiles from each member
    for split in {train, val, test}:
        L_members, M_members, U_members ← empty lists

        for m ← 1 to M do
            q_tau ← ensemble_members[m].predict(X_split)  // (N, 3) standardized
            q_original ← q_tau × y_scale + y_mean         // Inverse-scale
            q_sorted ← sort(q_original, axis=1)           // Enforce L ≤ M ≤ U

            L_members.append(q_sorted[:, 0])  // Lower quantile
            M_members.append(q_sorted[:, 1])  // Median
            U_members.append(q_sorted[:, 2])  // Upper quantile

        // Aggregate across ensemble
        L_bar[split] ← mean(L_members, axis=0)
        M_bar[split] ← mean(M_members, axis=0)
        U_bar[split] ← mean(U_members, axis=0)

        // Uncertainty decomposition
        var_epistemic[split] ← var(M_members, axis=0, ddof=1)
        IQR ← U_bar[split] - L_bar[split]
        sigma_aleatoric ← IQR / 3.92  // Gaussian approximation
        var_aleatoric[split] ← sigma_aleatoric²

    // ============ PHASE 2: Split Conformal Calibration (on validation) ============
    // Compute non-conformity scores on validation
    non_conformity ← zeros(N_val)
    for i ← 1 to N_val do
        non_conformity[i] ← max(L_bar_val[i] - y_val[i],
                                y_val[i] - U_bar_val[i],
                                0)

    // Compute conformal threshold
    q_hat ← quantile(non_conformity, 1 - alpha, method='higher')

    // ============ PHASE 3: Conformalize test intervals ============
    L_test_conformalized ← L_bar_test - q_hat
    U_test_conformalized ← U_bar_test + q_hat

    return {
        mean: M_bar_test,
        lower: L_test_conformalized,
        upper: U_test_conformalized,
        var_epistemic: var_epistemic[test],
        var_aleatoric: var_aleatoric[test]
    }
```

---

## 4. TCN + Monte Carlo Dropout (MCD)

### Architecture & Training

```
Algorithm TCN_MCD_Train(X_train, y_train, X_val, y_val, config):
    Input: Training features X_train (N_train, lookback, n_features)
           Training targets y_train (N_train,)
           Validation data
           Hyperparameters: filters, kernel_size, dilations, dropout, lr

    // Step 1: Build TCN with dropout blocks
    model ← Sequential()
    model.add(Input(shape=(lookback, n_features)))

    // Build TCN blocks with residual connections
    for num_stacks iterations do
        for each dilation_rate in dilations do
            // TCN block = 2 causal conv layers + residual connection
            x ← Conv1D(filters, kernel_size, padding='causal',
                       dilation_rate=dilation_rate)(x)
            x ← LayerNormalization()(x)
            x ← Activation('relu')(x)
            x ← SpatialDropout1D(dropout)(x)  // Critical for MC

            x ← Conv1D(filters, kernel_size, padding='causal',
                       dilation_rate=dilation_rate)(x)
            x ← LayerNormalization()(x)
            x ← Activation('relu')(x)
            x ← SpatialDropout1D(dropout)(x)

            // Residual connection with projection if needed
            if input_channels ≠ filters:
                input ← Conv1D(filters, 1)(input)
            x ← Add()([x, input])

    // Compress and predict
    model.add(Conv1D(1, 1, padding='same'))
    model.add(Lambda(λ t: t[:, -1, :]))  // Take last timestep
    model.add(Dense(1))

    model.compile(optimizer=Adam(lr=lr), loss='mse')

    // Step 2: Train
    callbacks ← [EarlyStopping(patience=patience)]
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    return model
```

### Inference

```
Algorithm TCN_MCD_Inference(model, X_test, y_scaler, n_mc=100, alpha=0.05):
    Input: Trained TCN model with spatial dropout
           Test features
           Scaler for inverse-transform
           Number of MC passes

    // Identical to RNN_MCD_Inference
    MC_predictions ← empty (N_test, n_mc)

    for i ← 1 to n_mc do
        ŷ_i ← model.predict(X_test, training=True)
        MC_predictions[:, i] ← ŷ_i.squeeze()

    MC_predictions_original ← MC_predictions × y_scale + y_mean

    μ_test ← mean(MC_predictions_original, axis=1)
    σ_test ← std(MC_predictions_original, axis=1)
    L_test ← quantile(MC_predictions_original, α/2, axis=1)
    U_test ← quantile(MC_predictions_original, 1-α/2, axis=1)

    // Uncertainty decomposition
    σ²_aleatoric ← var(actual_val - μ_val)
    σ²_epistemic ← σ²_test - σ²_aleatoric

    return {mean, std, lower, upper, var_epistemic, var_aleatoric}
```

---

## 5. TCN + Heteroscedastic NLL + Last-Layer Laplace (HLLLA)

### Architecture & Training

```
Algorithm TCN_HLLLA_Train(X_train, y_train, X_val, y_val, config):
    Input: Training features and targets
           Validation data
           TCN hyperparameters

    // Step 1: Build TCN with heteroscedastic output (2 heads)
    model ← Sequential()
    model.add(Input(shape=(lookback, n_features)))

    // Build TCN blocks (same as MCD)
    for num_stacks iterations do
        for each dilation_rate in dilations do
            x ← Conv1D_block_with_residual(...)

    // Final layers
    model.add(Conv1D(1, 1, padding='same'))
    model.add(Lambda(λ t: t[:, -1, :]))
    model.add(Dense(2))  // [μ_scaled, log(σ²)_scaled]

    // Step 2: Heteroscedastic NLL loss
    function loss_nll(y_true, y_pred):
        μ ← y_pred[:, 0]
        log_var ← clip(y_pred[:, 1], -20, 5)
        σ² ← exp(log_var)
        nll ← 0.5 × (log_var + (y_true - μ)² / σ²)
        return mean(nll)

    model.compile(optimizer=Adam(lr=lr), loss=loss_nll)

    // Step 3: Train
    callbacks ← [EarlyStopping(patience=patience)]
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, callbacks=callbacks)

    return model
```

### Laplace + Inference

```
Algorithm TCN_HLLLA_Inference(model, X_train, y_train, X_penultimate,
                              X_test, y_scaler, lambda_prior=1.0):
    Input: Trained TCN model
           Training data for Laplace approximation
           Penultimate features
           Test data

    // Step 1: Laplace approximation on last layer
    y_pred_train ← model.predict(X_train)
    μ_train ← y_pred_train[:, 0]
    residuals ← y_train - μ_train
    σ²_noise ← mean(residuals²)

    // Extract penultimate features
    penultimate_model ← Model(input=model.input, output=model.layers[-2].output)
    φ_train ← penultimate_model.predict(X_train)
    Φ_train ← [φ_train, ones(N_train, 1)]

    // Diagonal Hessian
    H_diag ← (1/σ²_noise) × sum(Φ_train², axis=0) + lambda_prior
    Σ_diag ← 1 / H_diag

    // Step 2: Test predictions
    y_pred_test ← model.predict(X_test)
    μ_scaled ← y_pred_test[:, 0]
    log_var_scaled ← y_pred_test[:, 1]

    // Inverse-scale
    μ_original ← μ_scaled × y_scale + y_mean
    σ²_aleatoric ← exp(log_var_scaled) × y_scale²

    // Step 3: Epistemic uncertainty via Laplace
    φ_test ← penultimate_model.predict(X_test)
    Φ_test ← [φ_test, ones(N_test, 1)]
    σ²_epistemic ← sum((Φ_test² × Σ_diag), axis=1) × y_scale²

    // Step 4: Total uncertainty and intervals
    σ²_total ← σ²_aleatoric + σ²_epistemic
    σ_total ← sqrt(σ²_total)
    z ← 1.96
    L_test ← μ_original - z × σ_total
    U_test ← μ_original + z × σ_total

    return {mean, sigma_aleatoric, sigma_epistemic, lower, upper}
```

---

## 6. TCN + Conformal Quantile Regression (CQR)

### Ensemble Training

```
Algorithm TCN_CQR_Train_Ensemble(X_train, y_train, X_val, y_val, M=5, config):
    Input: Training/validation data
           Number of ensemble members M
           Quantile levels: TAUS = [0.025, 0.5, 0.975]

    ensemble_members ← empty list

    for m ← 1 to M do
        // Step 1: Bootstrap sample
        seed_m ← SEED_BASE + 137 × m
        idx_bs ← sample_without_replacement(range(N_train), N_train, seed=seed_m)
        X_bs ← X_train[idx_bs]
        y_bs ← y_train[idx_bs]

        // Step 2: Build TCN for quantile regression (3 outputs)
        model_m ← build_tcn_architecture(config)  // Standard TCN blocks
        model_m.add(Lambda(λ t: t[:, -1, :]))
        model_m.add(Dense(len(TAUS)))  // 3 quantile heads

        // Step 3: Pinball loss
        model_m.compile(optimizer=Adam(lr=lr), loss=pinball_loss(TAUS))

        // Step 4: Train on bootstrap
        model_m.fit(X_bs, y_bs, validation_data=(X_val, y_val),
                    epochs=epochs, batch_size=batch_size)

        ensemble_members.append(model_m)

    return ensemble_members
```

### Conformal Calibration & Inference

```
Algorithm TCN_CQR_Inference(ensemble_members, X_train, y_train,
                            X_val, y_val, X_test, y_scaler, alpha=0.05):
    Input: M ensemble TCN models
           Training/validation/test data
           Scaler
           Significance level

    TAUS ← [alpha/2, 0.5, 1-alpha/2]

    // ============ PHASE 1: Ensemble predictions ============
    for split in {train, val, test}:
        L_members, M_members, U_members ← empty lists

        for m ← 1 to M do
            q_tau ← ensemble_members[m].predict(X_split)  // (N, 3)
            q_original ← q_tau × y_scale + y_mean
            q_sorted ← sort(q_original, axis=1)

            L_members.append(q_sorted[:, 0])
            M_members.append(q_sorted[:, 1])
            U_members.append(q_sorted[:, 2])

        L_bar[split] ← mean(L_members)
        M_bar[split] ← mean(M_members)
        U_bar[split] ← mean(U_members)

        var_epistemic[split] ← var(M_members, ddof=1)
        var_aleatoric[split] ← (IQR/3.92)²

    // ============ PHASE 2: Conformal calibration ============
    non_conformity ← max(L_bar_val - y_val, y_val - U_bar_val, 0)
    q_hat ← quantile(non_conformity, 1-alpha, method='higher')

    // ============ PHASE 3: Conformalize ============
    L_test_final ← L_bar_test - q_hat
    U_test_final ← U_bar_test + q_hat

    return {mean, lower, upper, var_epistemic, var_aleatoric}
```

---

## 7. Transformer + Monte Carlo Dropout (MCD)

### Architecture & Training

```
Algorithm Transformer_MCD_Train(X_train, y_train, X_val, y_val, config):
    Input: Training/validation data
           Hyperparameters: d_model, num_heads, dff, num_layers, dropout

    // Step 1: Build Transformer encoder
    model ← Sequential()
    model.add(Input(shape=(lookback, n_features)))

    // Dense projection to d_model
    model.add(Dense(d_model))

    // Positional encoding
    model.add(SinusoidalPositionalEncoding(d_model))

    // Transformer encoder blocks
    for layer ← 1 to num_layers do
        // Multi-head self-attention
        attn ← MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
        attn_out ← attn(x, x, use_causal_mask=True)
        x ← Add()([x, Dropout(dropout)(attn_out)])
        x ← LayerNormalization()(x)

        // Feed-forward network
        ff ← Dense(dff, activation='relu')(x)
        ff ← Dropout(dropout)(ff)
        ff ← Dense(d_model)(ff)
        x ← Add()([x, Dropout(dropout)(ff)])
        x ← LayerNormalization()(x)

    // Take last timestep and predict
    model.add(Lambda(λ t: t[:, -1, :]))
    model.add(Dense(1))  // Point prediction

    model.compile(optimizer=Adam(lr=lr), loss='mse')

    // Step 2: Train
    callbacks ← [EarlyStopping(patience=patience)]
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    return model
```

### Inference

```
Algorithm Transformer_MCD_Inference(model, X_test, y_scaler, n_mc=100, alpha=0.05):
    Input: Trained Transformer model with dropout
           Test data
           Scaler
           MC parameters

    // Identical to RNN/TCN MCD inference
    MC_predictions ← empty (N_test, n_mc)

    for i ← 1 to n_mc do
        ŷ_i ← model.predict(X_test, training=True)  // training=True keeps dropout active
        MC_predictions[:, i] ← ŷ_i.squeeze()

    MC_predictions_original ← MC_predictions × y_scale + y_mean

    μ_test ← mean(MC_predictions_original, axis=1)
    σ_test ← std(MC_predictions_original, axis=1)
    L_test ← quantile(MC_predictions_original, α/2)
    U_test ← quantile(MC_predictions_original, 1-α/2)

    σ²_aleatoric ← var(actual_val - μ_val)
    σ²_epistemic ← σ²_test - σ²_aleatoric

    return {mean, std, lower, upper, var_epistemic, var_aleatoric}
```

---

## 8. Transformer + Heteroscedastic NLL + Last-Layer Laplace (HLLLA)

### Architecture & Training

```
Algorithm Transformer_HLLLA_Train(X_train, y_train, X_val, y_val, config):
    Input: Training/validation data
           Transformer hyperparameters

    // Step 1: Build Transformer with heteroscedastic output (2 heads)
    model ← Sequential()
    model.add(Input(shape=(lookback, n_features)))
    model.add(Dense(d_model))
    model.add(SinusoidalPositionalEncoding(d_model))

    // Transformer encoder blocks (same as MCD)
    for layer ← 1 to num_layers do
        x ← MultiHeadAttention_block(x, ...)
        x ← FeedForward_block(x, ...)

    // Take last timestep
    model.add(Lambda(λ t: t[:, -1, :]))

    // Heteroscedastic output (2 heads)
    model.add(Dense(2))  // [μ_scaled, log(σ²)_scaled]

    // Step 2: Heteroscedastic NLL loss
    function loss_nll(y_true, y_pred):
        μ ← y_pred[:, 0]
        log_var ← clip(y_pred[:, 1], -20, 5)
        σ² ← exp(log_var)
        nll ← 0.5 × (log_var + (y_true - μ)² / σ²)
        return mean(nll)

    model.compile(optimizer=Adam(lr=lr), loss=loss_nll)

    // Step 3: Train
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, callbacks=[EarlyStopping(...)])

    return model
```

### Laplace + Inference

```
Algorithm Transformer_HLLLA_Inference(model, X_train, y_train, X_penultimate,
                                      X_test, y_scaler, lambda_prior=1.0):
    Input: Trained Transformer model
           Training data
           Penultimate features
           Test data
           Scaler

    // Step 1: Laplace approximation (identical to TCN/RNN)
    y_pred_train ← model.predict(X_train)
    μ_train ← y_pred_train[:, 0]
    σ²_noise ← var(y_train - μ_train)

    penultimate_model ← Model(input=model.input, output=model.layers[-2].output)
    φ_train ← penultimate_model.predict(X_train)
    Φ_train ← [φ_train, ones(N_train, 1)]

    H_diag ← (1/σ²_noise) × sum(Φ_train², axis=0) + lambda_prior
    Σ_diag ← 1 / H_diag

    // Step 2: Test predictions
    y_pred_test ← model.predict(X_test)
    μ_scaled ← y_pred_test[:, 0]
    log_var_scaled ← y_pred_test[:, 1]

    μ_original ← μ_scaled × y_scale + y_mean
    σ²_aleatoric ← exp(log_var_scaled) × y_scale²

    // Step 3: Epistemic uncertainty
    φ_test ← penultimate_model.predict(X_test)
    Φ_test ← [φ_test, ones(N_test, 1)]
    σ²_epistemic ← sum((Φ_test² × Σ_diag), axis=1) × y_scale²

    // Step 4: Intervals
    σ²_total ← σ²_aleatoric + σ²_epistemic
    σ_total ← sqrt(σ²_total)
    L_test ← μ_original - 1.96 × σ_total
    U_test ← μ_original + 1.96 × σ_total

    return {mean, sigma_aleatoric, sigma_epistemic, lower, upper}
```

---

## 9. Transformer + Conformal Quantile Regression (CQR)

### Ensemble Training

```
Algorithm Transformer_CQR_Train_Ensemble(X_train, y_train, X_val, y_val, M=5, config):
    Input: Training/validation data
           Number of ensemble members
           Quantile levels: TAUS = [0.025, 0.5, 0.975]

    ensemble_members ← empty list

    for m ← 1 to M do
        // Step 1: Bootstrap
        seed_m ← SEED_BASE + 137 × m
        idx_bs ← sample_without_replacement(range(N_train), N_train, seed=seed_m)
        X_bs ← X_train[idx_bs]
        y_bs ← y_train[idx_bs]

        // Step 2: Build Transformer for quantile regression
        model_m ← build_transformer_architecture(config)
        model_m.add(Lambda(λ t: t[:, -1, :]))
        model_m.add(Dense(len(TAUS)))  // 3 quantile outputs

        // Step 3: Pinball loss
        model_m.compile(optimizer=Adam(lr=lr), loss=pinball_loss(TAUS))

        // Step 4: Train on bootstrap
        model_m.fit(X_bs, y_bs, validation_data=(X_val, y_val),
                    epochs=epochs, batch_size=batch_size)

        ensemble_members.append(model_m)

    return ensemble_members
```

### Conformal Calibration & Inference

```
Algorithm Transformer_CQR_Inference(ensemble_members, X_train, y_train,
                                    X_val, y_val, X_test, y_scaler, alpha=0.05):
    Input: M ensemble Transformer models
           Training/validation/test data
           Scaler
           Significance level

    TAUS ← [alpha/2, 0.5, 1-alpha/2]

    // ============ PHASE 1: Ensemble predictions (identical to TCN/RNN CQR) ============
    for split in {train, val, test}:
        L_members, M_members, U_members ← empty lists

        for m ← 1 to M do
            q_tau ← ensemble_members[m].predict(X_split)  // (N, 3)
            q_original ← q_tau × y_scale + y_mean
            q_sorted ← sort(q_original, axis=1)

            L_members.append(q_sorted[:, 0])
            M_members.append(q_sorted[:, 1])
            U_members.append(q_sorted[:, 2])

        L_bar[split] ← mean(L_members)
        M_bar[split] ← mean(M_members)
        U_bar[split] ← mean(U_members)

        var_epistemic[split] ← var(M_members, ddof=1)
        IQR ← U_bar[split] - L_bar[split]
        var_aleatoric[split] ← (IQR/3.92)²

    // ============ PHASE 2: Conformal calibration (on validation) ============
    non_conformity ← max(L_bar_val - y_val, y_val - U_bar_val, 0)
    q_hat ← quantile(non_conformity, 1-alpha, method='higher')

    // ============ PHASE 3: Conformalize test intervals ============
    L_test_final ← L_bar_test - q_hat
    U_test_final ← U_bar_test + q_hat

    return {
        mean: M_bar_test,
        lower: L_test_final,
        upper: U_test_final,
        var_epistemic: var_epistemic[test],
        var_aleatoric: var_aleatoric[test]
    }
```

---

## Summary: Key Algorithm Differences

| **Aspect** | **MCD** | **HLLLA** | **CQR** |
|---|---|---|---|
| **Training Loss** | MSE (standard) | Heteroscedastic NLL | Pinball (quantile) |
| **Output Heads** | 1 (point) | 2 (μ, log σ²) | 3 (quantiles) |
| **Inference Passes** | 100 (MC) | 1 (deterministic) | 5 (ensemble) |
| **Epistemic Approach** | Ensemble variance | Laplace approximation | Ensemble spread |
| **Aleatoric Approach** | Var(residuals) | Learned output | IQR/3.92 proxy |
| **Calibration** | Implicit (training) | Implicit (training) | Explicit (conformal) |
| **Formal Guarantee** | No | No | Yes (distribution-free) |
| **Architecture Specific?** | No (all same) | No (all same) | No (all same) |
| **Key Innovation** | Dropout as Bayesian | Last-layer posteriors | Distribution-free CI |

---

## Cross-Cutting Implementation Patterns

### Pattern 1: Inverse-Scaling for Quantile Preservation

```
Algorithm Inverse_Scale_Quantiles(q_scaled, y_scaler):
    // Linear transformation preserves quantile order
    q_original ← q_scaled × y_scale + y_mean
    // Order: L ≤ M ≤ U is preserved
    q_sorted ← sort(q_original)
    return q_sorted
```

### Pattern 2: Uncertainty Decomposition (MCD & CQR)

```
Algorithm Decompose_Uncertainty(M_members, training_residuals):
    // Epistemic: variance across ensemble
    σ²_epistemic ← var(M_members, axis=ensemble_dimension)

    // Aleatoric: from residuals (MCD) or IQR (CQR)
    σ²_aleatoric ← var(training_residuals)  // MCD
    σ²_aleatoric ← (IQR / 3.92)²            // CQR

    σ²_total ← σ²_epistemic + σ²_aleatoric
    return {epistemic, aleatoric, total}
```

### Pattern 3: Prediction Intervals from Uncertainty

```
Algorithm Construct_PI(y_pred, sigma_total, alpha, z=1.96):
    L ← y_pred - z × sigma_total
    U ← y_pred + z × sigma_total
    return {L, U}
```

### Pattern 4: UQ Metrics Computation

```
Algorithm Compute_UQ_Metrics(y_true, L, U, alpha=0.05):
    // Coverage: fraction of points inside PI
    coverage ← (y_true >= L) AND (y_true <= U)
    PICP ← mean(coverage)

    // Width: average PI width
    width ← U - L
    MPIW ← mean(width)

    // Winkler: width + penalty for misses
    penalties ← zeros(len(y_true))
    for each sample:
        if y_true < L: penalty = (2/alpha) × (L - y_true)
        else if y_true > U: penalty = (2/alpha) × (y_true - U)
        else: penalty = 0
    Winkler ← mean(width + penalties)

    return {PICP, MPIW, Winkler}
```
