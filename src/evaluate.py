# -----------------------------------
# Phase 4A: Side-by-Side Comparison
# -----------------------------------
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Predict
y_pred_baseline = model.predict(X_test_scaled).flatten()
y_pred_physics = physics_model.predict(X_test_scaled).flatten()

# Metrics
def evaluate_predictions(y_true, y_pred, model_name="Model"):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 {model_name} Performance:")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R2 Score: {r2:.4f}")
    return mse, mae, r2

mse_base, mae_base, r2_base = evaluate_predictions(y_test, y_pred_baseline, "Baseline MLP")
mse_phys, mae_phys, r2_phys = evaluate_predictions(y_test, y_pred_physics, "Physics-Informed MLP")

# Plot True vs Predicted for both
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
plt.scatter(y_test, y_pred_baseline, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.title('Baseline MLP: True vs Predicted')
plt.xlabel('True HSE Defect Level (eV)')
plt.ylabel('Predicted')
plt.grid()

plt.subplot(1,2,2)
plt.scatter(y_test, y_pred_physics, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.title('Physics-Informed MLP: True vs Predicted')
plt.xlabel('True HSE Defect Level (eV)')
plt.ylabel('Predicted')
plt.grid()

plt.tight_layout()
plt.show()

# Plot Error Histograms
plt.figure(figsize=(12,5))

plt.hist(y_test - y_pred_baseline, bins=50, alpha=0.7, label='Baseline MLP')
plt.hist(y_test - y_pred_physics, bins=50, alpha=0.7, label='Physics-Informed MLP')
plt.title('Error Distribution: Baseline vs Physics-Informed')
plt.xlabel('Error (True - Predicted)')
plt.ylabel('Frequency')
plt.legend()
plt.grid()
plt.show()


# -----------------------------------
# Phase 4A: Side-by-Side Comparison
# -----------------------------------
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Predict
y_pred_baseline = model.predict(X_test_scaled).flatten()
y_pred_physics = physics_model.predict(X_test_scaled).flatten()

# Metrics
def evaluate_predictions(y_true, y_pred, model_name="Model"):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name} Performance:")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R2 Score: {r2:.4f}")
    return mse, mae, r2

mse_base, mae_base, r2_base = evaluate_predictions(y_test, y_pred_baseline, "Baseline MLP")
mse_phys, mae_phys, r2_phys = evaluate_predictions(y_test, y_pred_physics, "Physics-Informed MLP")

# Plot True vs Predicted for both
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
plt.scatter(y_test, y_pred_baseline, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.title('Baseline MLP: True vs Predicted')
plt.xlabel('True HSE Defect Level (eV)')
plt.ylabel('Predicted')
plt.grid()

plt.subplot(1,2,2)
plt.scatter(y_test, y_pred_physics, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.title('Physics-Informed MLP: True vs Predicted')
plt.xlabel('True HSE Defect Level (eV)')
plt.ylabel('Predicted')
plt.grid()

plt.tight_layout()
plt.show()

# Plot Error Histograms
plt.figure(figsize=(12,5))

plt.hist(y_test - y_pred_baseline, bins=50, alpha=0.7, label='Baseline MLP')
plt.hist(y_test - y_pred_physics, bins=50, alpha=0.7, label='Physics-Informed MLP')
plt.title('Error Distribution: Baseline vs Physics-Informed')
plt.xlabel('Error (True - Predicted)')
plt.ylabel('Frequency')
plt.legend()
plt.grid()
plt.show()


# -----------------------------------
# Force Plot for One Test Sample
# -----------------------------------
import shap

# Pick a test instance (e.g., first test sample)
instance_idx = 0

# Single instance prediction
shap.force_plot(
    explainer.expected_value,  # Base value
    shap_values[instance_idx], # SHAP values for one instance
    features=X_sample[instance_idx],
    feature_names=selected_features,
    matplotlib=True
)


#: Loop over different penalty weights
penalty_weights = [2.0, 5.0, 10.0]
results = []

for penalty in penalty_weights:
    print(f"\nTraining with Physics Penalty Weight = {penalty}")

    # Redefine model
    model_tuned = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    optimizer = Adam(learning_rate=0.001)
    model_tuned.compile(optimizer=optimizer, loss='mse')

    # Custom train loop (same as Phase 3) but pass new penalty weight
    train_losses = []
    for epoch in range(50):  # fewer epochs for tuning
        idx = np.random.permutation(train_size)
        X_train_shuffled = X_train_scaled[idx]
        y_train_shuffled = y_train[idx]

        for step in range(steps_per_epoch):
            start = step * batch_size
            end = (step + 1) * batch_size

            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            bandgap_batch = X_full.iloc[idx[start:end]]['host bandgap_[eV]'].values.reshape(-1, 1)
            bandgap_batch = tf.convert_to_tensor(bandgap_batch, dtype=tf.float32)

            with tf.GradientTape() as tape:
                y_pred = model_tuned(X_batch, training=True)
                mse = tf.reduce_mean(tf.square(tf.convert_to_tensor(y_batch.reshape(-1, 1), dtype=tf.float32) - y_pred))
                penalty_lower = tf.reduce_mean(tf.nn.relu(-y_pred))
                penalty_upper = tf.reduce_mean(tf.nn.relu(y_pred - bandgap_batch))
                physics_penalty = penalty_lower + penalty_upper
                loss = mse + penalty * physics_penalty

            grads = tape.gradient(loss, model_tuned.trainable_variables)
            optimizer.apply_gradients(zip(grads, model_tuned.trainable_variables))

    # Evaluate
    y_pred_tuned = model_tuned.predict(X_test_scaled).flatten()
    mse_tuned = mean_squared_error(y_test, y_pred_tuned)
    mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
    r2_tuned = r2_score(y_test, y_pred_tuned)

    results.append((penalty, mse_tuned, mae_tuned, r2_tuned))

# Show Results
print("\nHyperparameter Tuning Results:")
for penalty, mse, mae, r2 in results:
    print(f"Penalty {penalty} => MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")


# -----------------------------------
# Full Evaluation Report Script
# -----------------------------------
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# -----------------------------------
# Predict and Postprocess
# -----------------------------------
# Predict with model
y_pred_sigmoid_test = model_sigmoid.predict(X_test_scaled).flatten()

# Load bandgap values for test set
bandgap_test = X_test_unscaled['host bandgap_[eV]'].values

# Scale predictions by bandgap
y_pred_scaled = y_pred_sigmoid_test * bandgap_test

# -----------------------------------
# Calculate Accuracy Metrics
# -----------------------------------
mse = mean_squared_error(y_test, y_pred_scaled)
mae = mean_absolute_error(y_test, y_pred_scaled)
r2 = r2_score(y_test, y_pred_scaled)

# -----------------------------------
# Calculate Physics Penalty
# -----------------------------------
penalty_lower = np.maximum(0.0, -y_pred_scaled)
penalty_upper = np.maximum(0.0, y_pred_scaled - bandgap_test)
physics_penalty_total = np.mean(penalty_lower + penalty_upper)

# -----------------------------------
# Calculate Physics Violation Rate
# -----------------------------------
violations = (y_pred_scaled < 0) | (y_pred_scaled > bandgap_test)
violation_rate = np.mean(violations) * 100

# -----------------------------------
# Display Full Report
# -----------------------------------
print("\n Physics-Informed Model Evaluation Report:")
print("---------------------------------------------")
print(f"Test MSE: {mse:.6f} eV²")
print(f"Test MAE: {mae:.6f} eV")
print(f"Test R² Score: {r2:.4f}")
print(f" Average Physics Penalty: {physics_penalty_total:.6f} eV")
print(f" Physics Violation Rate: {violation_rate:.2f}%")
print("---------------------------------------------")
