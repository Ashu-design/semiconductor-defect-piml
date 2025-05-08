# -----------------------------------
# 1. Imports
# -----------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load prepared data
X_train_scaled = np.load('/content/X_train_scaled.npy')
X_test_scaled = np.load('/content/X_test_scaled.npy')
y_train = np.load('/content/y_train.npy')
y_test = np.load('/content/y_test.npy')

# -----------------------------------
# 2. Build a Standard MLP Model
# -----------------------------------
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')  # Regression output
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

model.summary()

# -----------------------------------
# 3. Train the Model
# -----------------------------------
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.1,
    epochs=150,
    batch_size=32,
    verbose=1
)

# -----------------------------------
# 4. Evaluate the Model
# -----------------------------------
y_pred = model.predict(X_test_scaled).flatten()

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Baseline MLP Model Performance:")
print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R2 Score: {r2:.4f}")

# -----------------------------------
# 5. Plot Loss Curve
# -----------------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('MLP Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid()
plt.show()

# -----------------------------------
# 6. Plot True vs Predicted
# -----------------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True HSE Defect Level (eV)')
plt.ylabel('Predicted HSE Defect Level (eV)')
plt.title('True vs Predicted Defect Levels (Baseline MLP)')
plt.grid()
plt.show()


# -----------------------------------
# 1. Correct model_predict Function
# -----------------------------------
# Recreate scaler to inverse-transform
scaler = StandardScaler()
X_all = np.vstack([X_train.values, X_test.values])
scaler.fit(X_all)

# Proper model_predict
def model_predict(X_scaled):
    # Inverse transform scaled X to real values
    X_real = scaler.inverse_transform(X_scaled)
    bandgap_column_index = selected_features.index('host bandgap_[eV]')
    bandgaps = X_real[:, bandgap_column_index]

    pred_sigmoid = model_sigmoid.predict(X_scaled)
    pred_scaled = pred_sigmoid.flatten() * bandgaps
    return pred_scaled

# 2. Sample background data
background = X_train_scaled[np.random.choice(X_train_scaled.shape[0], 100, replace=False)]

# 3. Initialize KernelExplainer
explainer = shap.KernelExplainer(model_predict, background)

# 4. Select test samples
X_sample = X_test_scaled[:300]

# 5. Calculate SHAP values
shap_values = explainer.shap_values(X_sample, nsamples=100)

# 6. SHAP Summary Plot
shap.summary_plot(
    shap_values,
    features=X_sample,
    feature_names=selected_features,
    plot_type="bar"
)

# 7. Detailed Beeswarm Plot
shap.summary_plot(
    shap_values,
    features=X_sample,
    feature_names=selected_features
)
