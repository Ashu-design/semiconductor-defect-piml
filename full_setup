# -----------------------------------
# 1. Imports
# -----------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------
# 2. Load the Dataset
# -----------------------------------

df = pd.read_csv('/content/Dataset_semiconductor_defectlevels_v1.csv')

# -----------------------------------
# 3. Subset: Only Keep Rows with HSE Defect Level Available
# -----------------------------------
df = df.dropna(subset=['hse defect level (relative to VBM)_[eV]'])

print(f"Dataset after filtering: {df.shape[0]} rows")

# -----------------------------------
# 4. Feature Selection
# -----------------------------------
# Selected Features (Physics-informed + Practical)
selected_features = [
    'pbe defect level (relative to VBM)_[eV]',
    'host bandgap_[eV]',
    'host lattice constant_[Ang.]',
    'host_epsilon',
    'charge_from',
    'charge_to',
    'is_interstitial',
    'is_a_latt',
    'is_b_latt'
]

# Target
target = 'hse defect level (relative to VBM)_[eV]'

X = df[selected_features]
y = df[target]

# -----------------------------------
# 5. Train-Test Split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# -----------------------------------
# 6. Feature Scaling
# -----------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------
# 7. Check Quick Statistics
# -----------------------------------
print("\nFeature Summary after Scaling:")
print(pd.DataFrame(X_train_scaled, columns=selected_features).describe())

# -----------------------------------
# 8. Save Prepared Data (Optional)
# -----------------------------------
np.save('/content/X_train_scaled.npy', X_train_scaled)
np.save('/content/X_test_scaled.npy', X_test_scaled)
np.save('/content/y_train.npy', y_train.values)
np.save('/content/y_test.npy', y_test.values)

print("\n Data Preparation Completed and Saved")


# -----------------------------------
# 7. Check Physical Violations
# -----------------------------------

# Load original full data
import pandas as pd

df_full = pd.read_csv('/content/Dataset_semiconductor_defectlevels_v1.csv')
df_full = df_full.dropna(subset=['hse defect level (relative to VBM)_[eV]'])

selected_features = [
    'pbe defect level (relative to VBM)_[eV]',
    'host bandgap_[eV]',
    'host lattice constant_[Ang.]',
    'host_epsilon',
    'charge_from',
    'charge_to',
    'is_interstitial',
    'is_a_latt',
    'is_b_latt'
]

X_full = df_full[selected_features].reset_index(drop=True)

from sklearn.model_selection import train_test_split
_, X_test_unscaled, _, _ = train_test_split(X_full, df_full['hse defect level (relative to VBM)_[eV]'], test_size=0.2, random_state=42)

# Extract host bandgap for test samples
bandgap_test = X_test_unscaled['host bandgap_[eV]'].values

# Now check violations
violations_lower = np.sum(y_pred < 0)
violations_upper = np.sum(y_pred > bandgap_test)
total_violations = violations_lower + violations_upper

violation_percentage = (total_violations / len(y_pred)) * 100

print(f"\n Physical Violations in Test Set:")
print(f"Predicted < 0 violations: {violations_lower}")
print(f"Predicted > Bandgap violations: {violations_upper}")
print(f"Total Violations: {total_violations} out of {len(y_pred)} samples")
print(f"Violation Rate: {violation_percentage:.2f}%")


# -----------------------------------
# 1. Imports
# -----------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load prepared data
X_train_scaled = np.load('/content/X_train_scaled.npy')
X_test_scaled = np.load('/content/X_test_scaled.npy')
y_train = np.load('/content/y_train.npy')
y_test = np.load('/content/y_test.npy')

# Also load original (unscaled) training features to access bandgap
df_full = pd.read_csv('/content/Dataset_semiconductor_defectlevels_v1.csv')
df_full = df_full.dropna(subset=['hse defect level (relative to VBM)_[eV]'])

selected_features = [
    'pbe defect level (relative to VBM)_[eV]',
    'host bandgap_[eV]',
    'host lattice constant_[Ang.]',
    'host_epsilon',
    'charge_from',
    'charge_to',
    'is_interstitial',
    'is_a_latt',
    'is_b_latt'
]

X_full = df_full[selected_features].reset_index(drop=True)
_, X_test_unscaled, _, y_test_unscaled = train_test_split(X_full, df_full['hse defect level (relative to VBM)_[eV]'], test_size=0.2, random_state=42)

# -----------------------------------
# 2. Define Physics-Informed Custom Loss
# -----------------------------------
def physics_informed_loss(y_true, y_pred, bandgap_batch):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    # Physics constraint penalties
    penalty_lower = tf.reduce_mean(tf.nn.relu(-y_pred))  # y_pred < 0
    penalty_upper = tf.reduce_mean(tf.nn.relu(y_pred - bandgap_batch))  # y_pred > Eg

    physics_penalty = penalty_lower + penalty_upper

    total_loss = mse + 1.0 * physics_penalty  # 1.0 is penalty weight (can tune)
    return total_loss

# -----------------------------------
# 3. Build Physics-Informed MLP Model
# -----------------------------------
input_dim = X_train_scaled.shape[1]

# Define model
physics_model = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')
])

optimizer = Adam(learning_rate=0.001)

# Placeholder compile (we will override loss manually in training loop)
physics_model.compile(optimizer=optimizer, loss='mse')

# -----------------------------------
# 4. Custom Training Loop with Physics-Informed Loss
# -----------------------------------
batch_size = 32
epochs = 150
train_size = X_train_scaled.shape[0]
steps_per_epoch = train_size // batch_size

train_losses = []

for epoch in range(epochs):
    epoch_losses = []

    idx = np.random.permutation(train_size)
    X_train_shuffled = X_train_scaled[idx]
    y_train_shuffled = y_train[idx]

    for step in range(steps_per_epoch):
        start = step * batch_size
        end = (step + 1) * batch_size

        X_batch = X_train_shuffled[start:end]
        y_batch = y_train_shuffled[start:end]

        # Need bandgap from unscaled features
        bandgap_batch = X_full.iloc[idx[start:end]]['host bandgap_[eV]'].values
        bandgap_batch = bandgap_batch.reshape(-1, 1)
        bandgap_batch = tf.convert_to_tensor(bandgap_batch, dtype=tf.float32)

        with tf.GradientTape() as tape:
            y_pred = physics_model(X_batch, training=True)
            loss = physics_informed_loss(tf.convert_to_tensor(y_batch.reshape(-1, 1), dtype=tf.float32), y_pred, bandgap_batch)

        grads = tape.gradient(loss, physics_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, physics_model.trainable_variables))

        epoch_losses.append(loss.numpy())

    avg_epoch_loss = np.mean(epoch_losses)
    train_losses.append(avg_epoch_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

# -----------------------------------
# 5. Evaluate Physics-Informed Model
# -----------------------------------
y_pred_physics = physics_model.predict(X_test_scaled).flatten()

mse_physics = mean_squared_error(y_test, y_pred_physics)
mae_physics = mean_absolute_error(y_test, y_pred_physics)
r2_physics = r2_score(y_test, y_pred_physics)

print("\n📊 Physics-Informed MLP Model Performance:")
print(f"Test MSE: {mse_physics:.4f}")
print(f"Test MAE: {mae_physics:.4f}")
print(f"Test R2 Score: {r2_physics:.4f}")

# -----------------------------------
# 6. Plot Loss Curve
# -----------------------------------
plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Train Loss (Physics-Informed)')
plt.title('Physics-Informed MLP Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.show()

# -----------------------------------
# 7. True vs Predicted Plot
# -----------------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_physics, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True HSE Defect Level (eV)')
plt.ylabel('Predicted HSE Defect Level (eV)')
plt.title('True vs Predicted Defect Levels (Physics-Informed MLP)')
plt.grid()
plt.show()


# -----------------------------------
# 1. Imports
# -----------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load prepared data
X_train_scaled = np.load('/content/X_train_scaled.npy')
X_test_scaled = np.load('/content/X_test_scaled.npy')
y_train = np.load('/content/y_train.npy')
y_test = np.load('/content/y_test.npy')

# Load full (unscaled) data to get bandgap
import pandas as pd
df_full = pd.read_csv('/content/Dataset_semiconductor_defectlevels_v1.csv')
df_full = df_full.dropna(subset=['hse defect level (relative to VBM)_[eV]'])

selected_features = [
    'pbe defect level (relative to VBM)_[eV]',
    'host bandgap_[eV]',
    'host lattice constant_[Ang.]',
    'host_epsilon',
    'charge_from',
    'charge_to',
    'is_interstitial',
    'is_a_latt',
    'is_b_latt'
]

X_full = df_full[selected_features].reset_index(drop=True)

from sklearn.model_selection import train_test_split
_, X_test_unscaled, _, _ = train_test_split(X_full, df_full['hse defect level (relative to VBM)_[eV]'], test_size=0.2, random_state=42)

# -----------------------------------
# 2. Define Smarter Physics-Informed Loss
# -----------------------------------
def improved_physics_informed_loss(y_true, y_pred, bandgap_batch):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    # Only penalize when there is violation
    penalty_lower = tf.reduce_sum(tf.maximum(0.0, -y_pred))
    penalty_upper = tf.reduce_sum(tf.maximum(0.0, y_pred - bandgap_batch))

    physics_penalty = (penalty_lower + penalty_upper) / tf.cast(tf.shape(y_true)[0], tf.float32)

    total_loss = mse + 1.0 * physics_penalty  # Penalty weight = 1.0
    return total_loss

# -----------------------------------
# 3. Build Optimized Physics-Informed MLP
# -----------------------------------
input_dim = X_train_scaled.shape[1]

model_optimized = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')
])

optimizer = Adam(learning_rate=0.001)
model_optimized.compile(optimizer=optimizer, loss='mse')  # Placeholder loss

# -----------------------------------
# 4. Custom Training Loop with Smarter Loss
# -----------------------------------
batch_size = 32
epochs = 150
train_size = X_train_scaled.shape[0]
steps_per_epoch = train_size // batch_size

train_losses = []

for epoch in range(epochs):
    epoch_losses = []

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
            y_pred = model_optimized(X_batch, training=True)
            loss = improved_physics_informed_loss(
                tf.convert_to_tensor(y_batch.reshape(-1, 1), dtype=tf.float32),
                y_pred,
                bandgap_batch
            )

        grads = tape.gradient(loss, model_optimized.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_optimized.trainable_variables))

        epoch_losses.append(loss.numpy())

    avg_epoch_loss = np.mean(epoch_losses)
    train_losses.append(avg_epoch_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

# -----------------------------------
# 5. Evaluate Optimized Physics-Informed Model
# -----------------------------------
y_pred_optimized = model_optimized.predict(X_test_scaled).flatten()

mse_optimized = mean_squared_error(y_test, y_pred_optimized)
mae_optimized = mean_absolute_error(y_test, y_pred_optimized)
r2_optimized = r2_score(y_test, y_pred_optimized)

print("\n📊 Optimized Physics-Informed MLP Performance:")
print(f"Test MSE: {mse_optimized:.4f}")
print(f"Test MAE: {mae_optimized:.4f}")
print(f"Test R2 Score: {r2_optimized:.4f}")

# -----------------------------------
# 6. Plot Loss Curve
# -----------------------------------
plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Train Loss (Optimized Physics-Informed)')
plt.title('Optimized Physics-Informed MLP Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.show()

# -----------------------------------
# 7. True vs Predicted Plot
# -----------------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_optimized, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True HSE Defect Level (eV)')
plt.ylabel('Predicted HSE Defect Level (eV)')
plt.title('True vs Predicted Defect Levels (Optimized Physics-Informed MLP)')
plt.grid()
plt.show()

# -----------------------------------
# 8. Check Physics Violations
# -----------------------------------
# Bandgap values for test set
bandgap_test = X_test_unscaled['host bandgap_[eV]'].values

# How many predictions violate 0 < pred < Eg
violations = np.sum((y_pred_optimized < 0) | (y_pred_optimized > bandgap_test))
violation_percentage = (violations / len(y_pred_optimized)) * 100

print(f"\n Physics Violations in Test Set: {violations} out of {len(y_pred_optimized)}")
print(f" Violation Rate: {violation_percentage:.2f}%")


# -----------------------------------
# 1. Imports
# -----------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load prepared data
X_train_scaled = np.load('/content/X_train_scaled.npy')
X_test_scaled = np.load('/content/X_test_scaled.npy')
y_train = np.load('/content/y_train.npy')
y_test = np.load('/content/y_test.npy')

# Load full (unscaled) data to get bandgap
import pandas as pd
df_full = pd.read_csv('/content/Dataset_semiconductor_defectlevels_v1.csv')
df_full = df_full.dropna(subset=['hse defect level (relative to VBM)_[eV]'])

selected_features = [
    'pbe defect level (relative to VBM)_[eV]',
    'host bandgap_[eV]',
    'host lattice constant_[Ang.]',
    'host_epsilon',
    'charge_from',
    'charge_to',
    'is_interstitial',
    'is_a_latt',
    'is_b_latt'
]

X_full = df_full[selected_features].reset_index(drop=True)

from sklearn.model_selection import train_test_split
_, X_test_unscaled, _, _ = train_test_split(X_full, df_full['hse defect level (relative to VBM)_[eV]'], test_size=0.2, random_state=42)

# -----------------------------------
# 2. Define Smarter Physics-Informed Loss
# -----------------------------------
def improved_physics_informed_loss(y_true, y_pred, bandgap_batch):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    # Only penalize when there is violation
    penalty_lower = tf.reduce_sum(tf.maximum(0.0, -y_pred))
    penalty_upper = tf.reduce_sum(tf.maximum(0.0, y_pred - bandgap_batch))

    physics_penalty = (penalty_lower + penalty_upper) / tf.cast(tf.shape(y_true)[0], tf.float32)

    total_loss = mse + 0.42 * physics_penalty  # Penalty weight = 0.42
    return total_loss

# -----------------------------------
# 3. Build Optimized Physics-Informed MLP
# -----------------------------------
input_dim = X_train_scaled.shape[1]

model_optimized = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')
])

optimizer = Adam(learning_rate=0.001)
model_optimized.compile(optimizer=optimizer, loss='mse')  # Placeholder loss

# -----------------------------------
# 4. Custom Training Loop with Smarter Loss
# -----------------------------------
batch_size = 32
epochs = 150
train_size = X_train_scaled.shape[0]
steps_per_epoch = train_size // batch_size

train_losses = []

for epoch in range(epochs):
    epoch_losses = []

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
            y_pred = model_optimized(X_batch, training=True)
            loss = improved_physics_informed_loss(
                tf.convert_to_tensor(y_batch.reshape(-1, 1), dtype=tf.float32),
                y_pred,
                bandgap_batch
            )

        grads = tape.gradient(loss, model_optimized.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_optimized.trainable_variables))

        epoch_losses.append(loss.numpy())

    avg_epoch_loss = np.mean(epoch_losses)
    train_losses.append(avg_epoch_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

# -----------------------------------
# 5. Evaluate Optimized Physics-Informed Model
# -----------------------------------
y_pred_optimized = model_optimized.predict(X_test_scaled).flatten()

mse_optimized = mean_squared_error(y_test, y_pred_optimized)
mae_optimized = mean_absolute_error(y_test, y_pred_optimized)
r2_optimized = r2_score(y_test, y_pred_optimized)

print("\nOptimized Physics-Informed MLP Performance:")
print(f"Test MSE: {mse_optimized:.4f}")
print(f"Test MAE: {mae_optimized:.4f}")
print(f"Test R2 Score: {r2_optimized:.4f}")

# -----------------------------------
# 6. Plot Loss Curve
# -----------------------------------
plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Train Loss (Optimized Physics-Informed)')
plt.title('Optimized Physics-Informed MLP Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.show()

# -----------------------------------
# 7. True vs Predicted Plot
# -----------------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_optimized, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True HSE Defect Level (eV)')
plt.ylabel('Predicted HSE Defect Level (eV)')
plt.title('True vs Predicted Defect Levels (Optimized Physics-Informed MLP)')
plt.grid()
plt.show()

# -----------------------------------
# 8. Check Physics Violations
# -----------------------------------
# Bandgap values for test set
bandgap_test = X_test_unscaled['host bandgap_[eV]'].values

# How many predictions violate 0 < pred < Eg
violations = np.sum((y_pred_optimized < 0) | (y_pred_optimized > bandgap_test))
violation_percentage = (violations / len(y_pred_optimized)) * 100

print(f"\n Physics Violations in Test Set: {violations} out of {len(y_pred_optimized)}")
print(f" Violation Rate: {violation_percentage:.2f}%")


# -----------------------------------
# 1. Imports
# -----------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load prepared data
X_train_scaled = np.load('/content/X_train_scaled.npy')
X_test_scaled = np.load('/content/X_test_scaled.npy')
y_train = np.load('/content/y_train.npy')
y_test = np.load('/content/y_test.npy')

# Load full (unscaled) data to get bandgap
import pandas as pd
df_full = pd.read_csv('/content/Dataset_semiconductor_defectlevels_v1.csv')
df_full = df_full.dropna(subset=['hse defect level (relative to VBM)_[eV]'])

selected_features = [
    'pbe defect level (relative to VBM)_[eV]',
    'host bandgap_[eV]',
    'host lattice constant_[Ang.]',
    'host_epsilon',
    'charge_from',
    'charge_to',
    'is_interstitial',
    'is_a_latt',
    'is_b_latt'
]

X_full = df_full[selected_features].reset_index(drop=True)

from sklearn.model_selection import train_test_split
_, X_test_unscaled, _, _ = train_test_split(X_full, df_full['hse defect level (relative to VBM)_[eV]'], test_size=0.2, random_state=42)

# -----------------------------------
# 2. Define Smarter Physics-Informed Loss
# -----------------------------------
def improved_physics_informed_loss(y_true, y_pred, bandgap_batch):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    # Only penalize when there is violation
    penalty_lower = tf.reduce_sum(tf.maximum(0.0, -y_pred))
    penalty_upper = tf.reduce_sum(tf.maximum(0.0, y_pred - bandgap_batch))

    physics_penalty = (penalty_lower + penalty_upper) / tf.cast(tf.shape(y_true)[0], tf.float32)

    total_loss = mse + 0.32 * physics_penalty  # Penalty weight = 0.32
    return total_loss

# -----------------------------------
# 3. Build Optimized Physics-Informed MLP
# -----------------------------------
input_dim = X_train_scaled.shape[1]

model_optimized = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')
])

optimizer = Adam(learning_rate=0.001)
model_optimized.compile(optimizer=optimizer, loss='mse')  # Placeholder loss

# -----------------------------------
# 4. Custom Training Loop with Smarter Loss
# -----------------------------------
batch_size = 32
epochs = 150
train_size = X_train_scaled.shape[0]
steps_per_epoch = train_size // batch_size

train_losses = []

for epoch in range(epochs):
    epoch_losses = []

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
            y_pred = model_optimized(X_batch, training=True)
            loss = improved_physics_informed_loss(
                tf.convert_to_tensor(y_batch.reshape(-1, 1), dtype=tf.float32),
                y_pred,
                bandgap_batch
            )

        grads = tape.gradient(loss, model_optimized.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_optimized.trainable_variables))

        epoch_losses.append(loss.numpy())

    avg_epoch_loss = np.mean(epoch_losses)
    train_losses.append(avg_epoch_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

# -----------------------------------
# 5. Evaluate Optimized Physics-Informed Model
# -----------------------------------
y_pred_optimized = model_optimized.predict(X_test_scaled).flatten()

mse_optimized = mean_squared_error(y_test, y_pred_optimized)
mae_optimized = mean_absolute_error(y_test, y_pred_optimized)
r2_optimized = r2_score(y_test, y_pred_optimized)

print("\nOptimized Physics-Informed MLP Performance:")
print(f"Test MSE: {mse_optimized:.4f}")
print(f"Test MAE: {mae_optimized:.4f}")
print(f"Test R2 Score: {r2_optimized:.4f}")

# -----------------------------------
# 6. Plot Loss Curve
# -----------------------------------
plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Train Loss (Optimized Physics-Informed)')
plt.title('Optimized Physics-Informed MLP Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.show()

# -----------------------------------
# 7. True vs Predicted Plot
# -----------------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_optimized, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True HSE Defect Level (eV)')
plt.ylabel('Predicted HSE Defect Level (eV)')
plt.title('True vs Predicted Defect Levels (Optimized Physics-Informed MLP)')
plt.grid()
plt.show()

# -----------------------------------
# 8. Check Physics Violations
# -----------------------------------
# Bandgap values for test set
bandgap_test = X_test_unscaled['host bandgap_[eV]'].values

# How many predictions violate 0 < pred < Eg
violations = np.sum((y_pred_optimized < 0) | (y_pred_optimized > bandgap_test))
violation_percentage = (violations / len(y_pred_optimized)) * 100

print(f"\n Physics Violations in Test Set: {violations} out of {len(y_pred_optimized)}")
print(f"Violation Rate: {violation_percentage:.2f}%")


# -----------------------------------
# 1. Imports
# -----------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load prepared data
X_train_scaled = np.load('/content/X_train_scaled.npy')
X_test_scaled = np.load('/content/X_test_scaled.npy')
y_train = np.load('/content/y_train.npy')
y_test = np.load('/content/y_test.npy')

# Load full (unscaled) data to get bandgap
import pandas as pd
df_full = pd.read_csv('/content/Dataset_semiconductor_defectlevels_v1.csv')
df_full = df_full.dropna(subset=['hse defect level (relative to VBM)_[eV]'])

selected_features = [
    'pbe defect level (relative to VBM)_[eV]',
    'host bandgap_[eV]',
    'host lattice constant_[Ang.]',
    'host_epsilon',
    'charge_from',
    'charge_to',
    'is_interstitial',
    'is_a_latt',
    'is_b_latt'
]

X_full = df_full[selected_features].reset_index(drop=True)

from sklearn.model_selection import train_test_split
_, X_test_unscaled, _, _ = train_test_split(X_full, df_full['hse defect level (relative to VBM)_[eV]'], test_size=0.2, random_state=42)

# -----------------------------------
# 2. Define Smarter Physics-Informed Loss
# -----------------------------------
def improved_physics_informed_loss(y_true, y_pred, bandgap_batch):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    # Only penalize when there is violation
    penalty_lower = tf.reduce_sum(tf.maximum(0.0, -y_pred))
    penalty_upper = tf.reduce_sum(tf.maximum(0.0, y_pred - bandgap_batch))

    physics_penalty = (penalty_lower + penalty_upper) / tf.cast(tf.shape(y_true)[0], tf.float32)

    total_loss = mse + 0.12 * physics_penalty  # Penalty weight = 0.12
    return total_loss

# -----------------------------------
# 3. Build Optimized Physics-Informed MLP
# -----------------------------------
input_dim = X_train_scaled.shape[1]

model_optimized = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')
])

optimizer = Adam(learning_rate=0.001)
model_optimized.compile(optimizer=optimizer, loss='mse')  # Placeholder loss

# -----------------------------------
# 4. Custom Training Loop with Smarter Loss
# -----------------------------------
batch_size = 32
epochs = 170
train_size = X_train_scaled.shape[0]
steps_per_epoch = train_size // batch_size

train_losses = []

for epoch in range(epochs):
    epoch_losses = []

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
            y_pred = model_optimized(X_batch, training=True)
            loss = improved_physics_informed_loss(
                tf.convert_to_tensor(y_batch.reshape(-1, 1), dtype=tf.float32),
                y_pred,
                bandgap_batch
            )

        grads = tape.gradient(loss, model_optimized.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_optimized.trainable_variables))

        epoch_losses.append(loss.numpy())

    avg_epoch_loss = np.mean(epoch_losses)
    train_losses.append(avg_epoch_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

# -----------------------------------
# 5. Evaluate Optimized Physics-Informed Model
# -----------------------------------
y_pred_optimized = model_optimized.predict(X_test_scaled).flatten()

mse_optimized = mean_squared_error(y_test, y_pred_optimized)
mae_optimized = mean_absolute_error(y_test, y_pred_optimized)
r2_optimized = r2_score(y_test, y_pred_optimized)

print("\nOptimized Physics-Informed MLP Performance:")
print(f"Test MSE: {mse_optimized:.4f}")
print(f"Test MAE: {mae_optimized:.4f}")
print(f"Test R2 Score: {r2_optimized:.4f}")

# -----------------------------------
# 6. Plot Loss Curve
# -----------------------------------
plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Train Loss (Optimized Physics-Informed)')
plt.title('Optimized Physics-Informed MLP Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.show()

# -----------------------------------
# 7. True vs Predicted Plot
# -----------------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_optimized, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True HSE Defect Level (eV)')
plt.ylabel('Predicted HSE Defect Level (eV)')
plt.title('True vs Predicted Defect Levels (Optimized Physics-Informed MLP)')
plt.grid()
plt.show()

# -----------------------------------
# 8. Check Physics Violations
# -----------------------------------
# Bandgap values for test set
bandgap_test = X_test_unscaled['host bandgap_[eV]'].values

# How many predictions violate 0 < pred < Eg
violations = np.sum((y_pred_optimized < 0) | (y_pred_optimized > bandgap_test))
violation_percentage = (violations / len(y_pred_optimized)) * 100

print(f"\n Physics Violations in Test Set: {violations} out of {len(y_pred_optimized)}")
print(f" Violation Rate: {violation_percentage:.2f}%")


# -----------------------------------
# 1. Imports
# -----------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load prepared data
X_train_scaled = np.load('/content/X_train_scaled.npy')
X_test_scaled = np.load('/content/X_test_scaled.npy')
y_train = np.load('/content/y_train.npy')
y_test = np.load('/content/y_test.npy')

# Load full (unscaled) data to get bandgap
import pandas as pd
df_full = pd.read_csv('/content/Dataset_semiconductor_defectlevels_v1.csv')
df_full = df_full.dropna(subset=['hse defect level (relative to VBM)_[eV]'])

selected_features = [
    'pbe defect level (relative to VBM)_[eV]',
    'host bandgap_[eV]',
    'host lattice constant_[Ang.]',
    'host_epsilon',
    'charge_from',
    'charge_to',
    'is_interstitial',
    'is_a_latt',
    'is_b_latt'
]

X_full = df_full[selected_features].reset_index(drop=True)

from sklearn.model_selection import train_test_split
_, X_test_unscaled, _, _ = train_test_split(X_full, df_full['hse defect level (relative to VBM)_[eV]'], test_size=0.2, random_state=42)

# -----------------------------------
# 2. Define Smarter Physics-Informed Loss
# -----------------------------------
def improved_physics_informed_loss(y_true, y_pred, bandgap_batch):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    # Only penalize when there is violation
    penalty_lower = tf.reduce_sum(tf.maximum(0.0, -y_pred))
    penalty_upper = tf.reduce_sum(tf.maximum(0.0, y_pred - bandgap_batch))

    physics_penalty = (penalty_lower + penalty_upper) / tf.cast(tf.shape(y_true)[0], tf.float32)

    total_loss = mse + 0.22 * physics_penalty  # Penalty weight = 0.22
    return total_loss

# -----------------------------------
# 3. Build Optimized Physics-Informed MLP
# -----------------------------------
input_dim = X_train_scaled.shape[1]

model_optimized = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')
])

optimizer = Adam(learning_rate=0.001)
model_optimized.compile(optimizer=optimizer, loss='mse')  # Placeholder loss

# -----------------------------------
# 4. Custom Training Loop with Smarter Loss
# -----------------------------------
batch_size = 32
epochs = 150
train_size = X_train_scaled.shape[0]
steps_per_epoch = train_size // batch_size

train_losses = []

for epoch in range(epochs):
    epoch_losses = []

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
            y_pred = model_optimized(X_batch, training=True)
            loss = improved_physics_informed_loss(
                tf.convert_to_tensor(y_batch.reshape(-1, 1), dtype=tf.float32),
                y_pred,
                bandgap_batch
            )

        grads = tape.gradient(loss, model_optimized.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_optimized.trainable_variables))

        epoch_losses.append(loss.numpy())

    avg_epoch_loss = np.mean(epoch_losses)
    train_losses.append(avg_epoch_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

# -----------------------------------
# 5. Evaluate Optimized Physics-Informed Model
# -----------------------------------
y_pred_optimized = model_optimized.predict(X_test_scaled).flatten()

mse_optimized = mean_squared_error(y_test, y_pred_optimized)
mae_optimized = mean_absolute_error(y_test, y_pred_optimized)
r2_optimized = r2_score(y_test, y_pred_optimized)

print("\n📊 Optimized Physics-Informed MLP Performance:")
print(f"Test MSE: {mse_optimized:.4f}")
print(f"Test MAE: {mae_optimized:.4f}")
print(f"Test R2 Score: {r2_optimized:.4f}")

# -----------------------------------
# 6. Plot Loss Curve
# -----------------------------------
plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Train Loss (Optimized Physics-Informed)')
plt.title('Optimized Physics-Informed MLP Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.show()

# -----------------------------------
# 7. True vs Predicted Plot
# -----------------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_optimized, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True HSE Defect Level (eV)')
plt.ylabel('Predicted HSE Defect Level (eV)')
plt.title('True vs Predicted Defect Levels (Optimized Physics-Informed MLP)')
plt.grid()
plt.show()

# -----------------------------------
# 8. Check Physics Violations
# -----------------------------------
# Bandgap values for test set
bandgap_test = X_test_unscaled['host bandgap_[eV]'].values

# How many predictions violate 0 < pred < Eg
violations = np.sum((y_pred_optimized < 0) | (y_pred_optimized > bandgap_test))
violation_percentage = (violations / len(y_pred_optimized)) * 100

print(f"\n Physics Violations in Test Set: {violations} out of {len(y_pred_optimized)}")
print(f" Violation Rate: {violation_percentage:.2f}%")


# -----------------------------------
# 1. Imports
# -----------------------------------
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Load prepared data
X_train_scaled = np.load('/content/X_train_scaled.npy')
X_test_scaled = np.load('/content/X_test_scaled.npy')
y_train = np.load('/content/y_train.npy')
y_test = np.load('/content/y_test.npy')

# Load unscaled data to get bandgap
df_full = pd.read_csv('/content/Dataset_semiconductor_defectlevels_v1.csv')
df_full = df_full.dropna(subset=['hse defect level (relative to VBM)_[eV]'])

selected_features = [
    'pbe defect level (relative to VBM)_[eV]',
    'host bandgap_[eV]',
    'host lattice constant_[Ang.]',
    'host_epsilon',
    'charge_from',
    'charge_to',
    'is_interstitial',
    'is_a_latt',
    'is_b_latt'
]

X_full = df_full[selected_features].reset_index(drop=True)
_, X_test_unscaled, _, _ = train_test_split(X_full, df_full['hse defect level (relative to VBM)_[eV]'], test_size=0.2, random_state=42)

# -----------------------------------
# 2. Define Custom Loss with Bandgap Scaling
# -----------------------------------
def custom_scaled_loss(bandgap_batch):
    def loss_fn(y_true, y_pred_sigmoid):
        y_pred_scaled = y_pred_sigmoid * bandgap_batch  # Scale output to [0, Eg]
        mse = tf.reduce_mean(tf.square(y_true - y_pred_scaled))
        return mse
    return loss_fn

# -----------------------------------
# 3. Build Sigmoid-Scaled Output MLP
# -----------------------------------
input_dim = X_train_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
x = Dense(128, activation='relu')(input_layer)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
output_layer = Dense(1, activation='sigmoid')(x)  # Sigmoid Output

model_sigmoid = Model(inputs=input_layer, outputs=output_layer)
optimizer = Adam(learning_rate=0.001)

# -----------------------------------
# 4. Custom Training Loop
# -----------------------------------
batch_size = 32
epochs = 150
train_size = X_train_scaled.shape[0]
steps_per_epoch = train_size // batch_size

train_losses = []

for epoch in range(epochs):
    epoch_losses = []

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
            y_pred_sigmoid = model_sigmoid(X_batch, training=True)
            loss = custom_scaled_loss(bandgap_batch)(
                tf.convert_to_tensor(y_batch.reshape(-1, 1), dtype=tf.float32),
                y_pred_sigmoid
            )

        grads = tape.gradient(loss, model_sigmoid.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_sigmoid.trainable_variables))

        epoch_losses.append(loss.numpy())

    avg_epoch_loss = np.mean(epoch_losses)
    train_losses.append(avg_epoch_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

# -----------------------------------
# 5. Evaluate the Model
# -----------------------------------
# Predict
y_pred_sigmoid_test = model_sigmoid.predict(X_test_scaled).flatten()

# Scale by bandgap
bandgap_test = X_test_unscaled['host bandgap_[eV]'].values
y_pred_scaled = y_pred_sigmoid_test * bandgap_test

# Evaluate
mse_final = mean_squared_error(y_test, y_pred_scaled)
mae_final = mean_absolute_error(y_test, y_pred_scaled)
r2_final = r2_score(y_test, y_pred_scaled)

print("\n📊 Sigmoid-Scaled Output MLP Performance:")
print(f"Test MSE: {mse_final:.4f}")
print(f"Test MAE: {mae_final:.4f}")
print(f"Test R2 Score: {r2_final:.4f}")

# -----------------------------------
# 6. Plot Training Loss
# -----------------------------------
plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Train Loss (Sigmoid-Scaled)')
plt.title('Training Loss: Sigmoid-Scaled Output MLP')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.show()

# -----------------------------------
# 7. True vs Predicted Plot
# -----------------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_scaled, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True HSE Defect Level (eV)')
plt.ylabel('Predicted HSE Defect Level (eV)')
plt.title('True vs Predicted Defect Levels (Sigmoid-Scaled Output MLP)')
plt.grid()
plt.show()

# -----------------------------------
# 8. Physics Violation Check
# -----------------------------------
violations = np.sum((y_pred_scaled < 0) | (y_pred_scaled > bandgap_test))
violation_percentage = (violations / len(y_pred_scaled)) * 100

print(f"\n Physics Violations in Test Set: {violations} out of {len(y_pred_scaled)}")
print(f" Violation Rate: {violation_percentage:.2f})
