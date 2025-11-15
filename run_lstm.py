"""
LSTM Model for EEG Stress Detection
This script trains an LSTM model and displays accuracy results.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from dataset import load_dataset, load_labels, split_data, format_labels
import variables as v
import matplotlib.pyplot as plt

print("=" * 60)
print("LSTM MODEL FOR EEG STRESS DETECTION")
print("=" * 60)

# Configuration
data_type = "ica_filtered"
test_type = "Arithmetic"
print(f"\nData type: {data_type}")
print(f"Test type: {test_type}")

# Load dataset
print("\n[1/8] Loading dataset...")
dataset_ = load_dataset(data_type=data_type, test_type=test_type)
dataset = split_data(dataset_, v.SFREQ)
print(f"Dataset shape after splitting: {dataset.shape}")
print(f"Shape breakdown: (trials={dataset.shape[0]}, epochs={dataset.shape[1]}, channels={dataset.shape[2]}, timepoints={dataset.shape[3]})")

# Load labels
print("\n[2/8] Loading labels...")
label_ = load_labels()
label = format_labels(label_, test_type=test_type, epochs=dataset.shape[1])
print(f"Label shape: {label.shape}")
print(f"Label distribution: {np.bincount(label.astype(int))}")

# Reshape data for LSTM
print("\n[3/8] Reshaping data for LSTM...")
n_trials, n_epochs, n_channels, n_timepoints = dataset.shape
X = dataset.reshape(n_trials * n_epochs, n_timepoints, n_channels)
y = label.reshape(-1)
print(f"Reshaped X shape: {X.shape} (samples, timesteps, features)")
print(f"Reshaped y shape: {y.shape}")
print(f"Number of samples: {X.shape[0]}")
print(f"Timesteps per sample: {X.shape[1]}")
print(f"Features per timestep: {X.shape[2]}")

# Split data
print("\n[4/8] Splitting data into train/validation/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Normalize data
print("\n[5/8] Normalizing data...")
n_samples_train, n_timesteps, n_features = X_train.shape
X_train_reshaped = X_train.reshape(-1, n_features)
X_val_reshaped = X_val.reshape(-1, n_features)
X_test_reshaped = X_test.reshape(-1, n_features)

scaler = MinMaxScaler()
scaler.fit(X_train_reshaped)

X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)
X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
print(f"Scaled training data shape: {X_train_scaled.shape}")
print(f"Data range after scaling: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")

# Convert labels to categorical
y_train_cat = to_categorical(y_train, num_classes=v.N_CLASSES)
y_val_cat = to_categorical(y_val, num_classes=v.N_CLASSES)
y_test_cat = to_categorical(y_test, num_classes=v.N_CLASSES)
print(f"Categorical labels shape: {y_train_cat.shape}")

# Build LSTM model
print("\n[6/8] Building LSTM model...")
keras.backend.clear_session()

model = models.Sequential([
    Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
    LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
    LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(v.N_CLASSES, activation='softmax', name='output')
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel Architecture:")
model.summary()

# Train the model
print("\n[7/8] Training LSTM model...")
print("This may take several minutes...\n")

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    X_train_scaled, y_train_cat,
    batch_size=32,
    epochs=100,
    validation_data=(X_val_scaled, y_val_cat),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate model
print("\n[8/8] Evaluating model...")
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_cat, verbose=0)

print("\n" + "=" * 60)
print("TEST SET RESULTS")
print("=" * 60)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print("=" * 60)

# Make predictions
y_pred_proba = model.predict(X_test_scaled, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test_cat, axis=1)

# Classification report
print("\n" + "=" * 60)
print("DETAILED CLASSIFICATION REPORT")
print("=" * 60)
print(metrics.classification_report(y_true, y_pred, 
                                    target_names=['Low Stress', 'High Stress']))
print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
cm = metrics.confusion_matrix(y_true, y_pred)
print(cm)
print("\nConfusion Matrix Breakdown:")
print(f"True Negatives (Low Stress correctly predicted):  {cm[0][0]}")
print(f"False Positives (Low Stress predicted as High):  {cm[0][1]}")
print(f"False Negatives (High Stress predicted as Low):  {cm[1][0]}")
print(f"True Positives (High Stress correctly predicted): {cm[1][1]}")
print("=" * 60)

# Training history summary
best_val_acc = max(history.history['val_accuracy'])
best_val_epoch = np.argmax(history.history['val_accuracy']) + 1
print(f"\nBest Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%) at epoch {best_val_epoch}")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f} ({history.history['accuracy'][-1]*100:.2f}%)")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f} ({history.history['val_accuracy'][-1]*100:.2f}%)")

print("\n" + "=" * 60)
print("LSTM MODEL TRAINING COMPLETE!")
print("=" * 60)


