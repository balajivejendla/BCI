"""
Transformer Model for EEG Stress Detection
This script trains a Transformer model and displays accuracy results.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Dense, Dropout, Input, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from dataset import load_dataset, load_labels, split_data, format_labels
import variables as v
import matplotlib.pyplot as plt

print("=" * 60)
print("TRANSFORMER MODEL FOR EEG STRESS DETECTION")
print("=" * 60)

# Configuration - using filtered_data
data_type = "wt_filtered"
test_type = "Arithmetic"
print(f"\nData type: {data_type}")
print(f"Test type: {test_type}")

# Load dataset
print("\n[1/9] Loading dataset...")
dataset_ = load_dataset(data_type=data_type, test_type=test_type)
dataset = split_data(dataset_, v.SFREQ)
print(f"Dataset shape after splitting: {dataset.shape}")
print(f"Shape breakdown: (trials={dataset.shape[0]}, epochs={dataset.shape[1]}, channels={dataset.shape[2]}, timepoints={dataset.shape[3]})")

# Load labels
print("\n[2/9] Loading labels...")
label_ = load_labels()
label = format_labels(label_, test_type=test_type, epochs=dataset.shape[1])
print(f"Label shape: {label.shape}")
print(f"Label distribution: {np.bincount(label.astype(int))}")

# Reshape data for Transformer
print("\n[3/9] Reshaping data for Transformer...")
n_trials, n_epochs, n_channels, n_timepoints = dataset.shape
# Transformer expects: (samples, sequence_length, features)
# Each epoch becomes one sample with timepoints as sequence and channels as features
X = dataset.reshape(n_trials * n_epochs, n_timepoints, n_channels)
y = label.reshape(-1)
print(f"Reshaped X shape: {X.shape} (samples, sequence_length, features)")
print(f"Reshaped y shape: {y.shape}")
print(f"Number of samples: {X.shape[0]}")
print(f"Sequence length: {X.shape[1]}")
print(f"Features per timestep: {X.shape[2]}")

# Split data
print("\n[4/9] Splitting data into train/validation/test sets...")
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
print("\n[5/9] Normalizing data...")
n_samples_train, n_seq_len, n_features = X_train.shape
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

# Transformer Encoder Block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.3):
    """
    Creates a transformer encoder block.
    
    Args:
        inputs: Input tensor
        head_size: Size of attention head
        num_heads: Number of attention heads
        ff_dim: Feed-forward network dimension
        dropout: Dropout rate
    """
    # Multi-head self-attention
    attention_output = MultiHeadAttention(
        key_dim=head_size, 
        num_heads=num_heads, 
        dropout=dropout
    )(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    x = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    # Feed-forward network
    ffn_output = Dense(ff_dim, activation="relu")(x)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    ffn_output = Dropout(dropout)(ffn_output)
    outputs = LayerNormalization(epsilon=1e-6)(x + ffn_output)
    
    return outputs

# Build Transformer model
print("\n[6/9] Building Transformer model...")
keras.backend.clear_session()

# Model parameters
sequence_length = X_train_scaled.shape[1]
num_features = X_train_scaled.shape[2]
head_size = 64
num_heads = 4
ff_dim = 128
num_transformer_blocks = 2
mlp_units = [64, 32]
mlp_dropout = 0.4
dropout = 0.3

# Input layer
inputs = Input(shape=(sequence_length, num_features))

# Transformer blocks
x = inputs
for _ in range(num_transformer_blocks):
    x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

# Global average pooling
x = GlobalAveragePooling1D()(x)

# MLP head
for dim in mlp_units:
    x = Dense(dim, activation="relu")(x)
    x = Dropout(mlp_dropout)(x)

# Output layer
outputs = Dense(v.N_CLASSES, activation="softmax", name="output")(x)

model = models.Model(inputs, outputs)

# Compile model
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel Architecture:")
model.summary()

# Train the model
print("\n[7/9] Training Transformer model...")
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
print("\n[8/9] Evaluating model...")
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
print("TRANSFORMER MODEL TRAINING COMPLETE!")
print("=" * 60)

