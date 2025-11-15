# Model Explanation: MLP (Multilayer Perceptron) for EEG Stress Detection

## ⚠️ Important Note
This model is actually a **Multilayer Perceptron (MLP)**, not a CNN (Convolutional Neural Network). MLPs use fully connected Dense layers, while CNNs use convolutional layers for spatial feature extraction.

## 📊 What This Model Does

### 1. **Data Pipeline**
- **Input**: EEG signals from 32 channels, sampled at 128 Hz
- **Data Type**: ICA-filtered data (artifact-removed)
- **Test Type**: Arithmetic stress test
- **Feature Extraction**: Uses **Fractal Features** (Higuchi and Katz Fractal Dimensions)
  - These features capture the complexity and irregularity of EEG signals
  - 2 features per channel × 32 channels = 64 features per sample

### 2. **Model Architecture (MLP)**
The model uses a **Sequential Neural Network** with:
- **Input Layer**: Takes flattened feature vector (64 features)
- **Hidden Layers**: 2-6 fully connected (Dense) layers (hyperparameter tuned)
  - Each layer has 32-1024 neurons (tuned)
  - Activation: ReLU or Sigmoid (tuned)
- **Output Layer**: 2 neurons with Softmax activation (binary classification)
  - Class 0: Low stress (False)
  - Class 1: High stress (True)

### 3. **Hyperparameter Tuning**
- Uses **Keras Tuner (RandomSearch)** to find best architecture
- Tests 15 different configurations
- Optimizes for **validation accuracy**
- Best validation accuracy found: **55.42%**

### 4. **Training Process**
- **Optimizer**: Adam (learning rate: 0.01, 0.001, or 0.0001 - tuned)
- **Loss Function**: Binary Crossentropy
- **Epochs**: 50 per trial
- **Data Split**:
  - Training: 60%
  - Validation: 20%
  - Test: 20%

## 📈 Where to See Accuracy

### **Accuracy is displayed in TWO places:**

#### 1. **During Hyperparameter Tuning** (Cell 22 output)
```
Best val_accuracy So Far: 0.5541666746139526
```
This shows the **validation accuracy** during model selection.

#### 2. **Final Test Results** (Cell 25 output)
```
              precision    recall  f1-score   support

           0       0.52      0.90      0.66       311
           1       0.52      0.11      0.19       289

    accuracy                           0.52       600
```
- **Overall Accuracy: 52%** (520 out of 600 test samples correctly classified)
- **Confusion Matrix**:
  - True Negatives (Class 0 correctly predicted): 281
  - False Positives: 30
  - False Negatives: 256
  - True Positives (Class 1 correctly predicted): 33

## 🔍 Model Performance Analysis

### Current Results:
- **Test Accuracy: 52%** (slightly better than random chance of 50%)
- **Issue**: The model is heavily biased toward predicting Class 0 (low stress)
  - Class 0 recall: 90% (good at identifying low stress)
  - Class 1 recall: 11% (very poor at identifying high stress)

### Comparison with Other Models:
- **k-NN**: 56% accuracy
- **SVM**: 58% accuracy
- **MLP**: 52% accuracy ⚠️ (lowest performance)

## 💡 Why the Model Might Be Underperforming

1. **Class Imbalance**: The model struggles with Class 1 (high stress)
2. **Feature Selection**: Fractal features alone may not be sufficient
3. **Model Complexity**: May need more layers or different architecture
4. **Data Quality**: May need more training data or better preprocessing

## 🚀 Potential Improvements

1. **Try Different Features**: 
   - Frequency band features
   - Entropy features
   - Time-series features
   - Or combine multiple feature types

2. **Address Class Imbalance**:
   - Use class weights
   - Oversample minority class
   - Use different evaluation metrics

3. **Consider CNN Architecture**:
   - If you want a true CNN, you'd need to:
     - Reshape data to preserve spatial/temporal structure
     - Use Conv1D layers for temporal patterns
     - Use Conv2D if treating channels as spatial dimensions


