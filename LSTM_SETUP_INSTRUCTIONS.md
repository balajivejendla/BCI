# LSTM Model Setup Instructions

## Issue Encountered

The LSTM model script (`run_lstm.py`) and notebook (`lstm_classification.ipynb`) have been created, but TensorFlow installation failed due to Windows Long Path limitations.

## Solutions

### Option 1: Enable Windows Long Path Support (Recommended)

1. **Run PowerShell as Administrator**
2. Execute this command:
   ```powershell
   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
   ```
3. **Restart your computer**
4. Then install TensorFlow:
   ```powershell
   pip install tensorflow
   ```

### Option 2: Use a Virtual Environment with Shorter Path

1. Create a virtual environment in a shorter path (e.g., `C:\venv`):
   ```powershell
   python -m venv C:\venv\bci
   C:\venv\bci\Scripts\Activate.ps1
   pip install tensorflow scikit-learn matplotlib
   ```

### Option 3: Use Existing Environment

If you already have TensorFlow/Keras installed (as your `classification.ipynb` uses Keras), you can:

1. **Run the notebook directly in Jupyter**:
   - Open `lstm_classification.ipynb` in Jupyter Notebook/Lab
   - Run all cells sequentially

2. **Or run the Python script**:
   ```powershell
   cd eeg_stress_detection
   python run_lstm.py
   ```

## Files Created

1. **`lstm_classification.ipynb`** - Jupyter notebook with LSTM model
2. **`run_lstm.py`** - Python script version (can be run directly)

## What the LSTM Model Does

- **Input**: Raw EEG time-series data (128 timesteps × 32 channels per sample)
- **Architecture**: 
  - LSTM Layer 1: 128 units
  - LSTM Layer 2: 64 units
  - Dense Layer: 32 units
  - Output: 2 classes (Low Stress / High Stress)
- **Features**: 
  - Captures temporal dependencies in EEG signals
  - Uses dropout for regularization
  - Early stopping to prevent overfitting

## Where Accuracy is Displayed

After running the model, accuracy will be shown in:

1. **Test Accuracy** - Overall accuracy on test set
2. **Classification Report** - Precision, Recall, F1-score for each class
3. **Confusion Matrix** - Detailed breakdown of predictions
4. **Training History** - Best validation accuracy during training

## Expected Output

The script will display:
```
============================================================
TEST SET RESULTS
============================================================
Test Loss: X.XXXX
Test Accuracy: X.XXXX (XX.XX%)
============================================================
```

Plus detailed classification metrics and confusion matrix.


