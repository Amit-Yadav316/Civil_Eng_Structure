# Structural Vibration Anomaly Detection

This project uses an LSTM_CNN_Attn_Autoencoder to detect anomalies in structural health monitoring (SHM) sensor data.

## Project Structure
- `model.py`: Defines the LSTM Autoencoder
- `data_utils.py`: Data loading and preprocessing
- `train.py`: Trains the model
- `evaluate.py`: Loads the model and plots anomaly scores

## Usage
1. Place `structural_health_dataset.csv` from Kaggle in the `data/` folder.
2. Install dependencies: `pip install -r requirements.txt`
3. Train: `python train.py`
4. Evaluate: `python evaluate.py`
