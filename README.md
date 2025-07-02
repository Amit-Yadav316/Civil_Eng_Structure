# Structural Vibration Anomaly Detection

This project uses an LSTM Autoencoder to detect anomalies in structural health monitoring (SHM) sensor data.

## Project Structure
- `model.ipynb`: Defines the LSTM Autoencoder
- `data_utils.ipynb`: Data loading and preprocessing
- `train.ipynb`: Trains the model
- `evaluate.ipynb`: Loads the model and plots anomaly scores

## Usage
1. Place `structural_health_dataset.csv` from Kaggle in the `data/` folder.
2. Install dependencies: `pip install -r requirements.txt`
3. Train: `python train.ipynb`
4. Evaluate: `python evaluate.ipynb`