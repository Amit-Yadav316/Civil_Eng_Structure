# Structural Vibration Anomaly Detection
![image](https://github.com/user-attachments/assets/6749bb63-d07b-43c0-9be4-cfa8c68f9cbe)

# Streamlit UI
![Screenshot 2025-07-03 012125](https://github.com/user-attachments/assets/1bafc81b-69cb-4a21-b25d-202fa799febd)

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
