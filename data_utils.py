import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

def load_and_preprocess(path, window=50):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df.sort_values(by='status', inplace=True)

    features = df.drop(columns=['status']).values 
    labels = df['status'].values

    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    healthy = features[labels == 'healthy']
    faulty = features[labels == 'faulty']

    def create_windows(data):
        return np.array([data[i:i+window] for i in range(len(data) - window)])

    X_healthy = create_windows(healthy)
    X_faulty = create_windows(faulty)


    X_healthy = torch.tensor(X_healthy, dtype=torch.float32)  
    X_faulty = torch.tensor(X_faulty, dtype=torch.float32)

    return X_healthy, X_faulty, scaler