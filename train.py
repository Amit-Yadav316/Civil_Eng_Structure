import torch
import numpy as np
import torch.nn as nn
from model import LSTM_CNN_Attn_Autoencoder
from data_utils import load_and_preprocess

X_healthy, X_faulty, _ = load_and_preprocess("simulated_structural_health_dataset.csv")
faulty_20_percent = int(0.2 * len(X_faulty))
X_faulty_subset = X_faulty[:faulty_20_percent]
X_train = np.concatenate([X_healthy, X_faulty_subset])
np.random.seed(42)
np.random.shuffle(X_train)
input_dim = X_train.shape[-1]


if isinstance(X_train, np.ndarray):
        X_train = torch.tensor(X_train, dtype=torch.float32)

model = LSTM_CNN_Attn_Autoencoder(input_dim=X_train.shape[-1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(25):
    model.train()
    output = model(X_train)
    loss = criterion(output, X_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

torch.save(model.state_dict(), "attention_model.pth")
