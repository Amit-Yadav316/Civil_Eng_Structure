import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score , confusion_matrix
from model import LSTM_CNN_Attn_Autoencoder
from data_utils import load_and_preprocess


X_healthy, X_faulty, _ = load_and_preprocess("simulated_structural_health_dataset.csv")

X_test = np.concatenate([X_healthy, X_faulty])
y_test = np.concatenate([
    np.zeros(len(X_healthy)),  
    np.ones(len(X_faulty))    
])


if isinstance(X_test, np.ndarray):
        X_test = torch.tensor(X_test, dtype=torch.float32)


input_dim = X_test.shape[-1]
model = LSTM_CNN_Attn_Autoencoder(input_dim=input_dim)
model.load_state_dict(torch.load("attention_model.pth"))
model.eval()

with torch.no_grad():
    reconstructed = model(X_test)
    errors = ((X_test - reconstructed) ** 2).mean(dim=(1, 2)).numpy()


from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, errors)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]



anomaly_indices = np.where(errors > best_thresh)[0]
y_pred = (errors > best_thresh).astype(int)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


plt.figure(figsize=(12, 5))
plt.plot(errors, label="Reconstruction Error (MSE)")
plt.axhline(y=best_thresh, color='r', linestyle='--', label=f"Threshold = {best_thresh:.4f}")
plt.scatter(anomaly_indices, errors[anomaly_indices], color='red', label="Anomalies", zorder=5)
plt.title("Anomaly Detection - Reconstruction Error")
plt.xlabel("Time Window Index")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


error_map = ((X_test - reconstructed) ** 2).mean(dim=1).numpy() 

plt.figure(figsize=(14, 6))
sns.heatmap(error_map.T, cmap="Reds", cbar=True, xticklabels=50)
plt.title("Sensor-wise Anomaly Heatmap")
plt.xlabel("Window Index")
plt.ylabel("Sensor Index")
plt.tight_layout()
plt.show()
