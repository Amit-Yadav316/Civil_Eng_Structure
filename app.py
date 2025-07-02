import streamlit as st
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import LSTM_CNN_Attn_Autoencoder
from data_utils import load_and_preprocess
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

st.title("Structural Health Monitoring - Anomaly Detection")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file:
    with st.spinner("Processing data..."):
        X_healthy, X_faulty, _ = load_and_preprocess(uploaded_file)

        X_test = np.concatenate([X_healthy, X_faulty])
        y_test = np.concatenate([
            np.zeros(len(X_healthy)),
            np.ones(len(X_faulty))
        ])
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

        model = LSTM_CNN_Attn_Autoencoder(input_dim=X_test.shape[-1])
        model.load_state_dict(torch.load("attention_model.pth", map_location="cpu"))
        model.eval()

        with torch.no_grad():
            recon = model(X_test_tensor)
            errors = ((X_test_tensor - recon) ** 2).mean(dim=(1, 2)).numpy()

        precision, recall, thresholds = precision_recall_curve(y_test, errors)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_idx]
        y_pred = (errors > best_thresh).astype(int)

        st.success(f"Best Threshold: {best_thresh:.4f} — F1 Score: {f1_scores[best_idx]:.4f}")

        
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred, digits=4))

        
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

       
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(errors, bins=100, alpha=0.7)
        ax_hist.axvline(best_thresh, color='red', linestyle='--', label=f"Threshold = {best_thresh:.4f}")
        ax_hist.set_title("Reconstruction Errors with Threshold")
        ax_hist.set_xlabel("Reconstruction Error (MSE)")
        ax_hist.set_ylabel("Frequency")
        ax_hist.legend()
        st.pyplot(fig_hist)