import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        scores = self.attn(x)  
        weights = torch.softmax(scores, dim=1)
        weighted = x * weights
        return weighted.sum(dim=1), weights

class LSTM_CNN_Attn_Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(hidden_dim*2, hidden_dim*2, kernel_size=3, padding=1)
        self.attn = Attention(hidden_dim*2)
        self.decoder = nn.LSTM(hidden_dim*2, input_dim, batch_first=True)

    def forward(self, x):
        enc_out, _ = self.encoder(x)
        cnn_in = enc_out.permute(0, 2, 1)               
        cnn_out = self.conv(cnn_in).permute(0, 2, 1)    
        attended_vec, _ = self.attn(cnn_out)
        attended=attended_vec.unsqueeze(1) 
        attended = attended.repeat(1, x.size(1), 1)    
        decoded, _ = self.decoder(attended)
        return decoded