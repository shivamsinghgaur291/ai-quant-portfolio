# models.py
import torch
import torch.nn as nn

class MultiAssetLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, num_layers=2, output_dim=0, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        # take last timestep
        last = out[:, -1, :]
        return self.fc(last)  # (batch, output_dim)

class MultiAssetTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=64, nhead=8, num_layers=2, dim_feedforward=128, output_dim=0, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, output_dim)
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)  # -> (batch, seq_len, model_dim)
        # Transformer expects (seq_len, batch, model_dim)
        x = x.permute(1,0,2)
        out = self.transformer(x)  # (seq_len, batch, model_dim)
        last = out[-1]  # (batch, model_dim)
        return self.fc(last)
