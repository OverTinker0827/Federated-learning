# model.py
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        
        # h_n: (num_layers, batch_size, hidden_dim)
        last_hidden = h_n[-1]
        
        out = self.fc(last_hidden)
        return out.squeeze(-1)

