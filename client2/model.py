# model.py
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2, bidirectional=True):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        # Simple attention over time steps
        self.attn = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_out_dim // 2, 1)
        )

        # MLP prediction head
        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        seq_out, _ = self.lstm(x)  # (batch, seq_len, lstm_out_dim)

        # Attention weights over time
        attn_scores = self.attn(seq_out).squeeze(-1)  # (batch, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=1)

        # Context vector as weighted sum of time steps
        context = torch.sum(seq_out * attn_weights.unsqueeze(-1), dim=1)  # (batch, lstm_out_dim)

        out = self.head(context)
        return out.squeeze(-1)
