import torch
import torch.nn as nn

class ClimateTransformer(nn.Module):
    def __init__(self, input_dim, num_labels, d_model=128, nhead=8, num_layers=4):
        super(ClimateTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, d_model))  # Max 1000 séquences
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, num_labels)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.embedding(x)  # Embed input features to d_model
        x = x + self.positional_encoding[:, :x.size(1), :]  # Add positional encoding
        x = self.encoder(x)  # Transformer encoder
        x = self.fc(x)  # Output layer for classification
        return x