import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Couche de sortie pour la classification
    
    def forward(self, x):
        # LSTM expects inputs of shape (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)
        # We only take the output of the last time step for classification
        final_out = lstm_out[:, -1, :]
        output = self.fc(final_out)
        return output