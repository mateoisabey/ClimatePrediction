import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x.shape: (batch, channels, height, width)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten pour passer au RNN
        return x

class CNN_RNN_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CNN_RNN_Model, self).__init__()
        self.cnn = CNN()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, sequence_length, channels, height, width = x.size()
        cnn_features = []

        # Appliquer CNN sur chaque image dans la séquence
        for t in range(sequence_length):
            cnn_out = self.cnn(x[:, t, :, :, :])
            cnn_features.append(cnn_out)

        cnn_features = torch.stack(cnn_features, dim=1)  # Convertir en un tenseur pour le RNN
        rnn_out, _ = self.lstm(cnn_features)
        output = self.fc(rnn_out[:, -1, :])  # Utiliser la dernière sortie du RNN
        return output