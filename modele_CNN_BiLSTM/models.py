import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(16, 64, kernel_size=3, padding=1)  # Augmentation des canaux de sortie
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Augmentation des canaux de sortie
        self.conv3 = nn.Conv2d(128, 3, kernel_size=1)  # Couche finale pour 3 classes

        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm1 = nn.BatchNorm2d(64)  # Ajout de BatchNorm
        self.batch_norm2 = nn.BatchNorm2d(128)  # Ajout de BatchNorm
        
    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))  # Convolution + BatchNorm + ReLU
        x = self.dropout(x)  # Dropout après la première convolution
        x = F.relu(self.batch_norm2(self.conv2(x)))  # Convolution + BatchNorm + ReLU
        x = self.dropout(x)  # Dropout après la deuxième convolution
        x = self.conv3(x)  # Couche finale
        return x