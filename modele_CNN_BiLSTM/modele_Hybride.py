import torch
import torch.nn as nn

class CNN_BiLSTM_Model(nn.Module):
    def __init__(self, cnn_model, hidden_dim, num_layers, output_dim, target_size=(32, 32)):
        super(CNN_BiLSTM_Model, self).__init__()
        self.cnn = cnn_model
        
        # Calculer input_dim en fonction de la sortie du CNN et de target_size
        num_features = 3  # Le nombre de sorties de votre SimpleCNN (par exemple, 3)
        input_dim = num_features * target_size[0] * target_size[1]  # Dimension après aplatissement
        
        # Initialiser le BiLSTM avec input_dim
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        
        # Ajuster la couche de sortie pour prédire par pixel
        self.fc = nn.Linear(hidden_dim * 2, output_dim * target_size[0] * target_size[1])  # x2 pour bidirectionnel
        self.output_dim = output_dim
        self.target_size = target_size
    
    def forward(self, x):
        batch_size, seq_length, channels, height, width = x.size()
        
        # Initialisation d'une liste pour stocker les caractéristiques CNN de chaque pas de temps
        cnn_features = []
        
        # Passer chaque pas de temps à travers le CNN
        for t in range(seq_length):
            # Appliquer le CNN sur chaque image individuelle de la séquence
            cnn_out = self.cnn(x[:, t, :, :, :])  # shape (batch_size, num_features, target_size[0], target_size[1])
            cnn_out = cnn_out.view(batch_size, -1)  # Aplatir pour avoir (batch_size, input_dim)
            cnn_features.append(cnn_out)
        
        # Empiler les caractéristiques CNN le long de la dimension temporelle pour former une séquence
        cnn_features = torch.stack(cnn_features, dim=1)  # shape (batch_size, seq_length, input_dim)
        
        # Passer les caractéristiques dans le BiLSTM
        bilstm_out, _ = self.bilstm(cnn_features)
        
        # Prendre la sortie du dernier pas de temps pour la classification
        final_out = bilstm_out[:, -1, :]  # Shape: (batch_size, hidden_dim * 2)
        
        # Utiliser la couche fully connected pour la prédiction par pixel
        output = self.fc(final_out)
        output = output.view(batch_size, self.output_dim, self.target_size[0], self.target_size[1])  # (batch_size, num_classes, height, width)
        
        return output