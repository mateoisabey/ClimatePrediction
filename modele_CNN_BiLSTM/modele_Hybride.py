import torch
import torch.nn as nn

class CNN_BiLSTM_Model(nn.Module):
    def __init__(self, cnn_model, hidden_dim, num_layers, output_dim, target_size=(32, 32)):
        super(CNN_BiLSTM_Model, self).__init__()
        self.cnn = cnn_model
        

        num_features = 3 
        input_dim = num_features * target_size[0] * target_size[1]  # Dimension après aplatissement
        
        # Initialiser le BiLSTM avec input_dim
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        
        # Ajuster la couche de sortie pour prédire par pixel
        self.fc = nn.Linear(hidden_dim * 2, output_dim * target_size[0] * target_size[1])
        self.output_dim = output_dim
        self.target_size = target_size
    
    def forward(self, x):
        batch_size, seq_length, channels, height, width = x.size()
        cnn_features = []
        
        # Passer chaque pas de temps à travers le CNN
        for t in range(seq_length):
            # Appliquer le CNN sur chaque image individuelle de la séquence
            cnn_out = self.cnn(x[:, t, :, :, :])
            cnn_out = cnn_out.view(batch_size, -1)
            cnn_features.append(cnn_out)
        
        # Empiler les caractéristiques CNN le long de la dimension temporelle pour former une séquence
        cnn_features = torch.stack(cnn_features, dim=1)
        
        # Passer les caractéristiques dans le BiLSTM
        bilstm_out, _ = self.bilstm(cnn_features)
        
        # Prendre la sortie du dernier pas de temps pour la classification
        final_out = bilstm_out[:, -1, :]
        
        # Utiliser la couche fully connected pour la prédiction par pixel
        output = self.fc(final_out)
        output = output.view(batch_size, self.output_dim, self.target_size[0], self.target_size[1])
        
        return output