import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import glob

# Fonction pour charger les fichiers .npy
def load_npy_data(npy_files, sequence_length=5):
    data = []
    labels = []
    
    for i in range(len(npy_files) - sequence_length + 1):
        sequence = []
        
        # Charger une séquence de fichiers consécutifs
        for j in range(sequence_length):
            sample = np.load(npy_files[i + j], allow_pickle=True).item()
            
            # Extraire les canaux de données spécifiques que vous voulez utiliser
            input_data = np.stack([sample['TMQ'], sample['U850'], sample['V850']])  # Remplacez par les canaux désirés
            sequence.append(input_data)
        
        # Ajouter la séquence et le label associé
        data.append(np.array(sequence))
        labels.append(sample['LABELS'])  # Assurez-vous que 'LABELS' est le nom correct
    
    return np.array(data), np.array(labels)

# Définition des classes CNN et CNN-RNN
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
        if len(x.size()) != 5:
            raise ValueError(f"Les données d'entrée doivent avoir 5 dimensions (batch, sequence_length, channels, height, width), mais ont {len(x.size())} dimensions.")
        
        batch_size, sequence_length, channels, height, width = x.size()
        cnn_features = []

        # Appliquer CNN sur chaque image dans la séquence
        for t in range(sequence_length):
            cnn_out = self.cnn(x[:, t, :, :, :])
            print(f"Shape of CNN output at timestep {t}: {cnn_out.shape}")  # Ajout pour déboguer
            cnn_features.append(cnn_out)

        cnn_features = torch.stack(cnn_features, dim=1)
        print(f"Shape of CNN features stacked: {cnn_features.shape}")  # Ajout pour déboguer

        rnn_out, _ = self.lstm(cnn_features)
        output = self.fc(rnn_out[:, -1, :])
        return output
    
def get_cnn_output_size(cnn, input_shape):
    dummy_input = torch.randn(*input_shape)
    output = cnn(dummy_input)
    return output.numel()
# Hyperparamètres d'exemple
input_size = 128 * 24 * 24  # Changez selon la sortie de votre CNN (hauteur, largeur)
hidden_size = 256
num_layers = 2
output_size = 3  # Trois classes : fond, cyclone tropical, rivière atmosphérique

# Déterminer la taille réelle de la sortie du CNN
input_shape = (1, 3, 768, 1152)  # Remplacez par les valeurs réelles de `channels`, `height`, `width`
actual_cnn_output_size = get_cnn_output_size(CNN(), input_shape)

# Initialisation du modèle
model = CNN_RNN_Model(input_size=actual_cnn_output_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Charger les fichiers .npy et préparer les données
train_files = sorted(glob.glob("data/sample/Train/*.npy"))
train_data, train_labels = load_npy_data(train_files, sequence_length=5)

# Convertir en tenseurs PyTorch
train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)

# Créer un DataLoader pour l'entraînement
train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Définir l'appareil
device = torch.device("cpu")  # ou 'cuda' si vous avez accès au GPU

# Initialisation du modèle
model = CNN_RNN_Model(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
model.to(device)

# Réduire la taille du batch
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Boucle d'entraînement
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        # Zéro le gradient
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass et optimisation
        loss.backward()
        optimizer.step()
        
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # Libérer la mémoire GPU si nécessaire
    torch.cuda.empty_cache()