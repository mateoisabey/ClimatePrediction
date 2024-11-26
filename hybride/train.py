import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import xarray as xr
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from modelsRNN import ClimateRNN
from dataRNN import ClimateDataset

# Paramètres du modèle
input_size = 16 * 768 * 1152  # Nombre total de variables spatiales aplaties pour une seule séquence
hidden_size = 128             # Taille de l'état caché du LSTM
num_layers = 2                # Nombre de couches du LSTM
num_classes = 3               # Nombre de classes de sortie
sequence_length = 5           # Longueur de la séquence temporelle utilisée

# Définir le device pour PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialisation du modèle, de la fonction de perte et de l'optimiseur
model = ClimateRNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Charger le dataset et créer un DataLoader
train_dataset = ClimateDataset(data_dir="./data/raw/Train", sequence_length=sequence_length)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Entraînement du modèle
num_epochs = 10
best_f1 = 0.0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    print(f"[INFO] Début de l'époque {epoch+1}/{num_epochs}", flush=True)
    
    for batch_idx, (features_seq, labels) in enumerate(train_loader):
        # Redimensionner pour que chaque timestep soit aplati en une dimension unique
        features_seq = features_seq.view(1, sequence_length, -1).to(device)  # (batch_size, sequence_length, input_size)
        labels = labels.to(device)

        # Passer vers l'avant
        outputs = model(features_seq)
        loss = criterion(outputs, labels.view(-1))

        # Rétropropagation et optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        # Convertir les sorties en prédictions de classe
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    # Calcul des métriques pour l'époque
    epoch_accuracy = accuracy_score(all_labels, all_preds)
    epoch_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    print(f"[INFO] Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, "
          f"Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}, F1-score: {epoch_f1:.4f}", flush=True)

    # Sauvegarder le meilleur modèle
    if epoch_f1 > best_f1:
        best_f1 = epoch_f1
        torch.save(model.state_dict(), f"best_climate_rnn_epoch_{epoch+1}.pth")
        print(f"[INFO] Modèle sauvegardé avec un F1-score de {best_f1:.4f}", flush=True) 