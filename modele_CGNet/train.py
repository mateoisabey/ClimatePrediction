import json
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score
from modele import CGNet
from ClimateDataset import ClimateDataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import logging

# Initialisation du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Initialisation du modèle
if torch.backends.mps.is_available():
    device = torch.device("mps")  # GPU M1/M2 via MPS
elif torch.cuda.is_available():
    device = torch.device("cuda")  # GPU CUDA si disponible
else:
    device = torch.device("cpu")
logger.info(f"Utilisation de l'appareil : {device}")

# Définir les paramètres
num_epochs = 200
batch_size = 8
learning_rate = 0.001
validation_split = 0.2

# Charger les données
data_dir = "./data/raw/Train"
dataset = ClimateDataset(data_dir=data_dir, sequence_length=3, target_size=(128, 128))
logger.info(f"Chargement des données terminé : {len(dataset)} échantillons disponibles.")

# Diviser les données en 80% entraînement et 20% validation
train_size = int((1 - validation_split) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
logger.info(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Calculer la distribution des classes dans l'ensemble des données
all_labels = []
for _, label in dataset:
    unique_labels, counts = torch.unique(label, return_counts=True)
    all_labels.extend(unique_labels.repeat_interleave(counts).cpu().numpy())

# Calculer les occurrences pour chaque classe
class_counts = np.bincount(all_labels, minlength=3)
logger.info(f"Distribution des classes : {class_counts}")

class_weights = torch.tensor([1.0 , 50.0 , 10.0], dtype=torch.float32).to(device)
logger.info(f"Poids des classes ajustés dynamiquement : {class_weights}")

# Définir la Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, reduction='none', weight=class_weights)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# Fonction pour calculer l'IoU
def calculate_iou(preds, labels, num_classes):
    iou = []
    for cls in range(num_classes):
        intersection = ((preds == cls) & (labels == cls)).sum()
        union = ((preds == cls) | (labels == cls)).sum()
        iou.append(intersection / union if union > 0 else 0)
    return iou

# Initialiser le modèle, la perte et l'optimiseur
model = CGNet(num_classes=3, num_channels=7 * 3).to(device)
criterion = FocalLoss(alpha=0.25, gamma=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# Initialisation des métriques
metrics = {
    "epochs": []
}

# Entraînement et validation
for epoch in range(num_epochs):
    epoch_start_time = time.time()

    # Entraînement
    model.train()
    total_train_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.view(images.size(0), -1, images.size(3), images.size(4)).to(device).float()
        labels = labels.to(device).long()

        optimizer.zero_grad()
        outputs = model(images)
        outputs = F.interpolate(outputs, size=labels.shape[1:], mode="bilinear", align_corners=False).float()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        if batch_idx % 10 == 0:
            logger.info(f"[Époch {epoch + 1}] Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

    # Validation
    model.eval()
    total_val_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.view(images.size(0), -1, images.size(3), images.size(4)).to(device).float()
            labels = labels.to(device).long()
            outputs = model(images)
            outputs = F.interpolate(outputs, size=labels.shape[1:], mode="bilinear", align_corners=False).float()
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    acc = accuracy_score(all_labels, all_preds)
    iou = calculate_iou(np.array(all_preds), np.array(all_labels), num_classes=3)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    epoch_time = time.time() - epoch_start_time
    logger.info(
        f"[Époch {epoch + 1}] terminé : Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
        f"Accuracy: {acc:.4f}, IoU: {iou}, F1: {f1:.4f} - Temps: {epoch_time:.2f}s"
    )

    # Sauvegarder les métriques pour cette époque
    metrics["epochs"].append({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "accuracy": acc,
        "iou": iou,
        "f1_score": f1,
        "time": epoch_time
    })

# Sauvegarder le modèle
model_path = "modele_CGNet/model.pth"
torch.save(model.state_dict(), model_path)
logger.info(f"Modèle sauvegardé à {model_path}.")

# Sauvegarder les métriques dans un fichier JSON
metrics_path = "modele_CGNet/metrics.json"
with open(metrics_path, "w") as json_file:
    json.dump(metrics, json_file, indent=4)
logger.info(f"Métriques sauvegardées dans {metrics_path}.")