import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Définir les paramètres de la validation croisée
k_folds = 2
num_epochs = 80
batch_size = 8
learning_rate = 0.001

# Charger les données
data_dir = "./data/raw/Train"
dataset = ClimateDataset(data_dir=data_dir, sequence_length=5, target_size=(64, 64))
logger.info(f"Chargement des données terminé : {len(dataset)} échantillons disponibles pour la validation croisée.")

# Calculer les poids des classes pour équilibrer la perte
all_labels = []
for _, label in dataset:
    unique_labels = torch.unique(label).cpu().numpy()
    all_labels.extend(unique_labels)

# Calculer les occurrences pour chaque classe
class_counts = np.bincount(all_labels, minlength=3)  # Assure que toutes les classes (0, 1, 2) sont comptées
logger.info(f"Distribution des classes : {class_counts}")

# Calculer les poids des classes pour CrossEntropyLoss
class_weights = torch.tensor([1.0 / c if c > 0 else 0.0 for c in class_counts], dtype=torch.float32).to(device)
logger.info(f"Poids des classes : {class_weights}")

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

# Sous-échantillonnage et sur-échantillonnage
class_0_indices = [i for i, (_, label) in enumerate(dataset) if (label == 0).any()]
class_1_indices = [i for i, (_, label) in enumerate(dataset) if (label == 1).any()]
class_2_indices = [i for i, (_, label) in enumerate(dataset) if (label == 2).any()]

# Taille cible pour équilibrer les classes
target_size = max(len(class_1_indices), len(class_2_indices))

# Sous-échantillonner la classe dominante (classe 0)
if len(class_0_indices) > target_size:
    class_0_indices = np.random.choice(class_0_indices, size=target_size, replace=False)
else:
    logger.warning(
        f"Taille de la classe 0 ({len(class_0_indices)}) inférieure à la taille cible ({target_size})."
        " Utilisation de tous les échantillons disponibles pour la classe 0."
    )

# Sur-échantillonner les classes minoritaires (classes 1 et 2) si nécessaire
class_1_indices = np.random.choice(class_1_indices, size=target_size, replace=True)
class_2_indices = np.random.choice(class_2_indices, size=target_size, replace=True)

# Combiner les indices pour créer un dataset équilibré
balanced_indices = np.concatenate([class_0_indices, class_1_indices, class_2_indices])

# K-Fold Cross-Validation
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_results = {"fold": [], "train_loss": [], "val_loss": [], "accuracy": [], "iou": [], "f1": []}

# Validation croisée
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    logger.info(f"Début de la validation croisée pour le fold {fold + 1}/{k_folds}.")
    fold_start_time = time.time()

    # Préparer les données pour ce fold
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    # Poids des échantillons pour WeightedRandomSampler
    samples_weights = []
    for _, label in train_subset:
        unique_labels, counts = torch.unique(label, return_counts=True)
        sample_weight = sum(class_weights[unique_labels] * counts.float()).item()
        samples_weights.append(sample_weight)
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)

    train_loader = DataLoader(train_subset, sampler=sampler, batch_size=batch_size)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Initialiser le modèle, la perte et l'optimiseur pour ce fold
    model = CGNet(num_classes=3, num_channels=16 * 5).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2)  # Utilisation de la Focal Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Scheduler pour ajuster dynamiquement le taux d'apprentissage
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

    # Historique des pertes pour ce fold
    fold_train_loss = []
    fold_val_loss = []

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
                logger.info(
                    f"[Fold {fold + 1}, Époch {epoch + 1}] Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}"
                )

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

        fold_train_loss.append(avg_train_loss)
        fold_val_loss.append(avg_val_loss)

        epoch_time = time.time() - epoch_start_time
        logger.info(
            f"[Fold {fold + 1}, Époch {epoch + 1}] terminé : Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
            f"Accuracy: {acc:.4f}, IoU: {iou}, F1: {f1:.4f} - Temps: {epoch_time:.2f}s"
        )
        scheduler.step(avg_val_loss)

    # Sauvegarder les métriques pour ce fold
    fold_results["fold"].append(fold + 1)
    fold_results["train_loss"].append(np.mean(fold_train_loss))
    fold_results["val_loss"].append(np.mean(fold_val_loss))
    fold_results["accuracy"].append(acc)
    fold_results["iou"].append(iou)
    fold_results["f1"].append(f1)

    fold_model_path = f"modele_CGNet/fold_{fold + 1}_model.pth"
    torch.save(model.state_dict(), fold_model_path)
    logger.info(f"Modèle pour le fold {fold + 1} sauvegardé à {fold_model_path}. Temps total: {time.time() - fold_start_time:.2f}s")

logger.info("Validation croisée terminée.")
results_path = "modele_CGNet/kfold_results.json"
with open(results_path, "w") as f:
    import json
    json.dump(fold_results, f, indent=4)
logger.info(f"Résultats sauvegardés dans {results_path}.")