import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from ClimateDataset import ClimateDataset
from modele import CGNet  # Importer le nouveau modèle simplifié

# Initialisation de l'appareil
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Utilisation de l'appareil : {device}")

# Charger le modèle simplifié
model_path = "modele_CGNet/model.pth"  # Nouveau chemin pour le modèle simplifié
model = CGNet(num_classes=3, num_channels=7 * 3).to(device)  # Adapter le nombre de canaux
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("[INFO] Modèle CGNet chargé avec succès.")

# Charger les données de test
test_data_dir = "./data/raw/Test"
test_dataset = ClimateDataset(data_dir=test_data_dir, sequence_length=3, target_size=(128, 128))
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
print(f"[INFO] Chargement des données de test depuis : {test_data_dir}")

# Initialisation des listes pour collecter les prédictions et labels
all_preds = []
all_labels = []

# Boucle de test
print("[INFO] Début de l'évaluation...")
with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(images.size(0), -1, images.size(3), images.size(4)).to(device)
        labels = labels.to(device)

        # Prédictions
        outputs = model(images)
        outputs = F.interpolate(outputs, size=labels.shape[1:], mode="bilinear", align_corners=False)

        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds.flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

print("[INFO] Évaluation terminée.")

# Calcul des métriques
accuracy = accuracy_score(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
report = classification_report(
    all_labels,
    all_preds,
    target_names=["Background", "Tropical Cyclone", "Atmospheric River"],
    zero_division=0,
)

# Affichage des résultats
print("\n[INFO] Résultats :")
print(f"Accuracy : {accuracy:.4f}")
print("\nRapport de classification :")
print(report)

# Affichage de la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Background", "Tropical Cyclone", "Atmospheric River"],
    yticklabels=["Background", "Tropical Cyclone", "Atmospheric River"],
)
plt.xlabel("Prédictions")
plt.ylabel("Vérité Terrain")
plt.title("Matrice de Confusion")
plt.savefig("modele_CGNet/confusion_matrix.png")
plt.show()

print("[INFO] Matrice de confusion sauvegardée sous 'confusion_matrix.png'.")