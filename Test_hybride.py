import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from modele_Hybride import CNN_LSTM_Model
from dataRNN import ClimateDataset
from modelsCNN import SimpleCNN

# Définir le device pour PyTorch (utiliser le GPU si disponible, sinon le CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fonction pour calculer l'IoU
def calculate_iou(preds, labels, num_classes):
    iou_per_class = []
    for cls in range(num_classes):
        intersection = ((preds == cls) & (labels == cls)).sum()
        union = ((preds == cls) | (labels == cls)).sum()
        if union == 0:
            iou = float('nan')  # Pour éviter la division par zéro
        else:
            iou = intersection / union
        iou_per_class.append(iou)
    return torch.nanmean(torch.tensor(iou_per_class))  # Moyenne des IoU par classe

# Charger le modèle et le déplacer sur le device
cnn_model = SimpleCNN()
hidden_dim = 128
num_layers = 2
output_dim = 3
target_size = (16, 16)
model = CNN_LSTM_Model(cnn_model=cnn_model, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim, target_size=target_size).to(device)

# Charger le modèle entraîné
model_path = "best_model_epoch_11.pth"  # Remplacez par le chemin du modèle sauvegardé que vous souhaitez utiliser
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Mettre le modèle en mode évaluation

# Charger le dataset de test
test_dataset = ClimateDataset(data_dir="./data/raw/Test", sequence_length=5, target_size=target_size)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Variables pour accumuler les prédictions et les vraies valeurs
all_preds = []
all_labels = []
total_iou = 0
batch_ious = []
num_classes = 3

# Boucle de test
print("[INFO] Début des prédictions sur le dataset de test...")
with torch.no_grad():
    for batch_idx, (images_seq, labels) in enumerate(test_loader):
        images_seq = images_seq.to(device)
        labels = labels.to(device)
        
        # Faire des prédictions
        outputs = model(images_seq)
        
        # Reshape pour correspondre aux dimensions (batch_size * height * width, num_classes)
        outputs = outputs.permute(0, 2, 3, 1).reshape(-1, outputs.shape[1])
        labels = labels.view(-1)
        
        # Convertir les sorties en prédictions de classe
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        true_labels = labels.cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(true_labels)

        # Calculer l'IoU pour ce batch
        batch_iou = calculate_iou(preds, true_labels, num_classes)
        total_iou += batch_iou
        batch_ious.append(batch_iou.item())

        # Vérifier que les formes correspondent avant de calculer les métriques
        if len(preds) == len(true_labels):
            # Calculer les métriques par batch
            batch_acc = accuracy_score(true_labels, preds)
            batch_precision = precision_score(true_labels, preds, average='weighted', zero_division=0)
            batch_recall = recall_score(true_labels, preds, average='weighted', zero_division=0)
            batch_f1 = f1_score(true_labels, preds, average='weighted', zero_division=0)
            
            print(f"[INFO] Batch {batch_idx+1}/{len(test_loader)} - "
                  f"Accuracy: {batch_acc:.4f}, Precision: {batch_precision:.4f}, "
                  f"Recall: {batch_recall:.4f}, F1-score: {batch_f1:.4f}, IoU: {batch_iou:.4f}")
        else:
            print(f"[WARNING] Les tailles des prédictions et des labels ne correspondent pas pour le batch {batch_idx+1}")

# Calculer les métriques globales sur tout le dataset de test
if len(all_preds) == len(all_labels):
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    mean_iou = total_iou / len(test_loader)

    # Afficher les résultats finaux
    print(f"[INFO] Résultats sur le dataset de test :")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1-score: {test_f1:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")

    # Matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matrice de Confusion")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # Histogramme des IoU par batch
    plt.figure()
    plt.hist(batch_ious, bins=10, color='blue', alpha=0.7)
    plt.xlabel("IoU par Batch")
    plt.ylabel("Nombre de Batches")
    plt.title("Distribution de l'IoU par Batch")
    plt.show()

else:
    print("[ERROR] Les tailles des prédictions et des labels ne correspondent pas sur l'ensemble du test.")