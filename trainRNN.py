from models import SimpleCNN
from modele_Hybride import CNN_BiLSTM_Model
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataRNN import ClimateDataset
import csv
from pathlib import Path
import matplotlib.pyplot as plt

# Définir le device pour PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger le modèle hybride CNN-LSTM
print("[INFO] Chargement du modèle hybride CNN-LSTM...", flush=True)

cnn_model = SimpleCNN(dropout_rate=0.4)  # Ajuster le taux de dropout si nécessaire
hidden_dim = 2048
num_layers = 4
output_dim = 3
target_size = (64, 64)

model = CNN_BiLSTM_Model(
    cnn_model=cnn_model, hidden_dim=hidden_dim, num_layers=num_layers, 
    output_dim=output_dim, target_size=target_size
).to(device)


# Charger le dataset
print("[INFO] Chargement du dataset...", flush=True)
train_dataset = ClimateDataset(data_dir="./data/raw/Train", sequence_length=5, target_size=target_size)
print(f"[INFO] Dataset chargé avec {len(train_dataset)} exemples", flush=True)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Configuration de l'entraînement
class_weights = torch.tensor([0.01, 150.0, 26.5]).to(device)  # Ajout d'une pondération pour la classe 1
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Initialiser les meilleures métriques et historique
num_epochs = 200
num_classes = 3
history = {"loss": [], "accuracy": [], "precision": [], "recall": [], "f1": [], "iou": []}

# Fonction de calcul d’IoU par classe
def calculate_iou_per_class(preds, labels, num_classes):
    iou_per_class = []
    for cls in range(num_classes):
        intersection = ((preds == cls) & (labels == cls)).sum()
        union = ((preds == cls) | (labels == cls)).sum()
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append(intersection / union)
    return iou_per_class

# Chemin vers le fichier CSV
csv_file = "training_metrics.csv"

# Initialiser le fichier CSV avec des en-têtes si le fichier n'existe pas
if not Path(csv_file).is_file():
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Loss", "Accuracy", "Precision", "Recall", "F1", "IoU_Class_0", "IoU_Class_1", "IoU_Class_2"])

# Boucle d'entraînement
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    iou_global_per_class = [0] * num_classes
    
    epoch_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
    num_batches = len(train_loader)

    print(f"[INFO] Début de l'époque {epoch+1}/{num_epochs}", flush=True)
    
    for batch_idx, (images_seq, labels) in enumerate(train_loader):
        images_seq = images_seq.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images_seq)
        outputs = outputs.permute(0, 2, 3, 1).reshape(-1, outputs.shape[1])
        labels = labels.view(-1)

        # Calcul de la perte
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        # Convertir les prédictions
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # Calcul des métriques par batch
        batch_acc = accuracy_score(labels.cpu().numpy(), preds)
        batch_precision = precision_score(labels.cpu().numpy(), preds, average='weighted', zero_division=0)
        batch_recall = recall_score(labels.cpu().numpy(), preds, average='weighted', zero_division=0)
        batch_f1 = f1_score(labels.cpu().numpy(), preds, average='weighted', zero_division=0)
        batch_iou = calculate_iou_per_class(preds, labels.cpu().numpy(), num_classes)

        # Affichage des métriques de batch
        print(f"[Batch {batch_idx+1}/{num_batches}] Loss: {loss.item():.4f}, Acc: {batch_acc:.4f}, "
              f"Prec: {batch_precision:.4f}, Rec: {batch_recall:.4f}, F1: {batch_f1:.4f}, IoU: {batch_iou}", flush=True)

        # Accumuler les IoU globales
        for cls, iou in enumerate(batch_iou):
            if not torch.isnan(torch.tensor(iou)):
                iou_global_per_class[cls] += iou

    # Calcul des métriques moyennes par époque
    epoch_avg_acc = accuracy_score(all_labels, all_preds)
    epoch_avg_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    epoch_avg_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    epoch_avg_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    epoch_avg_iou = [iou / num_batches for iou in iou_global_per_class]

    # Mise à jour de l'historique
    history["loss"].append(total_loss / num_batches)
    history["accuracy"].append(epoch_avg_acc)
    history["precision"].append(epoch_avg_precision)
    history["recall"].append(epoch_avg_recall)
    history["f1"].append(epoch_avg_f1)
    history["iou"].append(epoch_avg_iou)

    # Écriture des métriques dans le fichier CSV
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            epoch + 1, 
            total_loss / num_batches, 
            epoch_avg_acc, 
            epoch_avg_precision, 
            epoch_avg_recall, 
            epoch_avg_f1, 
            *epoch_avg_iou
        ])

    # Afficher les résultats
    print(f"[INFO] Epoch {epoch+1} - Loss: {total_loss / num_batches:.4f}, "
          f"Accuracy: {epoch_avg_acc:.4f}, Precision: {epoch_avg_precision:.4f}, "
          f"Recall: {epoch_avg_recall:.4f}, F1: {epoch_avg_f1:.4f}, IoU per class: {epoch_avg_iou}", flush=True)

    # Enregistrer le modèle à la dernière époque
    if epoch == num_epochs - 1:
        torch.save(model.state_dict(), f"final_model.pth")
        print("[INFO] Modèle sauvegardé à la fin de l'entraînement.", flush=True)

    # Mise à jour du scheduler
    scheduler.step(total_loss / num_batches)