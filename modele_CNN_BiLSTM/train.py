import time
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models import SimpleCNN
from data import CustomClimateDataset

# Définir le device pour PyTorch (utiliser le GPU si disponible, sinon le CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger le modèle et déplacer sur le device
print("[INFO] Chargement du modèle...", flush=True)
model = SimpleCNN().to(device)

# Charger le dataset
print("[INFO] Chargement du dataset...", flush=True)
train_dataset = CustomClimateDataset(data_dir="./data/raw/Train")

# Afficher le nombre d'exemples trouvés dans le dataset
print(f"[INFO] Dataset chargé avec {len(train_dataset)} exemples", flush=True)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Configuration de l'entraînement
class_weights = torch.tensor([0.5, 2.0, 2.5]).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# Utilisation d'un optimizer avec un scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Initialiser le meilleur F1-score pour la sauvegarde du modèle
best_f1 = 0.0

# Fonction de calcul de l'IoU
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

# Boucle d'entraînement
num_epochs = 15
num_classes = 3

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    epoch_acc_sum = 0
    epoch_precision_sum = 0
    epoch_recall_sum = 0
    epoch_f1_sum = 0
    epoch_iou_sum = 0
    num_batches = len(train_loader)

    print(f"[INFO] Début de l'époque {epoch+1}/{num_epochs}", flush=True)
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)  # Déplacer les images sur le GPU si disponible
        labels = labels.to(device)  # Déplacer les labels sur le GPU si disponible

        optimizer.zero_grad()
        outputs = model(images)  # Output attendu : (batch_size, num_classes, height, width)
        
        labels = labels.view(-1)
        
        outputs = outputs.permute(0, 2, 3, 1).reshape(-1, outputs.shape[1])
        
        # Calculer la perte
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        # Convertir les sorties en prédictions de classe
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # Calculer et afficher les métriques pour ce batch
        batch_acc = accuracy_score(labels.cpu().numpy(), preds)
        batch_precision = precision_score(labels.cpu().numpy(), preds, average='weighted', zero_division=0)
        batch_recall = recall_score(labels.cpu().numpy(), preds, average='weighted', zero_division=0)
        batch_f1 = f1_score(labels.cpu().numpy(), preds, average='weighted', zero_division=0)
        batch_iou = calculate_iou(preds, labels.cpu().numpy(), num_classes)

        print(f"[INFO] Batch {batch_idx+1}/{num_batches} - Loss: {loss.item():.4f}, "
              f"Accuracy: {batch_acc:.4f}, Precision: {batch_precision:.4f}, "
              f"Recall: {batch_recall:.4f}, F1-score: {batch_f1:.4f}, IoU: {batch_iou:.4f}", flush=True)

        # Accumuler les métriques pour calculer la moyenne par époque
        epoch_acc_sum += batch_acc
        epoch_precision_sum += batch_precision
        epoch_recall_sum += batch_recall
        epoch_f1_sum += batch_f1
        epoch_iou_sum += batch_iou

        torch.cuda.empty_cache()

    # Ajouter des messages de débogage pour vérifier le bon déroulement
    print("[DEBUG] Fin du dernier batch de l'époque.", flush=True)

    # Calculer les moyennes des métriques pour l'époque
    epoch_avg_acc = epoch_acc_sum / num_batches
    epoch_avg_precision = epoch_precision_sum / num_batches
    epoch_avg_recall = epoch_recall_sum / num_batches
    epoch_avg_f1 = epoch_f1_sum / num_batches
    epoch_avg_iou = epoch_iou_sum / num_batches
    print("[DEBUG] Calcul des métriques moyennes terminé.", flush=True)

    # Afficher la perte totale moyenne et les métriques moyennes par époque
    print(f"[INFO] Fin de l'époque {epoch+1}, Perte Moyenne : {total_loss / num_batches:.4f}", flush=True)
    print(f"[INFO] Epoch Metrics - Accuracy: {epoch_avg_acc:.4f}, Precision: {epoch_avg_precision:.4f}, "
          f"Recall: {epoch_avg_recall:.4f}, F1-score: {epoch_avg_f1:.4f}, Mean IoU: {epoch_avg_iou:.4f}", flush=True)

    # Sauvegarder le modèle si le F1-score moyen est meilleur que le précédent
    if epoch_avg_f1 > best_f1:
        best_f1 = epoch_avg_f1
        torch.save(model.state_dict(), f"best_model_epoch_{epoch+1}.pth")
        print(f"[INFO] Modèle sauvegardé avec un F1-score de {best_f1:.4f}", flush=True)
   
    print("[DEBUG] Fin de la sauvegarde du modèle et de l'époque.", flush=True)

    # Ajout d'une petite pause pour s'assurer que toutes les opérations sont terminées
    time.sleep(1)

    # Mise à jour du scheduler
    scheduler.step(total_loss / num_batches)

end_time = time.time()