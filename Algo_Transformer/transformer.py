import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings("ignore")

def print_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {message}", flush=True)

class IoUMetric:
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.intersection = torch.zeros(num_classes)
        self.union = torch.zeros(num_classes)
        self.class_names = ['background', 'cyclone', 'river']

    def update(self, outputs, targets):
        preds = outputs.argmax(1)
        for class_idx in range(self.num_classes):
            pred_mask = (preds == class_idx)
            target_mask = (targets == class_idx)
            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()
            self.intersection[class_idx] += intersection
            self.union[class_idx] += union

    def compute(self):
        iou = self.intersection / (self.union + 1e-8)
        return {self.class_names[i]: iou[i].item() for i in range(self.num_classes)}

    def reset(self):
        self.intersection = torch.zeros(self.num_classes)
        self.union = torch.zeros(self.num_classes)

class WeatherViT(nn.Module):
    def __init__(self, input_channels=6, num_classes=3):
        super().__init__()
        print_log("Initialisation de l'architecture du modèle")
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.encoded_size = 8 * 8
        self.encoded_dim = 128
        
        print_log("Configuration du Transformer")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.encoded_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        print_log("Configuration du classificateur")
        self.classifier = nn.Sequential(
            nn.Linear(self.encoded_dim * self.encoded_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        print_log("Initialisation des poids")
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.encoder(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B, H*W, C)
        x = self.transformer(x)
        x = x.reshape(B, -1)
        x = self.classifier(x)
        return x

class WeatherDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        print_log("Initialisation du Dataset...")
        self.data_dir = Path(data_dir)
        self.split = split
        self.samples = []
        
        patch_dir = self.data_dir / 'balanced_patches'
        print_log(f"Recherche des fichiers dans {patch_dir}")
        
        if not patch_dir.exists():
            print_log(f"ERREUR: Dossier {patch_dir} non trouvé!")
            raise FileNotFoundError(f"Dossier {patch_dir} non trouvé!")
        
        h5_files = list(patch_dir.glob('*.h5'))
        print_log(f"Nombre de fichiers h5 trouvés: {len(h5_files)}")
        
        total_samples = 0
        class_counts = {'background': 0, 'cyclone': 0, 'river': 0}
        
        for h5_file in tqdm(h5_files, desc="Chargement des références"):
            try:
                with h5py.File(h5_file, 'r') as f:
                    for class_name in ['background', 'cyclone', 'river']:
                        if class_name in f:
                            group = f[class_name]
                            n_samples = len(group.keys())
                            class_counts[class_name] += n_samples
                            total_samples += n_samples
                            
                            for patch_name in group.keys():
                                self.samples.append((str(h5_file), class_name, patch_name))
                
            except Exception as e:
                print_log(f"ERREUR lors du traitement de {h5_file}: {str(e)}")
                continue
        
        print_log(f"Chargement terminé:")
        print_log(f"- Total échantillons: {total_samples}")
        print_log(f"- Background: {class_counts['background']}")
        print_log(f"- Cyclones: {class_counts['cyclone']}")
        print_log(f"- Rivières: {class_counts['river']}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        h5_file, class_name, patch_name = self.samples[idx]
        
        try:
            with h5py.File(h5_file, 'r') as f:
                patch_group = f[class_name][patch_name]
                data = []
                for var in ['TMQ', 'U850', 'V850', 'UBOT', 'PS', 'PRECT']:
                    if var in patch_group:
                        data.append(patch_group[var][()])
                
                x = np.stack(data, axis=0)
                label = 0 if class_name == 'background' else 1 if class_name == 'cyclone' else 2
                
                return torch.FloatTensor(x), torch.LongTensor([label])
                
        except Exception as e:
            print_log(f"ERREUR lors du chargement de l'échantillon {idx}: {str(e)}")
            raise

def train_epoch(epoch, model, train_loader, criterion, optimizer, device, num_epochs):
    model.train()
    running_loss = 0.0
    iou_metric = IoUMetric()
    
    with tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}') as pbar:
        for batch_idx, (data, target) in enumerate(pbar):
            data = data.to(device)
            target = target.to(device).squeeze()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            iou_metric.update(output, target)
            
            current_ious = iou_metric.compute()
            pbar_dict = {
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'IoU_bg': f'{current_ious["background"]:.3f}',
                'IoU_cyc': f'{current_ious["cyclone"]:.3f}',
                'IoU_riv': f'{current_ious["river"]:.3f}'
            }
            pbar.set_postfix(pbar_dict)
    
    avg_loss = running_loss / len(train_loader)
    final_ious = iou_metric.compute()
    
    return {
        'loss': avg_loss,
        'iou_background': final_ious['background'],
        'iou_cyclone': final_ious['cyclone'],
        'iou_river': final_ious['river']
    }

def main():
    print_log("Démarrage du programme")
    
    config = {
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': Path('D:/Bureau/DATA/Data_Transformer'),
    }
    
    print_log("Configuration:")
    for key, value in config.items():
        print_log(f"- {key}: {value}")
    
    try:
        model_dir = config['save_dir'] / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        train_dataset = WeatherDataset('D:/Bureau/DATA/Data_restructure')
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        model = WeatherViT().to(config['device'])
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([0.1, 0.45, 0.45]).to(config['device'])
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        best_metrics = {
            'loss': float('inf'),
            'iou_background': 0,
            'iou_cyclone': 0,
            'iou_river': 0
        }
        
        print_log("Début de l'entraînement")
        
        for epoch in range(config['num_epochs']):
            metrics = train_epoch(
                epoch + 1,
                model,
                train_loader,
                criterion,
                optimizer,
                config['device'],
                config['num_epochs']
            )
            
            print_log(f"Epoch {epoch+1}:")
            print_log(f"  Loss: {metrics['loss']:.4f}")
            print_log(f"  IoU Background: {metrics['iou_background']:.4f}")
            print_log(f"  IoU Cyclone: {metrics['iou_cyclone']:.4f}")
            print_log(f"  IoU River: {metrics['iou_river']:.4f}")
            
            scheduler.step(metrics['loss'])
            
            if metrics['loss'] < best_metrics['loss']:
                best_metrics = metrics.copy()
                print_log(f"Nouveau meilleur modèle!")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics
                }, model_dir / 'best_model.pt')
            
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics
                }, model_dir / f'checkpoint_epoch_{epoch+1}.pt')
                print_log(f"Checkpoint sauvegardé (epoch {epoch+1})")
        
        print_log("\nMeilleures métriques obtenues:")
        print_log(f"  Loss: {best_metrics['loss']:.4f}")
        print_log(f"  IoU Background: {best_metrics['iou_background']:.4f}")
        print_log(f"  IoU Cyclone: {best_metrics['iou_cyclone']:.4f}")
        print_log(f"  IoU River: {best_metrics['iou_river']:.4f}")
        
    except Exception as e:
        print_log(f"ERREUR CRITIQUE: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_log("Interruption par l'utilisateur")
    except Exception as e:
        print_log(f"ERREUR FATALE: {str(e)}")
        print_log("Traceback complet:")
        import traceback
        traceback.print_exc()