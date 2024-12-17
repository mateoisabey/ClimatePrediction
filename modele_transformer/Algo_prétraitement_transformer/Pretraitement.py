import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import h5py
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import sys

def setup_logger():
    logger = logging.getLogger('training')
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    log_dir = Path('D:/Bureau/DATA/Data_Transformer/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

class WeatherDataset(Dataset):
    def __init__(self, data_dir, split='train', logger=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.samples = []
        self.logger = logger or logging.getLogger('training')
        
        patch_dir = self.data_dir / 'balanced_patches'
        self.logger.info(f"Chargement des données depuis {patch_dir}")
        
        if not patch_dir.exists():
            raise FileNotFoundError(f"Dossier {patch_dir} non trouvé!")
        
        for h5_file in patch_dir.glob('*.h5'):
            with h5py.File(h5_file, 'r') as f:
                for class_name in ['background', 'cyclone', 'river']:
                    if class_name in f:
                        group = f[class_name]
                        for patch_name in group.keys():
                            self.samples.append((h5_file, class_name, patch_name))
        
        self.logger.info(f"Dataset chargé: {len(self.samples)} échantillons")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        h5_file, class_name, patch_name = self.samples[idx]
        
        with h5py.File(h5_file, 'r') as f:
            patch_group = f[class_name][patch_name]
            data = []
            for var in ['TMQ', 'U850', 'V850', 'UBOT', 'PS', 'PRECT']:
                if var in patch_group:
                    data.append(patch_group[var][()])
            
            x = np.stack(data, axis=0)
            label = 0 if class_name == 'background' else 1 if class_name == 'cyclone' else 2
            
            return torch.FloatTensor(x), torch.LongTensor([label])

class WeatherViT(nn.Module):
    def __init__(
        self,
        input_channels=6,
        patch_size=16,
        dim=256,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        num_classes=3
    ):
        super().__init__()
        
        # Patch Embedding
        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                     p1=patch_size, p2=patch_size),
            nn.Linear((patch_size ** 2) * input_channels, dim)
        )
        
        # Position Embedding
        num_patches = (64 // patch_size) * (64 // patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer Encoder
        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                batch_first=True
            ) for _ in range(depth)
        ])
        
        # Classification Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # Patch Embedding
        x = self.patch_embed(x)
        b, n, _ = x.shape
        
        # Add CLS token and position embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.transformer:
            x = block(x)
        
        # MLP head (use CLS token)
        return self.mlp_head(x[:, 0])

def train_model():
    logger = setup_logger()
    
    config = {
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': Path('D:/Bureau/DATA/Data_Transformer')
    }
    
    logger.info(f"Configuration: {config}")
    
    model_dir = config['save_dir'] / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(config['save_dir'] / 'logs' / 'train')
    
    logger.info("Chargement du dataset...")
    try:
        train_dataset = WeatherDataset('D:/Bureau/DATA/Data_restructure', logger=logger)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        logger.info(f"Dataset chargé - {len(train_dataset)} échantillons")
        
    except Exception as e:
        logger.error(f"Erreur chargement dataset: {str(e)}")
        raise
    
    logger.info("Initialisation du modèle...")
    try:
        model = WeatherViT().to(config['device'])
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.45, 0.45]).to(config['device']))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            verbose=True
        )
        logger.info("Modèle initialisé")
        
    except Exception as e:
        logger.error(f"Erreur initialisation modèle: {str(e)}")
        raise
    
    logger.info("Début de l'entraînement")
    best_loss = float('inf')
    
    try:
        for epoch in range(config['num_epochs']):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            with tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}') as pbar:
                for batch_idx, (data, target) in enumerate(pbar):
                    data = data.to(config['device'])
                    target = target.to(config['device']).squeeze()
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    # Calcul de l'accuracy
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                    
                    current_loss = loss.item()
                    current_acc = 100. * correct / total
                    
                    pbar.set_postfix({
                        'loss': f'{current_loss:.4f}',
                        'acc': f'{current_acc:.2f}%'
                    })
                    
                    # Logging TensorBoard
                    step = epoch * len(train_loader) + batch_idx
                    writer.add_scalar('Loss/train', current_loss, step)
                    writer.add_scalar('Accuracy/train', current_acc, step)
            
            # Métriques de fin d'epoch
            avg_loss = total_loss / len(train_loader)
            accuracy = 100. * correct / total
            logger.info(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%')
            
            # Scheduler step
            scheduler.step(avg_loss)
            
            # Sauvegarde du meilleur modèle
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'accuracy': accuracy
                }, model_dir / 'best_model.pt')
                logger.info(f'Meilleur modèle sauvegardé (loss: {best_loss:.4f})')
            
            # Checkpoint régulier
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'accuracy': accuracy
                }, model_dir / f'checkpoint_epoch_{epoch+1}.pt')
                logger.info(f'Checkpoint sauvegardé (epoch {epoch+1})')
    
    except Exception as e:
        logger.error(f"Erreur pendant l'entraînement: {str(e)}")
        raise
    
    finally:
        writer.close()
        logger.info("Entraînement terminé")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Erreur critique: {str(e)}")
        raise