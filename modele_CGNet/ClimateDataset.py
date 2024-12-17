import os
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms

class ClimateDataset(Dataset):
    def __init__(self, data_dir, sequence_length=5, target_size=(16, 16), augment=False):
        # Chargement et tri des fichiers .nc
        self.data_files = sorted([os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.nc')])
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.augment = augment
        print(f"[INFO] Nombre de fichiers .nc trouvés : {len(self.data_files)}", flush=True)

        # Normalisation pour chaque variable
        self.stats = {
            'TMQ': {'mean': 19.218493883511705, 'std': 15.81727654092718},
            'U850': {'mean': 1.5530236518163378, 'std': 8.297621410859342},
            'V850': {'mean': 0.2541317070842832, 'std': 6.231631284217669},
            'PS': {'mean': 96571.61180904522, 'std': 9700.120187600089},
            'PSL': {'mean': 100814.0746789503, 'std': 1461.0813611540402},
            'T200': {'mean': 213.2090852937285, 'std': 7.889733635786149},
            'T500': {'mean': 253.03822411573202, 'std': 12.825342082153885},
        }

        # Définir les transformations pour les augmentations de données
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomApply([transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05)], p=0.5),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5)
        ])

    def __len__(self):
        return len(self.data_files) - self.sequence_length + 1

    def __getitem__(self, idx):
        features_seq = []

        for i in range(self.sequence_length):
            file_path = self.data_files[idx + i]
            ds = xr.open_dataset(file_path, engine='netcdf4')

            features = []
            for var, stats in self.stats.items():
                data = ds[var].values.squeeze()
                data = (data - stats['mean']) / stats['std']
                data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
                data = F.adaptive_avg_pool2d(data, self.target_size).squeeze(0)
                features.append(data.numpy())

            features = np.stack(features, axis=0)
            features_seq.append(features)

        features_seq = np.stack(features_seq, axis=0)

        # Appliquer les augmentations de données si nécessaire
        features_seq = torch.tensor(features_seq, dtype=torch.float32)
        if self.augment:
            features_seq = self.apply_augmentation(features_seq)

        # Traitement des labels avec réduction de taille
        labels = ds['LABELS'].values.astype(np.int64)
        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(0)  # Conversion en Float pour le pooling
        labels = F.adaptive_max_pool2d(labels, self.target_size).squeeze(0).to(torch.long)  # Reconvertir en Long après le pooling

        return features_seq, labels

    def apply_augmentation(self, features_seq):
        augmented_seq = []
        for t in range(features_seq.size(0)):  # Boucle sur la séquence temporelle
            features = features_seq[t]  # Shape: (Channels, H, W)
            features = transforms.ToPILImage()(features)  # Convertir en image PIL
            features = self.transform(features)  # Appliquer les transformations
            features = transforms.ToTensor()(features)  # Reconvertir en tenseur
            augmented_seq.append(features)

        return torch.stack(augmented_seq)  # Shape: (Sequence, Channels, H, W)