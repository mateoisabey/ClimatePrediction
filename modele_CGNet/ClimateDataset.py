import os
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class ClimateDataset(Dataset):
    def __init__(self, data_dir, sequence_length=5, target_size=(16, 16)):
        # Chargement et tri des fichiers .nc
        self.data_files = sorted([os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.nc')])
        self.sequence_length = sequence_length
        self.target_size = target_size
        print(f"[INFO] Nombre de fichiers .nc trouvés : {len(self.data_files)}", flush=True)

        # Normalisation pour chaque variable
        self.stats = {
            'TMQ': {'mean': 19.218493883511705, 'std': 15.81727654092718},
            'U850': {'mean': 1.5530236518163378, 'std': 8.297621410859342},
            'V850': {'mean': 0.2541317070842832, 'std': 6.231631284217669},
            'UBOT': {'mean': 0.12487873639762347, 'std': 6.653287200647422},
            'VBOT': {'mean': 0.31541635530239687, 'std': 5.784189046243406},
            'QREFHT': {'mean': 0.007787778190110574, 'std': 0.006215287649268782},
            'PS': {'mean': 96571.61180904522, 'std': 9700.120187600089},
            'PSL': {'mean': 100814.0746789503, 'std': 1461.0813611540402},
            'T200': {'mean': 213.2090852937285, 'std': 7.889733635786149},
            'T500': {'mean': 253.03822411573202, 'std': 12.825342082153885},
            'PRECT': {'mean': 2.945823891654793e-08, 'std': 1.5563764223453562e-07},
            'TS': {'mean': 278.7114812161211, 'std': 23.68249400173724},
            'TREFHT': {'mean': 278.4212252760475, 'std': 22.51189231061532},
            'Z1000': {'mean': 474.1727845862705, 'std': 832.8081734781491},
            'Z200': {'mean': 11736.104626139959, 'std': 633.2586634526778},
            'ZBOT': {'mean': 61.31149022201575, 'std': 4.909502525398299},
        }

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
        
        # Traitement des labels avec réduction de taille
        labels = ds['LABELS'].values.astype(np.int64)
        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(0)  # Conversion en Float pour le pooling
        labels = F.adaptive_max_pool2d(labels, self.target_size).squeeze(0).to(torch.long)  # Reconvertir en Long après le pooling

        return torch.tensor(features_seq, dtype=torch.float32), labels.to(torch.long)