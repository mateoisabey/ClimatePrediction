import os
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset

class CustomClimateDataset(Dataset):
    def __init__(self, data_dir):
        self.data_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.nc')]
        self.data_files.sort()
        print(f"[INFO] Nombre de fichiers .nc trouvés : {len(self.data_files)}", flush=True)
        
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
        return len(self.data_files)

    def __getitem__(self, idx):
        file = self.data_files[idx]

        ds = xr.open_dataset(file)
        
        # Extraire les 16 variables climatiques
        features = []
        for var in self.stats.keys():
            data = ds[var].values.squeeze()  # Supprimer la dimension temporelle
            data = (data - self.stats[var]['mean']) / self.stats[var]['std']  # Normalisation
            features.append(data)
        
        # Empiler les données pour créer un tensor avec 16 canaux
        features = np.stack(features, axis=0)  # (16, lat, lon)
        
        # Extraire les labels et s'assurer qu'ils sont en type int64
        labels = ds['LABELS'].values.astype(np.int64)  # Assurez-vous que les labels soient des entiers

        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)