import os
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader

# Normalisation basée sur les statistiques connues
VARIABLE_STATS = {
    "TMQ": {"mean": 19.218, "std": 15.817},
    "U850": {"mean": 1.553, "std": 8.297},
    'V850': {"mean": 0.2541317070842832, "std": 6.231631284217669},
    'UBOT': {"mean": 0.12487873639762347,  "std": 6.653287200647422},
    'VBOT': {"mean": 0.31541635530239687,  "std": 5.784189046243406},
    'QREFHT': {"mean": 0.007787778190110574,  "std": 0.006215287649268782},
    'PS': {"mean": 96571.61180904522,  "std":  9700.120187600089},
    'PSL': {"mean":  100814.0746789503, "std":  1461.0813611540402},
    'T200': { "mean": 213.2090852937285,  "std": 7.889733635786149},
    'T500': {"mean": 253.03822411573202, "std": 12.825342082153885},
    'PRECT': {"mean":  2.945823891654793e-08, "std":  1.5563764223453562e-07},
    'TS': {"mean": 278.7114812161211,  "std": 23.68249400173724},
    'TREFHT': {"mean":  278.4212252760475, "std":  22.51189231061532},
    'Z1000': {"mean":  474.1727845862705, "std":  832.8081734781491},
    'Z200': {"mean": 11736.104626139959,  "std": 633.2586634526778},
    'ZBOT': {"mean":  61.31149022201575,  "std": 4.909502525398299},
}

# Lire et normaliser les données
def normalize_data(data, var_name):
    mean = VARIABLE_STATS[var_name]["mean"]
    std = VARIABLE_STATS[var_name]["std"]
    return (data - mean) / std

# Dataset personnalisé pour PyTorch
class ClimateDataset(Dataset):
    def __init__(self, netcdf_dir, sequence_length):
        self.files = [os.path.join(netcdf_dir, f) for f in os.listdir(netcdf_dir) if f.endswith('.nc')]
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Charger un fichier NetCDF
        dataset = xr.open_dataset(self.files[idx])

        # Extraire les variables (exemple avec TMQ, U850, etc.)
        features = []
        for var in VARIABLE_STATS.keys():
            data = normalize_data(dataset[var].values, var)
            features.append(data)

        # Combiner les variables et créer un tableau
        features = np.stack(features, axis=-1)  # (lat, lon, features)
        labels = dataset["LABELS"].values  # (lat, lon)

        # Reshape pour créer une séquence (flattener lat/lon en une seule dimension)
        seq_features = features.reshape(-1, features.shape[-1])  # (lat * lon, features)
        seq_labels = labels.flatten()  # (lat * lon)

        return torch.tensor(seq_features, dtype=torch.float32), torch.tensor(seq_labels, dtype=torch.long)

# Préparer un DataLoader
def create_dataloader(netcdf_dir, sequence_length, batch_size):
    dataset = ClimateDataset(netcdf_dir, sequence_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)