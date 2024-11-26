import os
import numpy as np
from netCDF4 import Dataset

# Dictionnaire des statistiques (moyennes et écarts-types)
stats = {
    "TMQ": {"mean": 19.218493883511705, "std": 15.81727654092718},
    "U850": {"mean": 1.5530236518163378, "std": 8.297621410859342},
    "V850": {"mean": 0.2541317070842832, "std": 6.231631284217669},
    "UBOT": {"mean": 0.12487873639762347, "std": 6.653287200647422},
    "VBOT": {"mean": 0.31541635530239687, "std": 5.784189046243406},
    "QREFHT": {"mean": 0.007787778190110574, "std": 0.006215287649268782},
    "PS": {"mean": 96571.61180904522, "std": 9700.120187600089},
    "PSL": {"mean": 100814.0746789503, "std": 1461.0813611540402},
    "T200": {"mean": 213.2090852937285, "std": 7.889733635786149},
    "T500": {"mean": 253.03822411573202, "std": 12.825342082153885},
    "PRECT": {"mean": 2.945823891654793e-08, "std": 1.5563764223453562e-07},
    "TS": {"mean": 278.7114812161211, "std": 23.68249400173724},
    "TREFHT": {"mean": 278.4212252760475, "std": 22.51189231061532},
    "Z1000": {"mean": 474.1727845862705, "std": 832.8081734781491},
    "Z200": {"mean": 11736.104626139959, "std": 633.2586634526778},
    "ZBOT": {"mean": 61.31149022201575, "std": 4.909502525398299}
}

def normalize_variable(data, mean, std):
    return (data - mean) / std

def normalize_and_convert_to_npy(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".nc"):
            input_file = os.path.join(input_folder, filename)
            npy_output_file = os.path.join(output_folder, filename.replace(".nc", ".npy"))
            print(f"Traitement de {filename}...")
            
            # Conversion et normalisation directe vers .npy
            convert_nc_to_npy(input_file, npy_output_file)
    
    print("Conversion terminée pour tous les fichiers.")

def convert_nc_to_npy(nc_file, npy_file):
    with Dataset(nc_file, mode='r') as src:
        # Collecte des données normalisées dans un dictionnaire
        data_dict = {}
        for name, variable in src.variables.items():
            if name in stats:
                mean = stats[name]["mean"]
                std = stats[name]["std"]
                data_dict[name] = normalize_variable(variable[:], mean, std)
            else:
                data_dict[name] = variable[:]
        
        # Sauvegarde en .npy
        np.save(npy_file, data_dict)
        print(f"Fichier {npy_file} créé avec succès.")

# Exemple d'utilisation
input_folder = "data/raw/Train/part"
output_folder = "data/sample/Train"
normalize_and_convert_to_npy(input_folder, output_folder)