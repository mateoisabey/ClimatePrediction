import xarray as xr
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

def prepare_data(input_dir="D:/Bureau/DATA/Data_non_normaliser/Train", 
                output_dir="D:/Bureau/DATA/Data_restructure",
                normalize=True,
                visualize=True):
    """
    Prépare les données en optimisant pour la détection des 3 classes
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    nc_files = [f for f in os.listdir(input_dir) if f.endswith('.nc')]
    
    # Statistiques pour le suivi
    class_counts = {0: 0, 1: 0, 2: 0}
    total_pixels = 0
    
    print("Traitement des fichiers NetCDF...")
    for file_name in tqdm(nc_files, desc="Processing files"):
        file_path = os.path.join(input_dir, file_name)
        
        try:
            # Lecture du fichier NetCDF
            ds = xr.open_dataset(file_path)
            
            # Création des features
            features = np.stack([
                ds.TMQ.values,     # Eau précipitable (important pour AR)
                ds.U850.values,    # Composante U du vent (important pour les cyclones)
                ds.V850.values,    # Composante V du vent (important pour les cyclones)
                ds.UBOT.values,    # Vent de surface
                ds.VBOT.values,    # Vent de surface
                ds.QREFHT.values,  # Humidité (important pour AR)
                ds.PS.values,      # Pression de surface (important pour les cyclones)
                ds.PSL.values,     # Pression niveau mer (important pour les cyclones)
                ds.T200.values,    # Température
                ds.T500.values,    # Température
                ds.PRECT.values,   # Précipitations
                ds.TS.values,      # Température de surface
                ds.TREFHT.values,  # Température référence
                ds.Z1000.values,   # Géopotentiel
                ds.Z200.values,    # Géopotentiel
                ds.ZBOT.values     # Hauteur du fond
            ], axis=-1)
            
            # Normalisation des features
            if normalize:
                for i in range(features.shape[-1]):
                    mean = np.mean(features[..., i])
                    std = np.std(features[..., i])
                    if std == 0:
                        print(f"Warning: std=0 pour la variable {i} dans {file_name}")
                    features[..., i] = (features[..., i] - mean) / (std + 1e-8)
            
            # Préparation des labels
            labels = ds.LABELS.values
            unique_labels = np.unique(labels)
            print(f"Labels uniques dans {file_name}: {unique_labels}")
            
            # Compte des classes
            for label in [0, 1, 2]:
                count = np.sum(labels == label)
                class_counts[label] += count
                total_pixels += count
            
            # Création des labels one-hot
            labels_one_hot = np.zeros((*labels.shape, 3))
            labels_one_hot[..., 0] = (labels == 0).astype(np.float32)  # Background
            labels_one_hot[..., 1] = (labels == 1).astype(np.float32)  # Cyclone tropical
            labels_one_hot[..., 2] = (labels == 2).astype(np.float32)  # Rivière atmosphérique

            # Sauvegarde
            output_features_path = os.path.join(output_dir, f'features_{file_name}.npy')
            output_labels_path = os.path.join(output_dir, f'labels_{file_name}.npy')
            
            np.save(output_features_path, features)
            np.save(output_labels_path, labels_one_hot)
            
            # Visualisation pour le premier fichier
            if visualize and file_name == nc_files[0]:
                plt.figure(figsize=(15, 5))
                
                plt.subplot(131)
                plt.imshow(labels_one_hot[..., 0], cmap='Blues')
                plt.title('Background')
                plt.colorbar()
                
                plt.subplot(132)
                plt.imshow(labels_one_hot[..., 1], cmap='Reds')
                plt.title('Cyclones Tropicaux')
                plt.colorbar()
                
                plt.subplot(133)
                plt.imshow(labels_one_hot[..., 2], cmap='Greens')
                plt.title('Rivières Atmosphériques')
                plt.colorbar()
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'labels_visualization.png'))
                plt.close()
            
            ds.close()
            
        except Exception as e:
            print(f"Erreur lors du traitement de {file_name}: {str(e)}")
            continue
    
    # Affichage des statistiques
    print("\nStatistiques des classes:")
    for class_id, count in class_counts.items():
        percentage = (count / total_pixels) * 100
        class_name = ['Background', 'Cyclone tropical', 'Rivière atmosphérique'][class_id]
        print(f"{class_name}: {count} pixels ({percentage:.2f}%)")

if __name__ == "__main__":
    prepare_data()