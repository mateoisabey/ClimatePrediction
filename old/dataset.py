import numpy as np

def check_npy_structure_and_content(npy_file, num_samples=5):
    # Charger le fichier .npy
    data = np.load(npy_file, allow_pickle=True).item()
    
    # Afficher les clés, les formes des données et quelques exemples de valeurs
    print("Structure et Contenu du fichier .npy :")
    for key, value in data.items():
        print(f"\nVariable: {key}")
        print(f"Shape: {value.shape}")
        print(f"Exemples de valeurs (jusqu'à {num_samples} échantillons) :")
        if value.ndim == 1:
            print(value[:num_samples])
        elif value.ndim == 2:
            print(value[:num_samples, :num_samples])
        elif value.ndim == 3:
            print(value[:num_samples, :num_samples, :num_samples])
        else:
            print("Dimension trop élevée pour un aperçu rapide.")

# Exemple d'utilisation
npy_file = "data/sample/Train/data-1996-10-03-01-1_1.npy"
check_npy_structure_and_content(npy_file)