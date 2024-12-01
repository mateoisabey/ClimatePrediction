import netCDF4 as nc

# Charger le fichier .nc pour examiner sa structure et ses variables
file_path = "data/raw/Train/data-2000-02-21-01-1_0.nc"  # Remplacez par le chemin de votre fichier
dataset = nc.Dataset(file_path, 'r')

# Obtenir un aperçu des variables et des dimensions
dataset_summary = {
    "dimensions": {dim: dataset.dimensions[dim].size for dim in dataset.dimensions},
    "variables": {var: dataset.variables[var].shape for var in dataset.variables}
}

# Afficher un résumé des dimensions et des variables
print("Dimensions :")
for dim, size in dataset_summary["dimensions"].items():
    print(f"  - {dim}: {size}")

print("\nVariables :")
for var, shape in dataset_summary["variables"].items():
    print(f"  - {var}: {shape}")

# Optionnel : afficher des informations supplémentaires pour chaque variable
print("\nDétails des variables :")
for var_name in dataset.variables:
    var = dataset.variables[var_name]
    print(f"\nVariable : {var_name}")
    print(f"  Dimensions : {var.dimensions}")
    print(f"  Shape : {var.shape}")
    print(f"  Type : {var.dtype}")
    print(f"  Description : {getattr(var, 'long_name', 'Non défini')}")
    print(f"  Unités : {getattr(var, 'units', 'Non défini')}")

# Fermer le dataset une fois l'inspection terminée
dataset.close()