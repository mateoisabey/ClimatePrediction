import netCDF4 as nc

file_path = "data/raw/Train/data-2000-02-21-01-1_0.nc" 
dataset = nc.Dataset(file_path, 'r')

dataset_summary = {
    "dimensions": {dim: dataset.dimensions[dim].size for dim in dataset.dimensions},
    "variables": {var: dataset.variables[var].shape for var in dataset.variables}
}

print("Dimensions :")
for dim, size in dataset_summary["dimensions"].items():
    print(f"  - {dim}: {size}")

print("\nVariables :")
for var, shape in dataset_summary["variables"].items():
    print(f"  - {var}: {shape}")

print("\nDétails des variables :")
for var_name in dataset.variables:
    var = dataset.variables[var_name]
    print(f"\nVariable : {var_name}")
    print(f"  Dimensions : {var.dimensions}")
    print(f"  Shape : {var.shape}")
    print(f"  Type : {var.dtype}")
    print(f"  Description : {getattr(var, 'long_name', 'Non défini')}")
    print(f"  Unités : {getattr(var, 'units', 'Non défini')}")

dataset.close()