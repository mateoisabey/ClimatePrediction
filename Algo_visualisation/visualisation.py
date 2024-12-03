import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.patches as mpatches
import os
from tqdm import tqdm

# Dossier d'entrée et de sortie
input_folder = r'D:\Bureau\DATA\Data_non_normaliser\deprecated'
output_folder = r'D:\Bureau\DATA\DATA_visualiser\deprecated'

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(output_folder, exist_ok=True)

# Obtenir la liste des fichiers .nc dans le dossier d'entrée
files = [f for f in os.listdir(input_folder) if f.endswith('.nc')]

# Boucle sur chaque fichier avec barre de chargement
for filename in tqdm(files, desc="Processing files", unit="file"):
    file_path = os.path.join(input_folder, filename)
    
    # Charger le fichier NetCDF
    data = xr.open_dataset(file_path, engine='netcdf4')
    
    # Sélectionner la variable LABELS pour représenter les événements
    labels = data['LABELS']
    
    # Extraire les coordonnées
    latitudes = labels['lat'].values
    longitudes = labels['lon'].values
    event_data = labels.values

    # Création de la carte du monde avec une hauteur ajustée
    plt.figure(figsize=(12, 10))
    m = Basemap(projection='cyl', resolution='c', llcrnrlat=-90, urcrnrlat=90,
                llcrnrlon=0, urcrnrlon=360)

    # Dessiner les contours et les côtes
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.5)
    m.drawparallels(np.arange(-90., 91., 30.), labels=[1,0,0,0], color="grey", linewidth=0.2)
    m.drawmeridians(np.arange(0., 361., 60.), labels=[0,0,0,1], color="grey", linewidth=0.2)

    # Convertir les coordonnées en meshgrid pour la carte
    lon, lat = np.meshgrid(longitudes, latitudes)
    x, y = m(lon, lat)

    # Tracer les événements climatiques détectés avec une palette de couleurs contrastées
    cs = m.contourf(x, y, event_data, cmap='plasma', levels=[0, 1, 2], extend='both')

    # Créer une légende colorée pour la gauche de la carte
    background_patch = mpatches.Patch(color=plt.cm.plasma(0.1), label="Background")
    river_patch = mpatches.Patch(color=plt.cm.plasma(0.5), label="Tropical Cyclone")
    cyclone_patch = mpatches.Patch(color=plt.cm.plasma(0.8), label="Atmospheric River")

    plt.legend(handles=[background_patch, river_patch, cyclone_patch], loc="upper left", 
               title="Legend", frameon=True, fancybox=True, framealpha=1, bbox_to_anchor=(0, 1))

    # Titre
    plt.title("Map of Extreme Climate Event Detections", fontsize=15)

    # Ajustement de la position des étiquettes "Longitude" et "Latitude"
    plt.xlabel("Longitude", fontsize=12, labelpad=20)
    plt.ylabel("Latitude", fontsize=12, labelpad=40)

    # Enregistrer la figure dans le dossier de sortie
    output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
    plt.savefig(output_file_path, bbox_inches='tight')
    plt.close()  # Fermer la figure pour éviter une surcharge de mémoire
