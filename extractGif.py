import os
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

# Dossier contenant les fichiers .nc
directory = 'data/raw/Test'

# Création de la figure et de la carte du monde avec une hauteur ajustée
fig, ax = plt.subplots(figsize=(12, 10))
m = Basemap(projection='cyl', resolution='c', llcrnrlat=-90, urcrnrlat=90,
            llcrnrlon=-180, urcrnrlon=180, ax=ax)

# Dessiner les contours et les côtes (ces éléments ne changent pas entre les frames)
m.drawcoastlines(linewidth=0.5)
m.drawcountries(linewidth=0.5)
m.drawparallels(np.arange(-90., 91., 30.), labels=[1,0,0,0], color="grey", linewidth=0.2)
m.drawmeridians(np.arange(-180., 181., 60.), labels=[0,0,0,1], color="grey", linewidth=0.2)

# Ajouter la légende (ces éléments ne changent pas entre les frames)
background_patch = mpatches.Patch(color=plt.cm.plasma(0.1), label="Background")
cyclone_patch = mpatches.Patch(color=plt.cm.plasma(0.5), label="Tropical Cyclone")
river_patch = mpatches.Patch(color=plt.cm.plasma(0.8), label="Atmospheric River")
plt.legend(handles=[background_patch, cyclone_patch, river_patch], loc="upper left", 
           title="Legend", frameon=True, fancybox=True, framealpha=1, bbox_to_anchor=(0, 1))

# Initialisation de la variable de contour
contour_plot = None

# Fonction pour mettre à jour l'animation
def update(frame):
    global contour_plot
    file_name = frame
    file_path = os.path.join(directory, file_name)
    
    # Charger le fichier NetCDF
    data = xr.open_dataset(file_path, engine='netcdf4')
    
    # Extraire la variable LABELS pour représenter les événements
    labels = data['LABELS']
    latitudes = labels['lat'].values
    longitudes = labels['lon'].values
    
    # Convertir les longitudes de 0-360 en -180 à 180 si nécessaire
    longitudes = np.where(longitudes > 180, longitudes - 360, longitudes)
    
    event_data = labels.values

    # Effacer le tracé précédent des événements climatiques, mais conserver les autres éléments
    if contour_plot:
        for c in contour_plot.collections:
            c.remove()

    # Convertir les coordonnées en meshgrid pour la carte
    lon, lat = np.meshgrid(longitudes, latitudes)
    x, y = m(lon, lat)

    # Tracer les événements climatiques détectés avec des couleurs spécifiées
    contour_plot = m.contourf(x, y, event_data, cmap='plasma', levels=[-0.5, 0.5, 1.5, 2.5], extend='both')

    # Mettre à jour le titre pour chaque frame
    ax.set_title(f"Map of Extreme Climate Event Detections - {file_name}")

# Liste des fichiers .nc dans le répertoire
file_list = [f for f in os.listdir(directory) if f.endswith('.nc')]

# Créer l'animation
ani = FuncAnimation(fig, update, frames=file_list, repeat=False)

# Sauvegarder l'animation en format GIF ou MP4
# Option 1 : Sauvegarder en GIF
ani.save("climate_event_animation.gif", writer="pillow", fps=2)

# Option 2 : Sauvegarder en MP4 (nécessite ffmpeg)
# ani.save("climate_event_animation.mp4", writer="ffmpeg", fps=2)

plt.show()