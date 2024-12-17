import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.patches as mpatches

# Charger le fichier NetCDF (remplacez le chemin par le vôtre)
file_path = r'data/raw/Test/data-2011-06-01-01-1_0.nc'
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
plt.xlabel("Longitude", fontsize=12, labelpad=20)  # Baisser l'étiquette "Longitude"
plt.ylabel("Latitude", fontsize=12, labelpad=40)   # Déplacer "Latitude" vers la gauche en augmentant labelpad

# Informations contextuelles et unités, positionnées pour éviter les superpositions
plt.figtext(0.5, 0.03, 
            "Data Source: ClimateNet Dataset | Variables Used: Total Water Vapor (TMQ), Zonal Wind (U850), Meridional Wind (V850), Sea Level Pressure (PSL)",
            ha="center", fontsize=9, color="grey")

plt.figtext(0.5, 0.01, 
            "Variables Units: TMQ: kg/m² (Total Water Vapor), U850 & V850: m/s (Wind Speed at 850 hPa), PSL: Pa (Sea Level Pressure)",
            ha="center", fontsize=9, color="grey")

# Affichage de la carte
plt.show()