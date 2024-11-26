import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_event(data, title="Event Visualization"):
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    ax.imshow(data, transform=ccrs.PlateCarree(), origin='upper')
    ax.coastlines()
    ax.set_title(title)
    plt.show()

