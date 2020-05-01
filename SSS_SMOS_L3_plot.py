"""
SSS- SMOS L3, v2.0 2013/Oct
Cada archivo es el promedio de 9 dias centrado en la fecha que dice.
"""

#Librerias
import numpy as np
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
from netCDF4 import Dataset
from os import listdir
import f_CTD_exploratorio as fCTD
import xarray as xr



path = '/media/california/DE_BERDEN/Doctorado/Datos/satelital/Salinidad/SMOS/'
archivos =  listdir(path)

archivos_campania = []  # solo las imagenes centradas en 6 y 7 de octubre.
for archivo in archivos:
    if int(archivo[19:21]) in [6,7]:
        archivos_campania.append(archivo)

for file in archivos_campania:
    data_smos = xr.open_dataset(path + file).sel(lat = slice(-40.125,-29.875),lon = slice(-60.125,-47.875))
    if file == archivos_campania[0]:
        data = data_smos
    else:
        data = xr.merge([data,data_smos])
data = data.mean('time')

#DOMINIO

lat_CTD,lon_CTD = fCTD.lat_lon_CTD()
lon200,lat200 = fCTD.isobata_200()
cmap_s = plt.cm.get_cmap('Reds', 7)
levels = [30,31,32,33,34,35,36,37]

fig = plt.figure('Map_S_SMOS_7sOct2013')
ax = plt.axes(projection=ccrs.Mercator())
ax.set_extent([-60,-48,-40,-30],crs = ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, color='#BDA973')
ax.add_feature(cfeature.LAKES, color='lightcyan')
ax.add_feature(cfeature.RIVERS, edgecolor='black')
ax.coastlines(resolution='50m', color='black', linewidth=1)
gl = ax.gridlines(crs = ccrs.PlateCarree(),draw_labels=True,
linewidth=1, color='black', alpha=0.5, linestyle='-')
gl.xlabels_top = False; gl.ylabels_right = False
gl.xlocator = mticker.FixedLocator([-60,-58,-56,-54,-52,-50,-48])
gl.ylocator = mticker.FixedLocator([-40,-38,-36,-34,-32,-30])
gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 24, 'color': 'k'}
gl.ylabel_style = {'size': 24, 'color': 'k'}
cr = plt.contourf(data.lon,data.lat,data.hr_sss,levels,cmap=cmap_s,transform=ccrs.PlateCarree())
#barra de color
cbar = plt.colorbar(cr, orientation='vertical', ticks = levels)
cbar.ax.tick_params(labelsize=24)
plt.scatter(lon_CTD,lat_CTD,s = 20,color = 'k', transform=ccrs.PlateCarree())
plt.plot(lon200,lat200,color='grey', transform=ccrs.PlateCarree())




##
