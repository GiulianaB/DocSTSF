"""
DATOS SATELITALES .nc
1. Abro datos SATELITALES
2. Ploteo
"""
from netCDF4 import Dataset
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import pandas as pd
import numpy as np
import f_CTD_exploratorio as fCTD
import matplotlib.pyplot as plt
import cmocean
from glob import glob



lon200,lat200 = fCTD.isobata_200()
lat_CTD,lon_CTD =fCTD.lat_lon_CTD()

"""
CLOROFILA
"""
path_gral = '/media/giuliana/DE_BERDEN/'
path_gral = '/media/giuliana/Disco1TB/'

path = path_gral + 'Doctorado/Datos/satelital/MODIS_AQUA_clorofila_October2013/'

#########   Datos cada 8 dias  #########
    # files = ['8dias/A20132652013272.L3m_8D_CHL_chlor_a_4km.nc','8dias/A20132732013280.L3m_8D_CHL_chlor_a_4km.nc','8dias/A20132812013288.L3m_8D_CHL_chlor_a_4km.nc','/8dias/A20132892013296.L3m_8D_CHL_chlor_a_4km.nc','8dias/A20132972013304.L3m_8D_CHL_chlor_a_4km.nc']
    # DATA = Dataset(path+files[1])
    # date_start = DATA.time_coverage_start[:10]
    # date_end = DATA.time_coverage_end[:10]
    # lat_modis = DATA.variables['lat'][2850:3150]
    # lon_modis = DATA.variables['lon'][2800:3200]
    # chl_a = DATA.variables['chlor_a'][2850:3150, 2800:3200]
    # print(date)
    # DATA.close()

#########   Datos diarios  #########
#Todo octubre
files =['A2013274.L3m_DAY_CHL_chlor_a_4km.nc','A2013275.L3m_DAY_CHL_chlor_a_4km.nc','A2013276.L3m_DAY_CHL_chlor_a_4km.nc','A2013277.L3m_DAY_CHL_chlor_a_4km.nc','A2013278.L3m_DAY_CHL_chlor_a_4km.nc','A2013279.L3m_DAY_CHL_chlor_a_4km.nc','A2013280.L3m_DAY_CHL_chlor_a_4km.nc','A2013281.L3m_DAY_CHL_chlor_a_4km.nc','A2013282.L3m_DAY_CHL_chlor_a_4km.nc','A2013283.L3m_DAY_CHL_chlor_a_4km.nc','A2013284.L3m_DAY_CHL_chlor_a_4km.nc','A2013285.L3m_DAY_CHL_chlor_a_4km.nc','A2013286.L3m_DAY_CHL_chlor_a_4km.nc','A2013287.L3m_DAY_CHL_chlor_a_4km.nc','A2013288.L3m_DAY_CHL_chlor_a_4km.nc','A2013289.L3m_DAY_CHL_chlor_a_4km.nc','A2013290.L3m_DAY_CHL_chlor_a_4km.nc','A2013291.L3m_DAY_CHL_chlor_a_4km.nc','A2013292.L3m_DAY_CHL_chlor_a_4km.nc','A2013293.L3m_DAY_CHL_chlor_a_4km.nc','A2013294.L3m_DAY_CHL_chlor_a_4km.nc','A2013295.L3m_DAY_CHL_chlor_a_4km.nc','A2013296.L3m_DAY_CHL_chlor_a_4km.nc','A2013297.L3m_DAY_CHL_chlor_a_4km.nc','A2013298.L3m_DAY_CHL_chlor_a_4km.nc','A2013299.L3m_DAY_CHL_chlor_a_4km.nc','A2013300.L3m_DAY_CHL_chlor_a_4km.nc','A2013301.L3m_DAY_CHL_chlor_a_4km.nc','A2013302.L3m_DAY_CHL_chlor_a_4km.nc','A2013303.L3m_DAY_CHL_chlor_a_4km.nc','A2013304.L3m_DAY_CHL_chlor_a_4km.nc']
#7-8-9 de oct 2013:
files=files[6:9]
aux=0
for file in files:
    DATA = Dataset(path+file)
    date = DATA.time_coverage_start[:10]
    if file == files[0]:
        lat_modis = DATA.variables['lat'][2850:3150]
        lon_modis = DATA.variables['lon'][2800:3200]
        C = np.nan*np.ones((len(files),len(lat_modis),len(lon_modis)))
    chl_a = DATA.variables['chlor_a'][2850:3150, 2800:3200]
    print(date)
    DATA.close()
    chl_a[chl_a.mask == True] = np.nan
    C[aux,:,:] = chl_a
    aux=aux+1
C_mean = np.nanmean(C,axis = 0)

### PLOT
#DOMINIO
x,y = np.meshgrid(lon_modis,lat_modis)

vmin,vmax = 0,10
levels = np.linspace(0,10,101)
data = C_mean
# # Figure
import sys
sys.path.append(path_gral + 'Doctorado/Scripts/Barra_colores')
from cpt_convert import loadCPT # Import the CPT convert function
from matplotlib.colors import LinearSegmentedColormap # Linear interpolation for color maps
# Converts the CPT file to be used in Python
cpt = loadCPT(path_gral  + 'Doctorado/Scripts/Barra_colores/intento.cpt')
# Makes a linear interpolation with the CPT file
cmap = LinearSegmentedColormap('cpt', cpt,101)

fig = plt.figure()
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


# --- isobata
plt.plot(lon200,lat200,color='grey',transform=ccrs.PlateCarree())
data[data>10]=10
cr = plt.contourf(x,y,data,levels,cmap=cmap,vmin=0,vmax=10,transform=ccrs.PlateCarree())
# plt.contour(x,y, data,levels, colors=('k'),linewidths=1,alpha=0.1,transform=ccrs.PlateCarree())
# cr = plt.scatter(x,y,c = data,s=5,cmap=cmap,vmin=vmin,vmax=vmax,alpha=0.99,transform=ccrs.PlateCarree())
#barra de color
cbar = plt.colorbar(cr, orientation='vertical')
cbar.set_ticks( np.linspace(0,10,11))
cbar.set_label(label='$mg.m^{-3}$',fontsize=22)
cbar.ax.tick_params(labelsize=24)
fecha = 'Chl-a MODIS-Aqua\n2013-10-08'
ax.text(-59.5,-31.2,fecha,size=24,bbox = dict(boxstyle='round', facecolor='white', alpha=0.8),transform=ccrs.PlateCarree())

plt.scatter(lon_CTD,lat_CTD,s = 110,color = 'k', edgecolor = 'white',transform=ccrs.PlateCarree())
plt.show()


"""
SST:
"""
# Mismo codigo de colores uqe la imgaen sup de temp de la campania pero con mayor resolucion.
#
#
path = '/home/california/Documents/Doctorado/Datos/satelital/SST-ModisA/'

#7-8-9 de oct 2013:
aux=0
dias = [278,279,280,281,282,283]
for dia in dias:
    file = 'A2013'+str(dia)+'.L3m_DAY_NSST_sst_4km.nc'
    DATA = Dataset(path+file)
    date = DATA.time_coverage_start[:10]
    if dia == 278:
        lat_modis = DATA.variables['lat'][2850:3150]
        lon_modis = DATA.variables['lon'][2800:3200]
        SST_modis = np.nan*np.ones((len(dias),len(lat_modis),len(lon_modis)))
    sst = DATA.variables['sst'][2850:3150, 2800:3200]
    print(date)
    DATA.close()
    sst[sst.mask == True] = np.nan
    SST_modis[aux,:,:] = sst
    aux=aux+1
T_mean = np.nanmean(SST_modis,axis = 0)


### PLOT
#DOMINIO
x,y = np.meshgrid(lon_modis,lat_modis)

vmin,vmax = 0,10
levels = np.linspace(5,25,10+1)
data = T_mean
# # Figure
cmap_t = plt.cm.get_cmap('coolwarm', len(levels))

fig = plt.figure()
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

# --- isobata
ax.plot(lon200,lat200,color='grey', transform=ccrs.PlateCarree())
cr = ax.contourf(x,y,data,levels,cmap=cmap_t,extend = 'both', transform=ccrs.PlateCarree())
#barra de color
cbar = plt.colorbar(cr, orientation='vertical')
cbar.set_ticks(levels[::2])
cbar.set_label(label='$°C$',fontsize=22)
cbar.ax.tick_params(labelsize=24)
fecha = 'SST'
ax.text(-59.5,-31.2,fecha,size=24,bbox = dict(boxstyle='round', facecolor='white', alpha=0.8),transform=ccrs.PlateCarree())

plt.scatter(lon_CTD,lat_CTD,s = 110,color = 'k', edgecolor = 'white',transform=ccrs.PlateCarree())



plt.show()

"""
SSS. Falta
"""




###
