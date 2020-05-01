"""
SSS - Aquarius L2 - Trazas de los 3 bines.
"""
# librerias
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import h5py
from glob import glob
import f_CTD_exploratorio as fCTD
from scipy.interpolate import griddata

#########################################
def ls4(path, filtro=""):
    spath=path + filtro
    return glob(spath)
##########################################
# Salinidad para el periodo de 2013 entre 30-40S 48-60W
# hdf
path = '/home/california/Documents/Doctorado/Datos/satelital/salinidad/Sal_L2_Aquarius_2013/'
archivos =  listdir(path)
it = 0
DATA_s = h5py.File(path+archivos[it],'r')
# DATA_s.keys()

#Selecciono solo las pasadas durante la campania: 3/10/2013-10/10/2013
dia_inicial = 276
dia_final = dia_inicial + 7
dias = np.linspace(dia_inicial,dia_final,dia_final-dia_inicial+1)
archivos_campania = []
for dia in dias:
    filtro = '*2013'+str(int(dia))+'*'
    archivos_campania = archivos_campania + ls4(path,filtro = filtro)

#Mapa
DATA,lat_CTD,lon_CTD = fCTD.carga_datos_CTD()
T,S,O2,Fluo,z = fCTD.regrillado_vertical(DATA)

#DOMINIO
lon_min, lon_max = -60, -48
lat_min, lat_max = -40, -30
x = np.linspace(lon_min,lon_max,500)
y = np.linspace(lat_min,lat_max,500)
x,y = np.meshgrid(x,y)

cmap_s = plt.cm.get_cmap('Reds', 7)
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


#Agrego Termosal si estoy en superficie
z0=0
lon_TS,lat_TS,temp_TS,sal_TS = fCTD.carga_datos_termosal()
yy,yy_f = 25180-1700,25184+800
sal_TS[yy:yy_f] = 26
LONS, LATS = np.concatenate([lon_TS,lon_CTD]), np.concatenate([lat_TS,lat_CTD])
Ssup_aux =  np.concatenate([sal_TS,S[int(z0),:]])
""" Arreglo """
lat_arreglo1 = [-38 ,-34.8 ,-34.8 ,-34.8 ,-37   ,-36   ,-34  ,-32 ,-35.2,-36 ,-36  ,-36  ,-32.8,-37.8,-37  ,-36.5,-34.9]
lon_arreglo1 = [-56 ,-54   ,-54.5 ,-55   ,-57   ,-57   ,-51.5,-51 ,-51.8,-53 ,-52.9,-53.1,-51  ,-54  ,-55  ,-54.5,-52.6 ]
s_arreglo1 =   [33.8,28    ,28    ,28    ,np.nan,np.nan,32.5 ,25  ,35   ,32.5,33   ,33   ,28   ,34.8 ,33.8 ,33.8 , 31.5]
lat_arreglo2 = [-37.45 ,-37.35,-37.35   ,-36.8 ,-38   ,-33.04]
lon_arreglo2 = [-52.7,-52.75  ,-52.8    ,-52.7 ,-57   ,-50.16]
s_arreglo2 =   [34.5  ,33     ,34.5     ,33.5  , 33.8 , 35.5]
lat_arreglo = np.concatenate([lat_arreglo1,lat_arreglo2])
lon_arreglo = np.concatenate([lon_arreglo1,lon_arreglo2])
s_arreglo = np.concatenate([s_arreglo1,s_arreglo2])
LONS, LATS = np.concatenate([LONS,lon_arreglo]), np.concatenate([LATS,lat_arreglo])
Ssup = np.concatenate([Ssup_aux,s_arreglo])
lon,lat = LONS, LATS

#Interpolación de T en el dominio:
points = (lon,lat)
values = Ssup
S_grillada = griddata(points,values,(x,y))
data = fCTD.suavizado(S_grillada)
# # Figure

# --- isobata
lon200,lat200 = fCTD.isobata_200()
plt.plot(lon200,lat200,color='grey', transform=ccrs.PlateCarree())

levels = [30,31,32,33,34,35,36,37]
cr = plt.contourf(x,y,data,levels,cmap=cmap_s, transform=ccrs.PlateCarree())
plt.contour(x,y, data,levels, colors=('k'),linewidths=1, transform=ccrs.PlateCarree())
#barra de color
cbar = plt.colorbar(cr, orientation='vertical', ticks = levels)
cbar.ax.tick_params(labelsize=24)
plt.scatter(lon_CTD,lat_CTD,s = 20,color = 'k', transform=ccrs.PlateCarree())


#Aquarius L2
for it in range(len(archivos_campania)):
    DATA_s = h5py.File(archivos_campania[it],'r')
    beam_lat = DATA_s['Navigation']['beam_clat']
    beam_lon = DATA_s['Navigation']['beam_clon']
    SSS = DATA_s['Aquarius Data']['SSS'][:,:]
    SSS[SSS == -9.9990000e+03] = np.nan
    # SSS[SSS>36.8] = 36.5
    ax.scatter(beam_lon[:,0],beam_lat[:,0],c = SSS[:,0],cmap = cmap_s,vmin = 29,vmax = 38,transform = ccrs.PlateCarree())
    ax.scatter(beam_lon[:,1],beam_lat[:,1],c = SSS[:,1],cmap = cmap_s,vmin = 29,vmax = 38,transform = ccrs.PlateCarree())
    cs = ax.scatter(beam_lon[:,2],beam_lat[:,2],c = SSS[:,2],cmap = cmap_s,vmin = 29,vmax = 38,transform = ccrs.PlateCarree())
    DATA_s.close()
