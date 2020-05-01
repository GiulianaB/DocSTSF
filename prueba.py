""" Para jugar"""

from seabird import fCNV
import pandas as pd
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy.matlib
import math
import seawater as gsw
from matplotlib import patches
import cmocean as cmo
import netCDF4
from netCDF4 import Dataset
import f_CTD_exploratorio as fCTD

DATA,lat_CTD,lon_CTD = fCTD.carga_datos_CTD()

T,S,O2,Fluo,z = fCTD.regrillado_vertical(DATA)


seccion = 2
estaciones_a_plotear = '12'

ind_i,ind_f = 8,19   #indices dentro de la matriz_grillada
estaciones=['9' ,'10' ,'11' ,'12' ,'13' ,'14' ,'15' ,'16','17','18','19']

T_seccion = T[:,ind_i:ind_f]
S_seccion = S[:,ind_i:ind_f]
smin, smax,tmin, tmax = 27, 37,0, 27
# Calculate how many gridcells we need in the x and y dimensions
xdim = round((smax-smin)/0.1+1,0)
ydim = round((tmax-tmin)+1,0)
# Create empty grid of zeros
densidad = np.zeros((int(ydim),int(xdim)))
# Create temp and salt vectors of appropiate dimensions
ti = np.linspace(1,int(ydim)-1,int(ydim))+tmin
si = np.linspace(1,int(xdim)-1,int(xdim))*0.1+smin
# Loop to fill in grid with densities
for j in range(0,int(ydim)):
    for i in range(0, int(xdim)):
        densidad[j,i]=gsw.dens(si[i],ti[j],0)
# Substract 1000 to convert to sigma-t
densidad = densidad - 1000
colores = ['blue']
labels = '12'


fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.85,0.85])
ax.set_xlabel('Salinity', size=30)
ax.set_ylabel('Temperature (°C)', size=30)
plt.xticks(size = 30)
plt.yticks(size = 30)
ax.axis([27,37,0,27])
   #contornos de densidad
CS = ax.contour(si,ti,densidad, linestyles='dashed', colors='k',alpha=0.5)
ax.clabel(CS, fontsize=22, inline=1, fmt='%1.0f') # Label every second level

#nombre de las masas de agua
ax.text(30, 18.5, 'RDP',size=30, color='k')
ax.text(36.5, 19.7, 'TW',size=30,color='k')
ax.text(32.4, 9, 'SASW',size=30,color='k')
ax.arrow(33.1, 9.2, 0.3, 0, width=0.05,head_width=0.3, head_length=0.1,color='k' )
ax.text(35.6, 12, 'SACW',size=30,color='k')
ax.scatter(34.2,3.5, c='k',s=50,alpha=0.6)
ax.text(33.5,3.4,'AAIW',size=30,color='k')

ax.text(27.2,1,'B',size=30)

ax.scatter(S,T,s=1,c='grey',alpha=0.2)
#ploteo las estaciones:
cmap_z = plt.cm.get_cmap('gist_ncar', 16)
sc = ax.scatter(S_seccion[:400,3],T_seccion[:400,3],c = z[:400],cmap=cmap_z,s=50,label='12')
# sc = ax.scatter(S_seccion[:,8],T_seccion[:,8],c='k',s=50,label='17')
plt.legend()
cbar = plt.colorbar(sc)
ax.plot(S_seccion[:,3],T_seccion[:,3],c='b',lw= 1)
ax.plot(S_seccion[:,8],T_seccion[:,8],c='k',lw= 1)

plt.show()































##
