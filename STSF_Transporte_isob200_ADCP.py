"""
Calculo de Transporte off-shore sobre la linea de control usando datos de ADCP.

1. Defino linea control como la linea recta segun batimetria de fuente: 'ETOPO','GEBCO','Mati','STSF' o 'Topex'.
2. Busco batimetria sobre linea de control usando las fuentes, excepto para fuente 'STSF' que uso la batimetria de 'GEBCO'.
3. Abro datos ADCP y los regrillo c/10m o 1m.
4. Interpolacion sobre linea de control
5. Completo el perfil vertical extrapolando con metodo : 'lineal', 'cte' o 'ceros'
6. Calculo de transporte
7. Ploteo
"""


import f_CTD_exploratorio as fCTD
import f_transporte_200_insitu as ft_insitu
import f_ADCP_exploratorio as fADCP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


#Datos de campaña (estaciones):
DATA, lat_CTD,lon_CTD = fCTD.carga_datos_CTD()
fuentes = ['GEBCO','ETOPO','Palma','Topex']
fuente = 'Mati'     # 'GEBCO','ETOPO','Palma','Mati','Topex','STSF'
# lon_sur,lat_sur, lon_norte, lat_norte = ft_insitu.buscar_limites_linea_control(fuente)
#
# #DEFINO LINEA DE CONTROL:
# lat200 = np.linspace(lat_sur,lat_norte, 200)
# lon200 = np.linspace(lon_sur,lon_norte, 200)
#
#  #----------  Guardo la batimetria sobre la isob 200 para cada fuente
# if fuente == 'STSF':
#     z_g = ft_insitu.batimetria_sobre_transecta(lon200,lat200,fuente = 'ETOPO')
# else:
#     z_g = ft_insitu.batimetria_sobre_transecta(lon200,lat200,fuente = fuente)
# d = { 'fuente': fuente,'lon200': lon200, 'lat200': lat200, 'bat': z_g}
# df = pd.DataFrame(data = d)
# df.to_csv('/media/giuliana/DE_BERDEN/Doctorado/Scripts/Salidas_npy/Bat_linea_control_fuente_'+str(fuente)+'.csv')


Bat_linea_control = pd.read_csv('/media/giuliana/DE_BERDEN/Doctorado/Scripts/Salidas_npy/Bat_linea_control_fuente_'+str(fuente)+'.csv')
z_g = Bat_linea_control['bat']
lon200 = Bat_linea_control['lon200']
lat200 = Bat_linea_control['lat200']

""" Datos de Velocidad """
#Abro ADCP y acomodo los datos c/10m
depth = np.linspace(0,4999,5000)

U_1, V_1 = fADCP.datos_ADCP_cada_1m()

# Interpolo sobre la linea control:
u_g,v_g = ft_insitu.interpolacion_por_nivel_considerando_nans(lon_CTD,lat_CTD,U_1,V_1,lon200,lat200)

# Extrapolo linealmente hasta el fondo definido por gebco discretizado c/10m
u_g_lineal,v_g_lineal = ft_insitu.extrapolacion_lineal_hacia_el_fondo(u_g,v_g,lon200,lat200,z_g, metodo = 'lineal')

# Proyeccion sobre la linea control
Vf = ft_insitu.proyeccion_sobre_recta(lon200,lat200,u_g_lineal,v_g_lineal)

"""
TRANSPORTE
"""
#dist2: matriz 2d con las distancias entre ptos repetida
dist2 = ft_insitu.gen_dist_2d(lon200,lat200,depth)

#dz2 matriz 2d con las alturas de las cajitas
dz2 = ft_insitu.gen_dz_2d(lon200,lat200,depth)

T = Vf*dist2*dz2/100
# np.save('/media/giuliana/DE_BERDEN/Doctorado/Scripts/Salidas_npy/Transporte_'+str(fuente), T)

T_total = np.nansum(T)
print('T(Sv): '+str(fuente),T_total/1000000)

V_integrada= []
for k in range(0, len(Vf[0,:])):
    V_integrada.append(np.nanmean(Vf[:,k]))
levels_vel = [-30,-20,-10,0,10,20,30]

""" FIGURA"""

ft_insitu.plot_recta_fuente(lon200,lat200,fuente,T_total,V_integrada,est_CTD = True,isob_200=True)

plt.show()



T_neto     = []
T_offshore = []
T_inshore  = []
fuentes = ['GEBCO','ETOPO','Palma','Topex']
for fuente in fuentes:
     T = np.load('/media/giuliana/DE_BERDEN/Doctorado/Scripts/Salidas_npy/Transporte_'+str(fuente)+'.npy')
     T_neto.append(np.nansum(T)/1000000)
     T_offshore.append(np.nansum(T[T > 0])/1000000)
     T_inshore.append(np.nansum(T[T < 0])/1000000)

T_neto_mean,T_neto_std = np.mean(T_neto),np.std(T_neto)
T_offshore_mean,T_offshore_std = np.mean(T_offshore),np.std(T_offshore)
T_inshore_mean,T_inshore_std = np.mean(T_inshore),np.std(T_inshore)



""" Figura con todos las fuentes """
Fig = ft_insitu.plot_recta_fuentes(fuentes,est_CTD = True,isob_200=True, Transporte = 'neto',vel_ADCP=True)


# Estadisticas :
T_integrada = np.zeros((len(fuentes),T.shape[1]))
for count, fuente in enumerate(fuentes):
    T = np.load('/media/giuliana/DE_BERDEN/Doctorado/Scripts/Salidas_npy/Transporte_'+str(fuente)+'.npy')
    T_integrada[count,:] = np.nanmean(T,axis = 0)
T_integrada_media = np.nanmean(T_integrada,axis =0)


ft_insitu.plot_recta_fuente(lon200,lat200,fuente = 'Palma',T_total=T_neto_mean,T_integrada_media,est_CTD = True,isob_200=True)

























###
