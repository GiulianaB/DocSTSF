"""
Campaña STSF2013 - ADCP - Exploratorio
1. Cortes verticales secciones
"""
import f_ADCP_exploratorio as fADCP
import f_CTD_exploratorio as fCTD
import numpy as np

""" Cortes verticales """
#Cargo datos
DATA,lat_CTD,lon_CTD = fCTD.carga_datos_CTD()
U1,V1 = fADCP.datos_ADCP_cada_1m()
for seccion in [1,2,3,5,7]:
    fADCP.corte_seccion_along(seccion, U1,V1, DATA,suavizado = 'si')
fADCP.corte_seccion_cross(seccion, U1,V1, DATA,suavizado = 'si')

#Transporte en c/Transecta
for seccion in [1,2,3,5,7]:
    T = fADCP.transporte_seccion(DATA,seccion,U1,V1)

seccion = 1
#Transporte entre st1 y 5 [Sv]:
T = fADCP.transporte_seccion(DATA,seccion,U1,V1)
T_1_5 = np.nansum(T[:,:5])/1000000
T_6 = np.nansum(T[:200,5])/1000000
T_7 = np.nansum(T[:200,6])/1000000

seccion = 3
T = fADCP.transporte_seccion(DATA,seccion,U1,V1)
T_shelf = T[:,:7]
T_positivo = np.nansum(T_shelf[T_shelf > 0])/1000000
T_TW = np.nansum(T[:,:4])/1000000

seccion = 5
T = fADCP.transporte_seccion(DATA,seccion,U1,V1)
T_TW = np.nansum(T[:,:2])/1000000
T_shelf = T[:,3:]
T_41_37 = np.nansum(T_shelf[T_shelf<0])/1000000

seccion = 7
T = fADCP.transporte_seccion(DATA,seccion,U1,V1)
T_TW = np.nansum(T)/1000000

















###
