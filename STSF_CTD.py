"""
Campaña STSF2013 - CTD - Exploratorio
1. Mapa
2. Cortes verticales secciones
3. TS
4. Triangulo de Mamayev
"""

import numpy as np
import matplotlib.pyplot as plt
import f_CTD_exploratorio as fCTD
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import f_triangulo_mezcla as ftm
import f_ADCP_exploratorio as fADCP

DATA,lat_CTD,lon_CTD = fCTD.carga_datos_CTD()
#Regrillado c/1m.
T,S,O2,Fluo,z = fCTD.regrillado_vertical(DATA)

MAPA_BAT = fCTD.mapa_batimetrico(-40,-30,-60,-48,OSCAR = 'si')

""" 1. MAPA """
#Figuras de temperatura y salinidad:
prof = 0 #profundidad del mapa
T_sup_grillado = fCTD.mapa_temp(lat_CTD,lon_CTD,prof,T)
S_sup_grillado = fCTD.mapa_sal(lat_CTD,lon_CTD,prof,S)

U,V = fADCP.datos_ADCP_cada_1m()
Mapa_s_vel = fCTD.mapa_sal_vel(lat_CTD,lon_CTD,prof,S,U,V,seccion = 3)


""" 2. Cortes verticales """
# Temperatura:
for seccion in [1,2,3,4,5,6,7]:
    fCTD.corte_seccion(seccion,'T',T, DATA)
#Salinidad:
for seccion in [1,2,3,4,5,6,7]:
    fCTD.corte_seccion(seccion,'S',S, DATA)
# Saturacion de Oxigeno disuelto:
for seccion in [1,2,3,4,5,6,7]:
    fCTD.corte_seccion(seccion,'O2',O2, DATA)
# Fluorescencia:
for seccion in [1,2,3,4,5,6,7]:
    fCTD.corte_seccion(seccion,'Fluo',Fluo, DATA)

""" 3. TS """
estaciones_a_plotear = []
estaciones_a_plotear.append(['2','4','6','8'])
estaciones_a_plotear.append(['9','15','19'])
estaciones_a_plotear.append(['21','23','26','27'])
estaciones_a_plotear.append(['29','30','31','32'])
estaciones_a_plotear.append(['33','37','38','40'])
estaciones_a_plotear.append(['42','43','44','45'])
estaciones_a_plotear.append(['47','49','51','54'])
seccion = 2
fCTD.TS_plot(T[:,:],S[:,:], seccion, estaciones_a_plotear[seccion-1])

seccion = 1
fCTD.TS_plot(T[:,:],S[:,:], seccion, estaciones_a_plotear[seccion-1],'C')
seccion = 3
fCTD.TS_plot(T[:,:],S[:,:], seccion, estaciones_a_plotear[seccion-1],'F')
seccion = 5
fCTD.TS_plot(T[:,:],S[:,:], seccion, estaciones_a_plotear[seccion-1],'I')
seccion = 7
fCTD.TS_plot(T[:,:],S[:,:], seccion, estaciones_a_plotear[seccion-1],'L')


""" 4. Triangulo de Mamayev """
t_parametros = [22.12,7.6,15.02]
s_parametros = [36.77,33.8,27.73]

m_TW, m_SASW, m_RDP = ftm.porcentaje_triangulo_mamayev(T_sup_grillado,S_sup_grillado,t_parametros,s_parametros)

ftm.mapa_porcentaje(lat_CTD,lon_CTD,30,'RDP',m_RDP)
ftm.mapa_porcentaje(lat_CTD,lon_CTD,30,'TW',m_TW)
ftm.mapa_porcentaje(lat_CTD,lon_CTD,30,'SASW',m_SASW)
ftm.mapa_porcentaje_3masas(lat_CTD,lon_CTD,30,['     SASW','      TW','    RDP'],m_SASW,m_TW,m_RDP)

# Busqueda del frente STSF
profs = [30,40,50,100]
limite = 5
STSF = ftm.f_ubicacion_STSF(lon_CTD,lat_CTD,T,S,profs,limite,t_parametros = [22.12,7.6,15.02],s_parametros = [36.77,33.8,27.73])


""" PAPER """

MAPA_BAT = fCTD.mapa_batimetrico_paper(-40,-30,-60,-48,OSCAR = 'si')
seccion = 1
fCTD.corte_seccion_paper(seccion,'S',T,S, DATA,letra='A')
fCTD.corte_seccion_paper(seccion,'T',T,S, DATA,letra='B')
seccion = 3
fCTD.corte_seccion_paper(seccion,'S',T,S, DATA,letra='D')
fCTD.corte_seccion_paper(seccion,'T',T,S, DATA,letra='E')
seccion = 5
fCTD.corte_seccion_paper(seccion,'S',T,S, DATA,letra='G')
fCTD.corte_seccion_paper(seccion,'T',T,S, DATA,letra='H')
seccion = 7
fCTD.corte_seccion_paper(seccion,'S',T,S, DATA,letra='J')
fCTD.corte_seccion_paper(seccion,'T',T,S, DATA,letra='K')
seccion = 2
fCTD.corte_seccion_paper(seccion,'S',T,S, DATA,letra='A')
fCTD.corte_seccion_paper(seccion,'T',T,S, DATA,letra='B')









##
