"""
Calculo de Transporte off-shore baroclinico usando datos de T, S.

1. Busco las estaciones entre las cuales voy a calcular el transporte baroclinico.
SECCION 1 _ st.5 = 151m
SECCION 3 _ st.22= 192m
SECCION 5 _ st.36= 149m
SECCION 7 _ st.49= 143m

2. Son 3 tramos:
A) entre st 5 y 22
B) entre st 22 y 32
C) entre 32 y 36
D) entre st 36 y 45
E) entre 45 y 49
"""
import f_transporte_200_insitu as ft
import seawater as sw
import matplotlib.pyplot as plt
import numpy as np
import f_CTD_exploratorio as fCTD

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


xi, yi = np.meshgrid(np.linspace(-60,-48,500),np.linspace(-40,-30,500))
fig = plt.figure()
ax = fig.add_axes([0.05,0.05,0.8,0.95],projection=ccrs.PlateCarree())

ax.set_extent([-60,-48,-40,-30])
ax.add_feature(cfeature.LAND, color='#BDA973')
ax.add_feature(cfeature.LAKES, color='lightcyan')
ax.add_feature(cfeature.RIVERS, edgecolor='black')
ax.coastlines(resolution='50m', color='black', linewidth=1)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='black', alpha=0.5, linestyle='-')
gl.xlabels_top = False; gl.ylabels_right = False
gl.xlocator = mticker.FixedLocator([-60,-58,-56,-54,-52,-50,-48])
gl.ylocator = mticker.FixedLocator([-40,-38,-36,-34,-32,-30])
gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 24, 'color': 'k'}
gl.ylabel_style = {'size': 24, 'color': 'k'}
lat_CTD,lon_CTD = fCTD.lat_lon_CTD()
ax.plot([lon_CTD[4],lon_CTD[21]],[lat_CTD[4],lat_CTD[21]],c='g',lw=3,ls='--')
ax.plot([lon_CTD[21],lon_CTD[31]],[lat_CTD[21],lat_CTD[31]],c='g',lw=3,ls='--')
ax.plot([lon_CTD[31],lon_CTD[35]],[lat_CTD[31],lat_CTD[35]],c='g',lw=3,ls='--')
ax.plot([lon_CTD[35],lon_CTD[49]],[lat_CTD[35],lat_CTD[49]],c='g',lw=3,ls='--')
ax.scatter(lon_CTD,lat_CTD,color='k',edgecolors ='white')

TRAMOS = ['A','B','C','D','E']
for TRAMO in TRAMOS:
    if TRAMO == 'A':
        sts = [5,22]
        fecha_inicial,fecha_final = '2013/10/03','2013/10/06'
    if TRAMO == 'B':
        sts = [22,32]
        fecha_inicial,fecha_final = '2013/10/06','2013/10/07'
    if TRAMO == 'C':
        sts = [32,36]
        fecha_inicial,fecha_final = '2013/10/07','2013/10/07'
    if TRAMO == 'D':
        sts = [36,45]
        fecha_inicial,fecha_final = '2013/10/07','2013/10/09'
    if TRAMO == 'E':
        sts = [45,49]
        fecha_inicial,fecha_final = '2013/10/07','2013/10/09'




    # Batimetria entre estaciones
    recta_lat,recta_lon,bat = ft.batimetria_recta_entre_st(sts,fuente='GEBCO')

    # Vel geostrofica relativa, usando perfiles T,S.
    #DUDA: que pasa con el fondo entre las estaciones.. hay partes qe parece ser mas somero.. (?)
    vg_rel = ft.f_vel_geo_baroclinica_entre_st(sts)
    # Vel geostrofica barotropica, derivada de anomalias de SSH (Copernicus)
    vg_ref,sla_1,sla_2 = ft.f_vel_barotropica_sup_entre_st(sts,fecha_inicial,fecha_final)
    #vel geostrofica:
    vg = vg_rel + vg_ref
    vg_mean = np.nanmean(vg)
    # Transporte geostrofico:
    Tg = ft.f_transporte_entre_st(sts,vg_mean)

    ## PLOT
    # z = np.linspace(0,5000,5000)
    # plt.plot(vg_rel,-z, label = 'Vg relativa')
    # plt.plot(vg,-z, label = 'Vg = relativa + referencia')
    # plt.grid()
    # plt.legend(fontsize=24)
    # plt.show()

    # Transporte Ekman
    lat_media = np.nanmean(recta_lat)
    W,u_mean,v_mean = ft.f_W_paralela_a_recta(fecha_inicial,fecha_final,recta_lon,recta_lat)
    Me_vol = ft.f_Me_vol(W,recta_lat,recta_lon,sts,rho_air = 1.225,Cd = 10**-3)


    TXT = 'T (Sv): '+str(Tg+Me_vol)[:5]
    ax.scatter(-30,-60,s=1,label=TXT)
    # ax.quiver(np.nanmean(recta_lon),np.nanmean(recta_lat),u_mean,v_mean)


# print('sts:',sts)
# print('sla1:',sla_1)
# print('sla2:',sla_2)
# print('W:',W)
# print('Me_vol:',Me_vol)
# print('Vg_ref:',vg_ref)
# print('vg_rel:',np.nanmean(vg_rel))
# print('Tg:',Tg)



fuentes = ['GEBCO','ETOPO','Palma','Topex']


Fig = ft.plot_transporte_paper(fuentes,est_CTD = True,isob_200=True,vel_ADCP=False,vel_OSCAR = True)



plt.legend()
plt.show()








##
