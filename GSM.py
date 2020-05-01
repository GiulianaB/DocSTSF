import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cmocean as cmo
path = '/media/giuliana/DE_BERDEN/Doctorado/Datos/batimetria/ETOPO1_Bed_g_gmt4.grd'
DATA = Dataset(path,'r')
DATA.set_auto_mask(False)


x = DATA['x'][:]
y = DATA['y'][:]
z = DATA['z'][:]

x_GSM = x[6850:7100]
y_GSM = y[2600:3000]
z_GSM = z[2600:3000,6850:7100]

interior = z_GSM > 0
z_GSM[interior] = 3000

#Presencia de la ostra Puelchana:
#bancos del sur:
LAT_ostra = [-42.0083,-42.0083,-42.0125,-42.0375,-42.04167,-42.04167,
            -42.04583,-42.04583,-42.05,-42.075,-42.0833,
            -42.09167,-42.09167,-42.19328]

LON_ostra = [-65.0636,-65.05805,-65.0578,-65.0525,-65.0475,-65.04472,
            -65.04556,-65.04278,-65.04028,-65.02875,-65.01556,
            -65.00972,-65.00694,-64.8]
#bancos del norte:
LAT_BR = np.array(np.mean([-40-46.59/60,-40-46.68/60,-40-47.46/60,-40-47.34/60]))
LON_BR = np.array(np.mean([-64-54.03/60,-64-54.12/60,-64-54.79/60,-64-55.01/60]))
LAT_ostra.append(LAT_BR); LON_ostra.append(LON_BR)

LAT_LG = np.array(np.mean([-40-49.68/60,-40-49.33/60,-40-49.50/60,-40-49.43/60]))
LON_LG = np.array(np.mean([-65-5.33/60 ,-65-4.99/60 ,-65-5.34/60 ,-65-5.44/60 ]))
LAT_ostra.append(LAT_LG); LON_ostra.append(LON_LG)

LAT_ES = np.array(np.mean([-40-56/60,-40-56/60,-41,-40-59.50/60]))
LON_ES = np.array(np.mean([-65-8.5/60,-65-6.4/60,-65-6/60,-65-8.5/60]))
LAT_ostra.append(LAT_ES); LON_ostra.append(LON_ES)

LAT_CLL = np.array(np.mean([-41-3.42/60,-41-3.97/60,-41-3.9/60,-41-5.86/60]))
LON_CLL = np.array(np.mean([-64-3.39/60,-64-3.76/60,-64-8.28/60,-64-8.28/60]))
LAT_ostra.append(LAT_CLL); LON_ostra.append(LON_CLL)

LAT_BO = np.array(np.mean([-40-52.5/60,-40-52.5/60,-40.55-5/60,-40-55.5/60]))
LON_BO = np.array(np.mean([-65-5/60,-65-3/60,-65-3/60,-65-5/60]))
LAT_ostra.append(LAT_BO); LON_ostra.append(LON_BO)
#            bahia anegada,     bahia blanca, golfo nuevo
LAT_GIGAS = [-40-11/60-11/3600, -38.8       ,  -42.7]
LON_GIGAS = [-62-13/60-15/3600, -62.2       , -65 ]

import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs

levels = [-200,-150,-100,-70,-50,-30,-9,0]
levels_iso = [-70,-30,-9]
for fig in range(1):
    plt.figure(figsize=(30,36))
    cmap=plt.cm.get_cmap('cmo.tempo_r', 40)
    ax = plt.axes(projection=ccrs.Mercator())

    BATH = ax.contourf(x_GSM,y_GSM,z_GSM,levels, cmap=cmap,vmin=-200, vmax=0,extend='both',transform=ccrs.PlateCarree())
    BATH_cont= ax.contour(x_GSM,y_GSM,z_GSM,levels_iso,colors= ('grey'),linewidht=1,transform=ccrs.PlateCarree())
    cbar = plt.colorbar(BATH,ticks=levels)
    cbar.ax.tick_params(labelsize=24)
    cbar.ax.set_yticklabels(['200', '150', '100','70','50','30','9','0'])
    cbar.ax.set_xlabel('Meters',fontsize=24)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
    linewidth=1, color='black', alpha=0.3, linestyle='-')
    gl.xlabels_top = False; gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator([-66,-65,-64,-63,-62,-61])
    gl.ylocator = mticker.FixedLocator([-43.5,-43,-42.5,-42,-41.5,-41,-40.5,-40])
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 24, 'color': 'k'}
    gl.ylabel_style = {'size': 24, 'color': 'k'}

    ax.add_feature(cfeature.LAKES, color='lightcyan')
    ax.add_feature(cfeature.RIVERS, edgecolor='black')
    ax.set_extent([-65.5,-61.8,-43.5,-40])
    ax.coastlines(resolution='10m', color='black', linewidth=2)

    plt.scatter(LON_ostra,LAT_ostra,s=100,color='r',label='Ostrea puelchana', transform=ccrs.PlateCarree(),zorder = 5)
    plt.scatter(LON_GIGAS,LAT_GIGAS,s=100,color='b',label='Cassostrea gigas', transform=ccrs.PlateCarree(),zorder = 5)
    plt.legend(loc=2,markerscale=1.5,prop={'style':'italic','size':26})

    #Separacion de areas
    plt.plot([-65,-64.2],[-41.5,-41.5],lw =2, c='orange',ls='--',transform=ccrs.PlateCarree(),zorder=4)
    plt.plot([-64.2,-64.2],[-41,-42.2],lw =2, c='orange',ls='--',transform=ccrs.PlateCarree(),zorder=4)
    plt.text(-64.7,-41.2,'NA',color='orange',size='26',transform=ccrs.PlateCarree())
    plt.text(-64.7,-41.8,'SA',color='orange',size='26',transform=ccrs.PlateCarree())
    plt.text(-64,-41.6,'MA',color='orange',size='26',transform=ccrs.PlateCarree())
    plt.show()

#mapa mas grande
for fig_grande in range(1):
    plt.figure(figsize=(30,36))
    cmap=plt.cm.get_cmap('cmo.tempo_r', 40)
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.plot([-65.5,-61.8],[-43.5,-43.5],color='k',transform = ccrs.PlateCarree())
    ax.plot([-65.5,-61.8],[-40.,-40.],color='k',transform = ccrs.PlateCarree())
    ax.plot([-65.5,-65.5],[-43.5,-40],color='k',transform = ccrs.PlateCarree())
    ax.plot([-61.8,-61.8],[-43.5,-40.],color='k',transform = ccrs.PlateCarree())

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
    linewidth=1, color='black', alpha=0.3, linestyle='-')
    gl.xlabels_top = False; gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator([-76,-72,-68,-64,-60,-56,-52])
    gl.ylocator = mticker.FixedLocator([-55,-50,-45,-40,-35,-32])
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 24, 'color': 'k'}
    gl.ylabel_style = {'size': 24, 'color': 'k'}

    ax.add_feature(cfeature.LAKES, color='lightcyan')
    ax.add_feature(cfeature.RIVERS, edgecolor='black')
    ax.add_feature(cfeature.LAND, edgecolor='#BDA975')
    ax.add_feature(cfeature.BORDERS, edgecolor='k')
    ax.set_extent([-76,-52,-55,-33])
    ax.coastlines(resolution='10m', color='black', linewidth=0.5)

    plt.scatter(-58-22/60,-34-36/60,s=100,color='k',label='Buenos Aires', transform=ccrs.PlateCarree())
    plt.text(-65.3,-34-36/60,'Buenos Aires', color = 'k', alpha=0.8,size = 22 )
    plt.text(-69,-40,'ARGENTINA', color = 'k',size = 24,rotation = 85 )
    plt.text(-73,-39,'CHILE', color = 'k',size = 24,rotation = 85 )

    plt.scatter(LON_ostra,LAT_ostra,s=100,color='r',label='Ostrea puelchana', transform=ccrs.PlateCarree(),zorder = 5)
    plt.scatter(LON_GIGAS,LAT_GIGAS,s=100,color='b',label='Cassostrea gigas', transform=ccrs.PlateCarree(),zorder = 5)


    plt.show()



# """
# COPERNICUS
# Climatologia
# """
# from netCDF4 import Dataset
# import netCDF4
# DATA = Dataset('/media/giuliana/TOSHIBA EXT/Masas materia/proyecto_GSM/global-reanalysis-phy-001-030-monthly_2010_2018.nc','r')
# DATA.set_auto_mask(False)
# lat = DATA.variables['latitude'][:]
# lon = DATA.variables['longitude'][:]      #va de 0:360
# time = np.array(DATA.variables['time'][:])
# u = DATA.variables['uo'][:,0,:,:]
# v = DATA.variables['vo'][:,0,:,:]
#
# u[np.abs(u)>15] = 0
# v[np.abs(v)>15] = 0
#
# # ssh = DATA.variables['zos'][:,:,:]  #altura respecto al geoide
# s = DATA.variables['so'][:,0,:,:]
# s[s < 0] = np.nan
#
# theta = DATA.variables['thetao'][:,0,:,:] # temp potencial
# theta[theta < 0] = np.nan
#
# T_fondo = DATA.variables['bottomT'][:,:,:]
# T_fondo[T_fondo < 0] = np.nan
#
# dates = []
# for t_ind in range(len(time)):
#     dates.append(netCDF4.num2date(time[t_ind], DATA['time'].units,DATA['time'].calendar))
#
# """verano"""
# for verano in range(1):
#     u_djf = np.nan * np.ones(u[0,:,:].shape)
#     v_djf = np.nan * np.ones(v[0,:,:].shape)
#     t_djf = np.nan * np.ones(T_fondo[0,:,:].shape)
#     s_djf = np.nan * np.ones(s[0,:,:].shape)
#     for i_lat in range(len(lat)):
#         for i_lon in range(len(lon)):
#             aux_u_djf = [];        aux_v_djf = []
#             aux_t_djf = [];        aux_s_djf = []
#             for anio in range(0,8):
#                 for mes in range(3):
#                     aux_u_djf.append(u[anio*12+mes,i_lat,i_lon])
#                     aux_v_djf.append(v[anio*12+mes,i_lat,i_lon])
#                     aux_t_djf.append(T_fondo[anio*12+mes,i_lat,i_lon])
#                     aux_s_djf.append(s[anio*12+mes,i_lat,i_lon])
#
#             u_djf[i_lat,i_lon] = np.mean(aux_u_djf)
#             v_djf[i_lat,i_lon] = np.mean(aux_v_djf)
#             t_djf[i_lat,i_lon] = np.mean(aux_t_djf)
#             s_djf[i_lat,i_lon] = np.mean(aux_s_djf)
#
# """invierno"""
# for invierno in range(1):
#     u_jja = np.nan * np.ones(u[0,:,:].shape)
#     v_jja = np.nan * np.ones(v[0,:,:].shape)
#     t_jja = np.nan * np.ones(T_fondo[0,:,:].shape)
#     s_jja = np.nan * np.ones(s[0,:,:].shape)
#     for i_lat in range(len(lat)):
#         for i_lon in range(len(lon)):
#             aux_u_jja = [];        aux_v_jja = []
#             aux_t_jja = [];        aux_s_jja = []
#             for anio in range(0,8):
#                 for mes in range(6,9):
#                     aux_u_jja.append(u[anio*12+mes,i_lat,i_lon])
#                     aux_v_jja.append(v[anio*12+mes,i_lat,i_lon])
#                     aux_t_jja.append(theta[anio*12+mes,i_lat,i_lon])
#                     aux_s_jja.append(s[anio*12+mes,i_lat,i_lon])
#
#             u_jja[i_lat,i_lon] = np.mean(aux_u_jja)
#             v_jja[i_lat,i_lon] = np.mean(aux_v_jja)
#             t_jja[i_lat,i_lon] = np.mean(aux_t_jja)
#             s_jja[i_lat,i_lon] = np.mean(aux_s_jja)
#
# """ Media temporal """
# u_mean = np.mean(u, axis = 0)
# v_mean = np.mean(v, axis = 0)
# t_mean = np.mean(theta, axis = 0)
# s_mean = np.mean(s, axis = 0)
#
# x, y = np.meshgrid(lon,lat)
# cmap_sal = plt.cm.get_cmap('Reds', 20)
# levels_s = np.linspace(33.5,34,11)
# cmap_temp = plt.cm.get_cmap('coolwarm',20)
# # levels_t = [13,13.5,14,14.5,15,15.5,16]
# for fig in range(1):
#     plt.figure(figsize=(30,36))
#     ax = plt.axes(projection=ccrs.Mercator())
#
#     gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#     linewidth=1, color='black', alpha=0.3, linestyle='-')
#     gl.xlabels_top = False; gl.ylabels_right = False
#     gl.xlocator = mticker.FixedLocator([-65.5,-65,-64.5,-64,-63.5,-63,-62.5])
#     gl.ylocator = mticker.FixedLocator([-42.5,-42,-41.5,-41,-40.5])
#     gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
#     gl.xlabel_style = {'size': 24, 'color': 'k'}
#     gl.ylabel_style = {'size': 24, 'color': 'k'}
#
#     ax.add_feature(cfeature.LAKES, color='lightcyan')
#     ax.add_feature(cfeature.RIVERS, edgecolor='black')
#     ax.set_extent([-65.5,-62.5,-42.5,-40.5])
#     ax.coastlines(resolution='10m', color='black', linewidth=2)
#
#     # cf = ax.contourf(x,y,s_mean, levels_s,cmap=cmap_sal, vmin=33.5, vmax=34,extend='both',transform=ccrs.PlateCarree())
#     cf = ax.contourf(x,y,t_djf, cmap=cmap_temp, vmin=9, vmax=20,extend='both',transform=ccrs.PlateCarree())
#     cbar = plt.colorbar(cf)
#     # c = ax.contour(cf,levels_s, colors='k')
#     cbar.ax.tick_params(labelsize=24)
#     cbar.ax.set_xlabel('°C',fontsize=24)
#
#     # plt.scatter(LON_ostra,LAT_ostra,s=100,color='r',label='O. puelchana', transform=ccrs.PlateCarree())
#     # plt.legend(loc=1,markerscale=1.5,fontsize=26)
#
#     Q = ax.quiver(x,y,u_djf*100,v_djf*100,color='k',alpha=0.7,transform=ccrs.PlateCarree())
#     ax.quiverkey(Q,0.1,0.9,10,r'$10 \frac{cm}{s}$', fontproperties={'size': 26})
#     plt.show()
#
#




















# xxx
