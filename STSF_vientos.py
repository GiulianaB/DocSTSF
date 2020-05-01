""" Vientos ERA5 y Viento de Cross-Calibrated Multi-Platform (CCMP)
 """

# Librerias
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import xarray as xr
from glob import glob
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import f_CTD_exploratorio as fCTD
import f_transporte_200_insitu as ft
from os import listdir
import SCT_Mareografo as SCTM
import cartopy.io.shapereader as shpreader

###################################
path_gral = '/media/giuliana/Disco1TB/'

""" Funciones """
def ls4(path, filtro=""):
    spath=path + filtro
    return glob(spath)
def gen_archivo_viento():
    """
    Genera un xarray con datos diarios de viento ERA5 (u,v)

    Parametro de salida
    -------------------
    data_u,data_v: xarray de u,v con datos diarios de viento para el 2013
    """
    # u
    path_1 = path_gral + 'Doctorado/Datos/satelital/ERA5_2013/VIENTO_10/U10/'
    filtro = "*326827.VAR_10U.e5.oper.an.sfc.128_165_10u*"
    lista_arq = ls4(path_1,filtro)
    for path in lista_arq:
        DATA = xr.open_dataset(path).resample(time = '1D').mean()
        if path == lista_arq[0]: data = DATA
        else:
            data = xr.merge([data,DATA])
    data_u = data
    # v
    path_1 = path_gral + 'Doctorado/Datos/satelital/ERA5_2013/VIENTO_10/V10/'
    filtro = "*326827.VAR_10V.e5.oper.an.sfc.128_166_10v*"
    lista_arq = ls4(path_1,filtro)
    for path in lista_arq:
        DATA = xr.open_dataset(path).resample(time = '1D').mean()
        if path == lista_arq[0]: data = DATA
        else:
            data = xr.merge([data,DATA])
    data_v = data
    return data_u,data_v
def gen_archivo_msl():
    """
    Genera un xarray con datos diarios de msl

    Parametro de salida
    -------------------
    msl: xarray de msl con datos diarios para el 2013
    """
    # u
    path_1 = path_gral + 'Doctorado/Datos/satelital/ERA5_2013/MSL/'
    filtro = "*326827.MSL.e5.oper.an.sfc.128_151_msl.regn320sc*"
    lista_arq = ls4(path_1,filtro)
    for path in lista_arq:
        DATA = xr.open_dataset(path).resample(time = '1D').mean()
        if path == lista_arq[0]: data = DATA
        else:
            data = xr.merge([data,DATA])
    msl = data
    return msl
def gen_archivo_viento_CCMP():
    """
    Genera un xarray con datos diarios de viento CCMP

    Parametro de salida
    -------------------
    data: xarray de viento con datos diarios de viento para el 2013
    """
    # u
    path_1 = '/home/california/Documents/Doctorado/Datos/satelital/Wind/Cross_Calibrated_Multi_Platform_CCMP'
    lista_arq = listdir(path_1)
    for path in lista_arq:
        print(path[23:27])
        DATA = xr.open_dataset(path_1 +'/'+ path).sel(latitude = slice(-40.125,-29.875),longitude = slice(299.875,312.125))
        if path == lista_arq[0]: data = DATA
        else:
            data = xr.merge([data,DATA])
    # data = data.resample(time = '1D').mean()
    return data

def plot_mapa_viento(data_u,data_v,fechas):
    """
    Plot de mapa de vectores de viento para la fecha indicada en el
    dominio: 48-60W, 30-40S

    Parametro de entrada
    --------------------
    data_u,data_v: xarray de viento
    fechas: string. ['2013-01-01','2013-12-31']. Fecha a plotear.
    Si es la fecha inicial es la misma que la final solo plotea ese dia, sino
    hace la media entre esas fechas.

    Paramtro de salida
    ------------------
    plot
    """
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
    lon200,lat200 = fCTD.isobata_200()
    plt.plot(lon200,lat200,color='grey')
    # --- viento
    x_vel,y_vel = np.meshgrid(data_u.longitude.values,data_u.latitude.values)
    u = data_u.VAR_10U.sel(time = slice(fechas[0],fechas[1])).mean(dim = 'time').values
    v = data_v.VAR_10V.sel(time = slice(fechas[0],fechas[1])).mean(dim = 'time').values

    cs = ax.quiver(x_vel,y_vel,u,v,color='0.2',units='width',scale=160,transform = ccrs.PlateCarree())
    ax.quiverkey(cs, 0.4, 0.81, 0.5, r'$ 50\frac{m}{s}$', labelpos='E',coordinates='figure',fontproperties={'size' :24})

    # cr = plt.contourf(x,y,data,levels,cmap=cmap,vmin=0,vmax=10)
    # plt.contour(x,y, data,levels, colors=('k'),linewidths=1,alpha=0.1)
    # cr = plt.scatter(x,y,c = data,s=5,cmap=cmap,vmin=vmin,vmax=vmax,alpha=0.99)
    #barra de color
    # cbar = plt.colorbar(cr, orientation='vertical')
    # cbar.set_ticks( np.linspace(0,10,11))
    # cbar.set_label(label='$mg.m^{-3}$',fontsize=22)
    # cbar.ax.tick_params(labelsize=24)
    txt = 'Wind ERA5\n'+fechas[0]
    ax.text(-59.5,-31.2,txt,size=24,bbox = dict(boxstyle='round', facecolor='white', alpha=0.8),transform=ccrs.PlateCarree())
    #Estaciones de CTD
    lat_CTD,lon_CTD = fCTD.lat_lon_CTD()
    plt.scatter(lon_CTD,lat_CTD,s = 110,color = 'k', edgecolor = 'white',transform=ccrs.PlateCarree())
    plt.show()
def plot_mapa_MSL_viento(data_u,data_v,MSL,fecha):
    """
    Plot de mapa de vectores de viento para la fecha indicada en el
    dominio: 48-60W, 30-40S

    Parametro de entrada
    --------------------
    data_u,data_v:xarray con viento a 10m
    MSL: xarray de presion al nivel del mar.
    fechas: string. ['2013-01-01','2013-12-31']. Fecha a plotear.
    Si es la fecha inicial es la misma que la final solo plotea ese dia, sino
    hace la media entre esas fechas.

    Paramtro de salida
    ------------------
    plot
    """

    fig = plt.figure('MSL ERA5 '+ str(fecha))
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
    lon200,lat200 = fCTD.isobata_200()
    plt.plot(lon200,lat200,color='grey')
    # --- msl
    x,y = np.meshgrid(MSL.longitude.values,MSL.latitude.values)
    P = MSL.MSL.sel(time = fecha).values
    levels = np.linspace(100500,103500,31)
    cmap_presion = plt.cm.get_cmap('coolwarm', len(levels))
    cr = plt.contourf(x,y,P,levels,cmap=cmap_presion,vmin=levels[0],vmax=levels[-1],extend = 'both',transform=ccrs.PlateCarree())

    #barra de color
    cbar = plt.colorbar(cr, orientation='vertical')
    cbar.set_ticks( levels[::2])
    cbar.set_label(label='$Pa$',fontsize=22)
    cbar.ax.tick_params(labelsize=24)
    #viento
    x_vel,y_vel = np.meshgrid(data_u.longitude.values,data_u.latitude.values)
    u = data_u.VAR_10U.sel(time = fecha).values
    v = data_v.VAR_10V.sel(time = fecha).values
    cs = ax.quiver(x_vel[::2],y_vel[::2],u[::2],v[::2],color='0.2',units='width',scale=160,alpha = 0.8, transform = ccrs.PlateCarree())
    ax.quiverkey(cs, 0.9, 0.9, 10, r'$ 10\frac{m}{s}$', labelpos='E',coordinates='figure',fontproperties={'size' :24})

    txt = 'MSL ERA5\n'+fecha
    ax.text(-59.5,-31.2,txt,size=24,bbox = dict(boxstyle='round', facecolor='white', alpha=0.8),transform=ccrs.PlateCarree())
    #Estaciones de CTD
    lat_CTD,lon_CTD = fCTD.lat_lon_CTD()
    plt.scatter(lon_CTD,lat_CTD,s = 110,color = 'k', edgecolor = 'white',transform=ccrs.PlateCarree())
    plt.show()
def plot_mapa_ssh_viento(data_u,data_v,ssh,fecha):
    """
    Plot de mapa de vectores de viento para la fecha indicada en el
    dominio: 48-60W, 30-40S

    Parametro de entrada
    --------------------
    data_u,data_v: xarray con viento a 10m
    ssh: xarray de anomalia de la altura al nivel del mar (sla de multisatelite).
    fechas: string. ['2013-01-01','2013-12-31']. Fecha a plotear.
    Si es la fecha inicial es la misma que la final solo plotea ese dia, sino
    hace la media entre esas fechas.

    Paramtro de salida
    ------------------
    plot
    """

    fig = plt.figure('Viento ERA5 + SLA (multisatelite)'+ str(fecha))
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
    lon200,lat200 = fCTD.isobata_200()
    plt.plot(lon200,lat200,color='grey')
    # --- ssh
    x,y = np.meshgrid(ssh.longitude.values,ssh.latitude.values)
    levels = np.linspace(-0.7,0.7,15)
    cmap_sla = plt.cm.get_cmap('coolwarm', len(levels))
    cr = plt.contourf(x,y,ssh.sla.sel(time = fecha).values,levels,cmap=cmap_sla,vmin=levels[0],vmax=levels[-1],extend = 'both',transform=ccrs.PlateCarree())

    #barra de color
    cbar = plt.colorbar(cr, orientation='vertical')
    cbar.set_ticks( levels[::2])
    cbar.set_label(label='$m$',fontsize=22)
    cbar.ax.tick_params(labelsize=24)
    #viento
    x_vel,y_vel = np.meshgrid(data_u.longitude.values,data_u.latitude.values)
    u = data_u.VAR_10U.sel(time = fecha).values
    v = data_v.VAR_10V.sel(time = fecha).values
    cs = ax.quiver(x_vel[::2],y_vel[::2],u[::2],v[::2],color='0.2',units='width',scale=160,alpha = 0.8, transform = ccrs.PlateCarree())
    ax.quiverkey(cs, 0.75, 0.9, 10, r'$ 10\frac{m}{s}$', labelpos='E',coordinates='figure',fontproperties={'size' :24})

    txt = 'Viento (ERA5) + SLA \n'+fecha
    ax.text(-59.5,-31.2,txt,size=24,bbox = dict(boxstyle='round', facecolor='white', alpha=0.8),transform=ccrs.PlateCarree())
    #Estaciones de CTD
    lat_CTD,lon_CTD = fCTD.lat_lon_CTD()
    plt.scatter(lon_CTD,lat_CTD,s = 110,color = 'k', edgecolor = 'white',transform=ccrs.PlateCarree())


    lat_oyarvide = -35-6/60-1/3600
    lon_oyarvide = -57-7/60-39/3600
    plt.scatter(lon_oyarvide,lat_oyarvide,s = 50,color = 'r',transform = ccrs.PlateCarree())
    plt.text(lon_oyarvide-1,lat_oyarvide+1,'Torre Oyarvide',c = 'r',size = 20,bbox = dict(boxstyle='round', facecolor='white', alpha=0.8),transform = ccrs.PlateCarree())

    plt.show()
###################################
data_u,data_v = gen_archivo_viento()
# data_CCMP = gen_archivo_viento_CCMP()
# data_CCMP.to_netcdf('/home/california/Documents/Doctorado/Datos/satelital/Wind/Cross_Calibrated_Multi_Platform_CCMP/Wind_CCMP_2013.nc')
data_CCMP = xr.open_dataset(path_gral + 'Doctorado/Datos/satelital/Wind/Cross_Calibrated_Multi_Platform_CCMP/Wind_CCMP_2013_6h.nc')
# fechas = ['2013-10-03','2013-10-04','2013-10-05','2013-10-06','2013-10-07','2013-10-08','2013-10-09','2013-10-10']
# for fecha in fechas:
#     plot_mapa_viento(data_u,data_v,fecha)
# # MSL(pressure) + vientos era5
# msl = gen_archivo_msl()
# fechas = ['2013-09-21','2013-09-22','2013-09-23','2013-09-24','2013-09-25','2013-09-26','2013-09-27','2013-09-28','2013-09-29','2013-09-30','2013-10-01','2013-10-02','2013-10-03']
# for fecha in fechas:
#     plot_mapa_MSL(data_u,data_v,msl,fecha)
# # SSH + vientos era 5
path = path_gral + 'Doctorado/Datos/satelital/Altimetria_multisatelite/dataset-duacs-rep-global-merged-allsat-phy-l4_1575848408082.nc'
data_ssh = xr.open_dataset(path)
# for fecha in fechas:
#     plot_mapa_ssh_viento(data_u,data_v,data_ssh,fecha)

""" MAPA + SERIE DE TIEMPO"""
### Proyeccion sobre el eje del RDP
lat0,lat1 = -36.67 , -34.43
lon0,lon1 = -58 + 360, -54 + 360
u_rdp_ERA5 = np.nanmean(data_u.VAR_10U.sel(latitude = slice(lat1,lat0),longitude = slice(lon0,lon1)).values,axis = (1,2))
v_rdp_ERA5 = np.nanmean(data_v.VAR_10V.sel(latitude = slice(lat1,lat0),longitude = slice(lon0,lon1)).values,axis = (1,2))
u_rdp_CCMP = np.nanmean(data_CCMP.uwnd.sel(latitude = slice(lat0,lat1),longitude = slice(lon0,lon1)).values,axis = (1,2))
v_rdp_CCMP = np.nanmean(data_CCMP.vwnd.sel(latitude = slice(lat0,lat1),longitude = slice(lon0,lon1)).values,axis = (1,2))
# Vparalela positiva hacia oceano abierto
lon_rdp = [(-56.78-54.93)/2,-58.39]
lat_rdp = [(-36.29-34.98)/2,-34.21]
Vparalela_ERA5 = ft.f_proyeccion_paralela_sobre_recta(lon_rdp,lat_rdp,u_rdp_ERA5,v_rdp_ERA5)
Vparalela_CCMP = ft.f_proyeccion_paralela_sobre_recta(lon_rdp,lat_rdp,u_rdp_CCMP,v_rdp_CCMP)
t,vp_ERA5 =    data_u.time.values[240:290],Vparalela_ERA5[240:290]
t,vp_CCMP = data_CCMP.time.values[240:290],Vparalela_CCMP[240:290]

sla_3_10 = data_ssh.sla.sel(time = '2013-10-03')

# Mapa + wind era5 y CCMP
for mapa in range(1):
    fig = plt.figure()
    ax = fig.add_subplot(121, projection = ccrs.Mercator())
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
    gl.xlabel_style = {'size': 20, 'color': 'k'}
    gl.ylabel_style = {'size': 20, 'color': 'k'}
    x,y = np.meshgrid(data_ssh.longitude.values,data_ssh.latitude.values)
    levels = np.linspace(-0.7,0.7,15)
    cmap_sla = plt.cm.get_cmap('coolwarm', len(levels))
    cr = plt.contourf(x,y,data_ssh.sla.sel(time = '2013-10-03').values,levels,cmap=cmap_sla,vmin=levels[0],vmax=levels[-1],extend = 'both',transform=ccrs.PlateCarree())
    #barra de color
    cbar = plt.colorbar(cr, orientation='horizontal')
    cbar.set_ticks( levels[::2])
    cbar.set_label(label='$SLA [m]$',fontsize=22)
    cbar.ax.tick_params(labelsize=24)
    # Rectangulo azul
    plt.plot([lon0,lon0],[lat0,lat1],ls = '--',c = 'blue',lw = 2,transform=ccrs.PlateCarree())
    plt.plot([lon1,lon1],[lat0,lat1],ls = '--',c = 'blue',lw = 2,transform=ccrs.PlateCarree())
    plt.plot([lon0,lon1],[lat0,lat0],ls = '--',c = 'blue',lw = 2,transform=ccrs.PlateCarree())
    plt.plot([lon0,lon1],[lat1,lat1],ls = '--',c = 'blue',lw = 2,transform=ccrs.PlateCarree())
    plt.plot(lon_rdp,lat_rdp, ls = '--',c = 'red',lw = 2,transform=ccrs.PlateCarree())
    plt.arrow(np.mean(lon_rdp),np.mean(lat_rdp),(lon_rdp[0]-lon_rdp[1])/3,(lat_rdp[0]-lat_rdp[1])/3,color = 'red',head_width = .5,lw = 0.5,transform=ccrs.PlateCarree())
    # --- isobata
    lon200,lat200 = fCTD.isobata_200()
    plt.plot(lon200,lat200,color='grey',transform=ccrs.PlateCarree())
    #Estaciones de CTD
    lat_CTD,lon_CTD = fCTD.lat_lon_CTD()
    plt.scatter(lon_CTD,lat_CTD,s = 110,color = 'k', edgecolor = 'white',transform=ccrs.PlateCarree())

    #Series de tiempo de viento
    # ERA5
    ax1 = fig.add_axes([0.55,.60,.4,.3])
    plt.title('Offshore wind (ERA5)',fontsize = 24)
    ax1.axis([t[0],t[-1],-10,10])
    ax1.plot(t,vp_ERA5,label = 'Offshore wind (ERA5)')
    ax1.plot([t[35],t[35]],[-10,10],c = 'red',lw = 3)
    ax1.fill_between(t,vp_ERA5,where=vp_ERA5>=0, color='red',alpha = 0.5)
    ax1.fill_between(t,vp_ERA5,where=vp_ERA5<=0, color='blue',alpha = 0.5)
    ax1.set_ylabel('m/s',fontsize = 24)
    plt.xticks(t[::5],size = 0,rotation = 90)
    plt.yticks(size = 20)
    plt.grid()
    # CCMP
    ax2 = fig.add_axes([0.55,.20,.4,.3])
    plt.title('Offshore wind (CCMP)',fontsize = 24)
    ax2.axis([t[0],t[-1],-10,10])
    ax2.plot(t,vp_CCMP,label = 'Offshore wind (CCMP)')
    ax2.plot([t[35],t[35]],[-10,10],c = 'red',lw = 3)
    ax2.fill_between(t,vp_CCMP,where=vp_CCMP>=0, color='red',alpha = 0.5)
    ax2.fill_between(t,vp_CCMP,where=vp_CCMP<=0, color='blue',alpha = 0.5)
    ax2.set_ylabel('m/s',fontsize = 24)
    plt.xticks(t[::5],size = 20,rotation = 90)
    plt.yticks(size = 20)
    plt.grid()


##################################################################
##### INTENTO 2 ##################################################
###################################################################
# Caja sobre la paltaforma
# caja 1:
lat0,lat1 = -38.72,-36.5
lon0,lon1 = -57.55 + 360,-55. + 360

u_shelf_ERA5 = np.nanmean(data_u.VAR_10U.sel(latitude = slice(lat1,lat0),longitude = slice(lon0,lon1)).values,axis = (1,2))
v_shelf_ERA5 = np.nanmean(data_v.VAR_10V.sel(latitude = slice(lat1,lat0),longitude = slice(lon0,lon1)).values,axis = (1,2))
u_shelf_CCMP = np.nanmean(data_CCMP.uwnd.sel(latitude = slice(lat0,lat1),longitude = slice(lon0,lon1)).values,axis = (1,2))
v_shelf_CCMP = np.nanmean(data_CCMP.vwnd.sel(latitude = slice(lat0,lat1),longitude = slice(lon0,lon1)).values,axis = (1,2))
# Vparalela positiva hacia el noreste
lon_shelf = [-56.25,-57.93]
lat_shelf = [-38,-39.4]
Vparalela_ERA5 = ft.f_proyeccion_paralela_sobre_recta(lon_shelf,lat_shelf,u_shelf_ERA5,v_shelf_ERA5)
Vparalela_CCMP = ft.f_proyeccion_paralela_sobre_recta(lon_shelf,lat_shelf,u_shelf_CCMP,v_shelf_CCMP)
t,vp_ERA5 =    data_u.time.values[240:290],Vparalela_ERA5[240:290]
t,vp_CCMP = data_CCMP.time.values[240:290],Vparalela_CCMP[240:290]

sla_3_10 = data_ssh.sla.sel(time = '2013-10-03')

# Mapa + wind era5 y CCMP
for mapa in range(1):
    fig = plt.figure()
    ax = fig.add_subplot(121, projection = ccrs.Mercator())
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
    gl.xlabel_style = {'size': 20, 'color': 'k'}
    gl.ylabel_style = {'size': 20, 'color': 'k'}
    x,y = np.meshgrid(data_ssh.longitude.values,data_ssh.latitude.values)
    levels = np.linspace(-0.7,0.7,15)
    cmap_sla = plt.cm.get_cmap('coolwarm', len(levels))
    cr = plt.contourf(x,y,data_ssh.sla.sel(time = '2013-10-03').values,levels,cmap=cmap_sla,vmin=levels[0],vmax=levels[-1],extend = 'both',transform=ccrs.PlateCarree())
    #barra de color
    cbar = plt.colorbar(cr, orientation='horizontal')
    cbar.set_ticks( levels[::2])
    cbar.set_label(label='$SLA [m]$',fontsize=22)
    cbar.ax.tick_params(labelsize=24)
    # Rectangulo azul
    plt.plot([lon0,lon0],[lat0,lat1],ls = '--',c = 'blue',lw = 2,transform=ccrs.PlateCarree())
    plt.plot([lon1,lon1],[lat0,lat1],ls = '--',c = 'blue',lw = 2,transform=ccrs.PlateCarree())
    plt.plot([lon0,lon1],[lat0,lat0],ls = '--',c = 'blue',lw = 2,transform=ccrs.PlateCarree())
    plt.plot([lon0,lon1],[lat1,lat1],ls = '--',c = 'blue',lw = 2,transform=ccrs.PlateCarree())
    plt.plot(lon_shelf,lat_shelf, ls = '--',c = 'red',lw = 2,transform=ccrs.PlateCarree())
    plt.arrow(np.mean(lon_shelf),np.mean(lat_shelf),(lon_shelf[0]-lon_shelf[1])/3,(lat_shelf[0]-lat_shelf[1])/3,color = 'red',head_width = .5,lw = 0.5,transform=ccrs.PlateCarree())
    # --- isobata
    lon200,lat200 = fCTD.isobata_200()
    plt.plot(lon200,lat200,color='grey',transform=ccrs.PlateCarree())
    #Estaciones de CTD
    lat_CTD,lon_CTD = fCTD.lat_lon_CTD()
    plt.scatter(lon_CTD,lat_CTD,s = 110,color = 'k', edgecolor = 'white',transform=ccrs.PlateCarree())

    #Series de tiempo de viento
    # ERA5
    ax1 = fig.add_axes([0.55,.60,.4,.3])
    plt.title('Offshore wind (ERA5)',fontsize = 24)
    ax1.axis([t[0],t[-1],-10,10])
    ax1.plot(t,vp_ERA5,label = 'Offshore wind (ERA5)')
    ax1.plot([t[35],t[35]],[-10,10],c = 'red',lw = 3)
    ax1.fill_between(t,vp_ERA5,where=vp_ERA5>=0, color='red',alpha = 0.5)
    ax1.fill_between(t,vp_ERA5,where=vp_ERA5<=0, color='blue',alpha = 0.5)
    ax1.set_ylabel('m/s',fontsize = 24)
    plt.xticks(t[::5],size = 0,rotation = 90)
    plt.yticks(size = 20)
    plt.grid()
    # CCMP
    ax2 = fig.add_axes([0.55,.20,.4,.3])
    plt.title('Offshore wind (CCMP)',fontsize = 24)
    ax2.axis([t[0],t[-1],-10,10])
    ax2.plot(t,vp_CCMP,label = 'Offshore wind (CCMP)')
    ax2.plot([t[35],t[35]],[-10,10],c = 'red',lw = 3)
    ax2.fill_between(t,vp_CCMP,where=vp_CCMP>=0, color='red',alpha = 0.5)
    ax2.fill_between(t,vp_CCMP,where=vp_CCMP<=0, color='blue',alpha = 0.5)
    ax2.set_ylabel('m/s',fontsize = 24)
    plt.xticks(t[::5],size = 20,rotation = 90)
    plt.yticks(size = 20)
    plt.grid()


############################################################################################################################
# PAPER
############################################################################################################################

# San Clemente del Tuyu
SLA_SCT = SCTM.altura_San_Clemente_ASO()
t_SCT   = SLA_SCT.index
sla_SCT = SLA_SCT['Obs'] - SLA_SCT['predicha']
lat_SCT = -36-21/60-17/3600
lon_SCT = -56-42/60-54/3600


# Rectangulo
lat0,lat1 = -38.72,-36.5
lon0,lon1 = -57.55 + 360,-55. + 360

u_shelf_CCMP = np.nanmean(data_CCMP.uwnd.sel(latitude = slice(lat0,lat1),longitude = slice(lon0,lon1)).values,axis = (1,2))
v_shelf_CCMP = np.nanmean(data_CCMP.vwnd.sel(latitude = slice(lat0,lat1),longitude = slice(lon0,lon1)).values,axis = (1,2))
# Vparalela positiva hacia el noreste
lon_shelf = [-56.25,-57.93]
lat_shelf = [-38,-39.4]
Vparalela_CCMP = ft.f_proyeccion_paralela_sobre_recta(lon_shelf,lat_shelf,u_shelf_CCMP,v_shelf_CCMP)
t,vp_CCMP = data_CCMP.time.values[972:1148],Vparalela_CCMP[972:1148]

sla_3_10 = data_ssh.sla.sel(time = '2013-10-03')

# Mapa + wind CCMP y San Clemente del Tuyus
for mapa in range(1):
    fig = plt.figure()
    # ax = fig.add_subplot(121, projection = ccrs.Mercator())
    ax = fig.add_axes([0.04,0.17,0.45,.8],projection = ccrs.Mercator())
    ax.set_extent([-60,-48,-40,-30],crs = ccrs.PlateCarree())
    x,y = np.meshgrid(data_ssh.longitude.values,data_ssh.latitude.values)
    levels = np.linspace(-0.7,0.7,15)
    cmap_sla = plt.cm.get_cmap('coolwarm', len(levels))
    cr = plt.contourf(x,y,data_ssh.sla.sel(time = '2013-10-03').values,levels,cmap=cmap_sla,vmin=levels[0],vmax=levels[-1],extend = 'both',transform=ccrs.PlateCarree())
    #barra de color
    cax_a = fig.add_axes([0.06,0.08,0.4,0.03])
    cbar = plt.colorbar(cr, orientation='horizontal',cax = cax_a)
    cbar.set_ticks( levels[::2])
    cbar.set_label(label='$SLA [m]$',fontsize=22)
    cbar.ax.tick_params(labelsize=24)

    shpfilename = shpreader.natural_earth(resolution='10m',category='cultural',name='admin_0_countries')
    reader = list(shpreader.Reader(shpfilename).geometries())
    ax.add_geometries(reader, ccrs.PlateCarree(), edgecolor = 'black',facecolor = '#BDA973', linewidth=0.4)
    ax.add_feature(cfeature.LAKES, color='lightcyan')
    ax.add_feature(cfeature.RIVERS, edgecolor='black')
    ax.coastlines(resolution='50m', color='black', linewidth=1)
    gl = ax.gridlines(crs = ccrs.PlateCarree(),draw_labels=True,
    linewidth=1, color='black', alpha=0.5, linestyle='-')
    gl.xlabels_top = False; gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator([-60,-58,-56,-54,-52,-50,-48])
    gl.ylocator = mticker.FixedLocator([-40,-38,-36,-34,-32,-30])
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 20, 'color': 'k'}
    gl.ylabel_style = {'size': 20, 'color': 'k'}
    # --- isobata
    lon200,lat200 = fCTD.isobata_200()
    plt.plot(lon200,lat200,color='grey',transform=ccrs.PlateCarree())
    #Estaciones de CTD
    lat_CTD,lon_CTD = fCTD.lat_lon_CTD()
    plt.scatter(lon_CTD,lat_CTD,s = 110,color = 'k', edgecolor = 'white',transform=ccrs.PlateCarree())
    # San Clemente del Tuyu
    plt.scatter(lon_SCT,lat_SCT,s = 110,color = 'darkred',transform=ccrs.PlateCarree())
    # Rectangulo azul
    plt.plot([lon0,lon0],[lat0,lat1],ls = '--',c = 'blue',lw = 2,transform=ccrs.PlateCarree())
    plt.plot([lon1,lon1],[lat0,lat1],ls = '--',c = 'blue',lw = 2,transform=ccrs.PlateCarree())
    plt.plot([lon0,lon1],[lat0,lat0],ls = '--',c = 'blue',lw = 2,transform=ccrs.PlateCarree())
    plt.plot([lon0,lon1],[lat1,lat1],ls = '--',c = 'blue',lw = 2,transform=ccrs.PlateCarree())
    plt.plot(lon_shelf,lat_shelf, ls = '--',c = 'red',lw = 2,transform=ccrs.PlateCarree())
    plt.arrow(np.mean(lon_shelf),np.mean(lat_shelf),(lon_shelf[0]-lon_shelf[1])/3,(lat_shelf[0]-lat_shelf[1])/3,color = 'red',head_width = .5,lw = 0.5,transform=ccrs.PlateCarree())
    ax.text(-59.5,-39.5,'A)', fontsize = 24,transform = ccrs.PlateCarree())


    #Series de tiempo de viento y sla en San Clemente de Tuyu
    ax1 = fig.add_axes([0.58,.60,.4,.3])
    plt.title('Sea Level Anomaly - San Clemente del Tuyu',fontsize = 24)
    ax1.axis([t[0],t[-1],-1.,1.])
    ax1.plot([t_SCT[1504],t_SCT[1504]],[-1.,1.],c = 'red',lw = 3)
    ax1.plot(t_SCT,sla_SCT,label = 'SLA San Clemente del Tuyu')
    ax1.fill_between(t_SCT,sla_SCT,where=sla_SCT>=0, color='red',alpha = 0.5)
    ax1.fill_between(t_SCT,sla_SCT,where=sla_SCT<=0, color='blue',alpha = 0.5)
    ax1.set_ylabel('m',fontsize = 24)
    plt.xticks(t_SCT[784:1648:144],size = 0,rotation = 90)
    plt.yticks([-1,-.5,0,.5,1],size = 20)
    plt.grid()
    ax1.text(t[1],-0.9,'B)', fontsize = 24)
    # CCMP
    ax2 = fig.add_axes([0.58,.20,.4,.3])
    plt.title('Alongshore wind (CCMP)',fontsize = 24)
    ax2.axis([t[0],t[-1],-15,15])
    ax2.plot([t[128],t[128]],[-15,15],c = 'red',lw = 3)
    ax2.plot(t,vp_CCMP,label = 'Offshore wind (CCMP)')
    ax2.fill_between(t,vp_CCMP,where=vp_CCMP>=0, color='red',alpha = 0.5)
    ax2.fill_between(t,vp_CCMP,where=vp_CCMP<=0, color='blue',alpha = 0.5)
    ax2.set_ylabel('m/s',fontsize = 24)
    plt.xticks(t[8::24],size = 20,rotation = 90)
    plt.yticks(size = 20)
    plt.grid()
    ax2.text(t[1],-14,'C)', fontsize = 24)




























##
