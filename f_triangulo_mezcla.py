"""
Funcion porcentaje de masas segun triangulo de Mamayev
"""
path_gral = '/media/giuliana/Disco1TB/'
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import f_CTD_exploratorio as fCTD
from scipy.interpolate import griddata




def porcentaje_triangulo_mamayev(t1,s1,t_parametros,s_parametros):
    """
    Toma una muestra de agua (t1,s1) y el % de agua que tiene de tres masas caracteristicas (t_parametros,s_parametros)

    Parametros de entrada:
    --------------------
    s1 = array salinidades de la muestra de agua
    t1 = array temperatura de la muestra de agua
    t_parametros = [ta,tb,tc] array temperaturas de las tres masas de agua del triangulo
    s_parametros = [sa,sb,sc] array salinidades de las tres masas de agua del triangulo

    Parametros de salida:
    ---------------------
    ma = array con el porcentaje de agua correspondiente a la masa A
    mb = array con el porcentaje de agua correspondiente a la masa B
    mc = array con el porcentaje de agua correspondiente a la masa C

    """
    ta,sa = t_parametros[0],s_parametros[0]
    tb,sb = t_parametros[1],s_parametros[1]
    tc,sc = t_parametros[2],s_parametros[2]

    # -----  A
    ## Recta con la pendiente entre tb,sb y tc,sc en el punto t1,s1: t = m_bc*s+Bbc_porc
    # Bbc_porc es la ordenada al origen de esa recta. Cada recta esta asociado a un %
    # de la mb segun que tan cerca este de tb,sb y esa info te la da el Bbc_porc
    m_bc = np.array((tb-tc)/(sb-sc))
    Bbc_porc = t1-m_bc*s1

    ## valores de B(ordenada) para 0% (sobre la recta tb,sb - tc,sc) y 100% (en el punto ta,sa)
    Bbc_0   = tb - m_bc*sb
    Bbc_100 = ta - m_bc*sa

    #porcentaje de masa a
    ma = (100/(Bbc_100-Bbc_0))*(Bbc_porc-Bbc_0)

    ma[ma > 100] = 100
    ma[ma < 0] = 0

    # -----  B

    ## Recta con la pendiente entre ta,sa y tc,sc en el punto t1,s1: t = m_ac*s+Bac_porc
    # Bac_porc es la ordenada al origen de esa recta. Cada recta esta asociado a un %
    # de la mb segun que tan cerca este de tb,sb y esa info te la da el Bac_porc
    m_ac = np.array((ta-tc)/(sa-sc))
    Bac_porc = t1-m_ac*s1

    ## valores de B(ordenada) para 0% (sobre la recta ta,sa - tc,sc) y 100% (en el punto tb,sb)
    Bac_0   = ta - m_ac*sa
    Bac_100 = tb - m_ac*sb

    #porcentaje de masa b
    mb = (100/(Bac_100-Bac_0))*(Bac_porc-Bac_0)

    mb[mb > 100] = 100
    mb[mb < 0] = 0

    mb[mb < 0] = 0

    # -----  C

    ## Recta con la pendiente entre ta,sa y tb,sb en el punto t1,s1: t = m_ab*s+Bab_porc
    # Bab_porc es la ordenada al origen de esa recta. Cada recta esta asociado a un %
    # de la mc segun que tan cerca este de tc,sc y esa info te la da el Bab_porc

    m_ab = np.array((ta-tb)/(sa-sb))
    Bab_porc = t1-m_ab*s1

    ## valores de B(ordenada) para 0% (sobre la recta ta,sa - tb,sb) y 100% (en el punto tc,sc)
    Bab_0   = ta - m_ab*sa
    Bab_100 = tc - m_ab*sc

    mc = (100/(Bab_100-Bab_0))*(Bab_porc-Bab_0)

    mc[mc > 100] = 100
    mc[mc < 0] = 0

    return ma,mb,mc

def mapa_porcentaje(lat_CTD,lon_CTD,z0,masa,m_a):
    """
    Plotea el % de cierta agua caracteristica a la profundidad deseada. Para ello interpola usando griddata de scipy y suaviza la imagen

    Parámetros de entrada
    --------------------
    masa: 'TW', 'SASW' o 'RDP'. masa de agua a plotear
    m_a: 2D numpy.array (lon,lat) de % de agua a regrillado a la prof z0
    z: int entre 0 y 5000, profundidad a la que se desea plotear
    lat : 1D numpy.array (55):latitudes de las estaciones de CTD
    lon : 1D numpy.array (55):longitudes de las estaciones de CTD

    Parámetros de salida
    -------------------
    plot

    """
    #isobata 200m
    lon200,lat200 = fCTD.isobata_200()

    #DOMINIO
    lon_min, lon_max = -60, -48
    lat_min, lat_max = -40, -30
    x = np.linspace(lon_min,lon_max,500)
    y = np.linspace(lat_min,lat_max,500)
    x,y = np.meshgrid(x,y)

    # # Figure
    if masa == 'RDP':
        cmap = plt.cm.get_cmap('Greens', 4)
    if masa == 'SASW':
        cmap = plt.cm.get_cmap('Blues', 4)
    if masa == 'TW':
        cmap = plt.cm.get_cmap('Reds', 4)

    #suavizado
    sigma = 0.9
    data = gaussian_filter(m_a,sigma)

    fig = plt.figure()
    ax = fig.add_axes([0.05,0.05,0.8,0.9],projection=ccrs.Mercator())

    ax.set_extent([-60,-48,-40,-30],crs = ccrs.PlateCarree())
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

    # --- isobata
    plt.plot(lon200,lat200,color='grey', transform = ccrs.PlateCarree())

    levels = [0,33,50,75,100]
    cr = plt.contourf(x,y,data,levels,cmap=cmap, transform = ccrs.PlateCarree())
    plt.contour(x,y, data,levels, colors=('k'),linewidths=1, transform = ccrs.PlateCarree())
    #barra de color
    cbar = plt.colorbar(cr, orientation='vertical', ticks = levels)
    cbar.ax.tick_params(labelsize=24)
    plt.scatter(lon_CTD,lat_CTD,s = 10,color = 'k', transform = ccrs.PlateCarree())
    plt.show()

    return

def mapa_porcentaje_3masas(lat_CTD,lon_CTD,z0,nombres,m_a,m_b,m_c):
    """
    Plotea el % de cierta agua caracteristica a la profundidad deseada. Para ello interpola usando griddata de scipy y suaviza la imagen

    Parámetros de entrada
    --------------------
    nombres: ['a', 'b' , 'c']. masas de agua a plotear
    m_a,m_b,m_c: 2D numpy.array (5000,55):(prof,estacion) de % de agua a regrillado c/1metro
    z: int entre 0 y 5000, profundidad a la que se desea plotear
    lat : 1D numpy.array (55):latitudes de las estaciones de CTD
    lon : 1D numpy.array (55):longitudes de las estaciones de CTD

    Parámetros de salida
    -------------------
    plot

    """
    #isobata 200m
    lon200,lat200 = fCTD.isobata_200()

    #DOMINIO
    lon_min, lon_max = -60, -48
    lat_min, lat_max = -40, -30
    x = np.linspace(lon_min,lon_max,500)
    y = np.linspace(lat_min,lat_max,500)
    x,y = np.meshgrid(x,y)

    #Suavizado
    sigma = 0.001 #this depends on how noisy your data is, play with it!
    data_a = gaussian_filter(m_a,sigma)
    data_b = gaussian_filter(m_b,sigma)
    data_c = gaussian_filter(m_c,sigma)

    data_a[data_a < 32] = np.nan
    data_b[data_b < 32] = np.nan
    data_c[data_c < 32] = np.nan

    # # Figure
    cmap_a = plt.cm.get_cmap('Blues', 4)
    cmap_b = plt.cm.get_cmap('Reds', 4)
    cmap_c = plt.cm.get_cmap('Greens', 4)

    fig = plt.figure()
    ax = fig.add_axes([0.05,0.05,0.8,0.9],projection=ccrs.Mercator())

    ax.set_extent([-60,-48,-40,-30],crs = ccrs.PlateCarree())
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

    # --- isobata
    ax.plot(lon200,lat200,color='grey',transform = ccrs.PlateCarree())

    levels = [0,33,50,75,100]
    levels_33 = [33]
    cf_a = ax.contourf(x,y,data_a,levels,cmap=cmap_a,transform = ccrs.PlateCarree())
    cf_b = ax.contourf(x,y,data_b,levels,cmap=cmap_b,transform = ccrs.PlateCarree())
    cf_c = ax.contourf(x,y,data_c,levels,cmap=cmap_c,transform = ccrs.PlateCarree())


    #barra de color
    cax_a = fig.add_axes([0.75,0.1,0.02,0.8])
    cax_b = fig.add_axes([0.79,0.1,0.02,0.8])
    cax_c = fig.add_axes([0.83,0.1,0.02,0.8])

    cbar_a = plt.colorbar(cf_a, orientation='vertical', ticks = levels, cax = cax_a)
    cbar_a.set_label(label=nombres[0],fontsize=22)
    cbar_a.set_label(nombres[0], labelpad=-40, y=0, rotation=-45)
    cbar_a.ax.set_yticklabels(['','','',''])

    cbar_b = plt.colorbar(cf_b, orientation='vertical', ticks = levels,cax  =cax_b)
    cbar_b.set_label(label=nombres[1],fontsize=22)
    cbar_b.set_label(nombres[1], labelpad=-40, y=0, rotation=-45)
    cbar_b.ax.set_yticklabels(['','','',''])

    cbar_c = plt.colorbar(cf_c, orientation='vertical', ticks = levels,cax = cax_c)
    cbar_c.set_label(label=nombres[2],fontsize=22)
    cbar_c.set_label(nombres[2], labelpad=-60, y=0, rotation=-45)
    cbar_c.ax.tick_params(labelsize=24)

    ax.scatter(lon_CTD,lat_CTD,s = 10,color = 'k',transform = ccrs.PlateCarree())
    plt.show()

    return

def f_ubicacion_STSF(lon_CTD,lat_CTD,T,S,profs,t_parametros = [22.12,7.6,15.02],s_parametros = [36.77,33.8,27.73]):
    """
    Busca la posicion del frente STSF durante la campania STSF2013 utilizando
    los datos termohalinos y siguiendo la simple metodologia del triangulo
    de mezcla (Mamayev).
    Busca los lugares donde pasa de haber mayoria de SASW a mayoria de TW con un
    limite porcentual determinado por 'limite'. Esto lo realiza para todas las
    profs.

    Parametros de entrada
    ---------------------
    lon_CTD : 1D np.array (55):lons de las est. de CTD
    lat_CTD : 1D np.array (55):lats de las est. de CTD
    T   : 2D np.array (5000,55):(z,est) de Temp regrillado c/1metro (generado por fCTD.regrillado_vertical)
    S   : 2D np.array (5000,55):(z,est) de Sal  regrillado c/1metro (generado por fCTD.regrillado_vertical)
    profs: 1D list (nprof) profundidades (metros) donde se quiere buscar el frente STSF. nprof: # de niveles
    t_parametros: 1D np.array. [TW,SASW,RDP]. Temp de los vertices del triangulo de mezcla.
    s_parametros: 1D np.array. [TW,SASW,RDP]. Sal  de los vertices del triangulo de mezcla.

    Paramtros de salida
    --------------------
    STSF: list. c/elemento,asociado a una prof, tiene un 2d np.array con lon,lat
    del frente.
    """
    #DOMINIO p/grillado
    lon_min, lon_max = -60, -48
    lat_min, lat_max = -40, -30
    dx,dy = 500,500
    x_1d = np.linspace(lon_min,lon_max,dx)
    y_1d = np.linspace(lat_min,lat_max,dy)
    x,y = np.meshgrid(x_1d,y_1d)
    points = (lon_CTD,lat_CTD)
    STSF = []
    STSF_list = []
    for z in profs:
        temp_nivel,sal_nivel = T[z,:],S[z,:]
        values_t = temp_nivel
        values_s = sal_nivel
        t_nivel_grillada = griddata(points,values_t,(x,y))
        s_nivel_grillada = griddata(points,values_s,(x,y))
        m_TW,m_SASW,m_RDP = porcentaje_triangulo_mamayev(t_nivel_grillada,s_nivel_grillada,t_parametros,s_parametros)
        m_TW[m_TW < m_RDP]     = np.nan
        m_TW[m_TW < m_SASW]    = np.nan
        m_SASW[m_SASW < m_RDP] = np.nan
        m_SASW[m_SASW < m_TW]  = np.nan
        STSF_nivel = np.nan*np.ones_like(m_TW)
        STSF_nivel_list = []
        for i_lat in range(dy):
            for i_lon in range(dx):
                if np.isnan(m_SASW[i_lat,i_lon]) == False and np.isnan(m_SASW[i_lat,i_lon+1]) == True and np.isnan(m_TW[i_lat,i_lon+1]) == False:
                    STSF_nivel[i_lat,i_lon] = 1
        STSF_nivel[y<-35.8] = np.nan
        for i_lat in range(dy):
            for i_lon in range(dx):
                if np.isnan(STSF_nivel[i_lat,i_lon]) == False:
                    STSF_nivel_list.append([y_1d[i_lat],x_1d[i_lon]])
        STSF_nivel_array = np.array(STSF_nivel_list)

        STSF_list.append(STSF_nivel_array)
        STSF.append(STSF_nivel)

    return STSF_list,STSF,x,y

def plot_STSF(lon_CTD,lat_CTD,T,S,profs,t_parametros = [22.12,7.6,15.02],s_parametros = [36.77,33.8,27.73]):
    """
    Plotea la posicion del frente STSF durante la campania STSF2013 siguiendo
    a f_ubicacion_STSF.

    Parametros de entrada
    ---------------------
    lon_CTD : 1D np.array (55):lons de las est. de CTD
    lat_CTD : 1D np.array (55):lats de las est. de CTD
    T   : 2D np.array (5000,55):(z,est) de Temp regrillado c/1metro (generado por fCTD.regrillado_vertical)
    S   : 2D np.array (5000,55):(z,est) de Sal  regrillado c/1metro (generado por fCTD.regrillado_vertical)
    profs: 1D list (nprof) profundidades (metros) donde se quiere buscar el frente STSF. nprof: # de niveles
    t_parametros: 1D np.array. [TW,SASW,RDP]. Temp de los vertices del triangulo de mezcla.
    s_parametros: 1D np.array. [TW,SASW,RDP]. Sal  de los vertices del triangulo de mezcla.

    Paramtros de salida
    --------------------
    plot
    """

    STSF_list,STSF,x,y = f_ubicacion_STSF(lon_CTD,lat_CTD,T,S,profs,t_parametros = [22.12,7.6,15.02],s_parametros = [36.77,33.8,27.73])

    #isobata 200m
    lon200,lat200 = fCTD.isobata_200()

    fig = plt.figure()
    ax = fig.add_axes([0.05,0.05,0.8,0.9],projection=ccrs.Mercator())
    ax.set_extent([-60,-48,-40,-30],crs = ccrs.PlateCarree())
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
    # --- isobata
    ax.plot(lon200,lat200,color='grey', transform = ccrs.PlateCarree())
    # --- estaciones
    ax.scatter(lon_CTD,lat_CTD,s = 10,color = 'k', transform = ccrs.PlateCarree())
    colors = ['Red','Blue','Green','Grey','Orange','Purple']
    for iz in range(len(profs)):
        ax.scatter(STSF_list[iz][:,1],STSF_list[iz][:,0],s = 10,transform = ccrs.PlateCarree(),label = profs[iz])
        # ax.pcolor(x,y,STSF[iz],color = colors[iz], alpha = 0.7,transform = ccrs.PlateCarree(), label = str(profs[iz])+'m')
    plt.legend(fontsize = 30)



















###
