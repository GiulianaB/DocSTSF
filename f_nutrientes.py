"""
Funciones para analizar los Nutrientes
"""
import pandas as pd
import math
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import f_ADCP_exploratorio as fADCP
import cmocean

def regrillado_vertical_nutrientes():
    """
    Abre el .xlsx de nutrientes y los regrilla verticalmente c/1m

    Parametros de entrada
    --------------------

    Parametros de salida
    --------------------
    Sal: 2D numpy.array (5000,55):(prof,estaciones) matriz de Salinidad
    T  : 2D numpy.array (5000,55):(prof,estaciones) matriz de Temperatura
    NN : 2D numpy.array (5000,55):(prof,estaciones) matriz de Nitrato + Nitrito
    F  : 2D numpy.array (5000,55):(prof,estaciones) matriz de Fosfato
    S  : 2D numpy.array (5000,55):(prof,estaciones) matriz de Silicato
    Cl : 2D numpy.array (5000,55):(prof,estaciones) matriz de Clorofila


    """
    a = pd.read_excel('/media/giuliana/DE_BERDEN/Doctorado/Datos/STSF_data/stsf_sal_o2_cla_nut.xls')

    z = np.linspace(0,4999,5000)
    T = np.empty((5000,51))
    Sal = np.empty((5000,51))
    NN = np.empty((5000,51))
    F = np.empty((5000,51))
    S = np.empty((5000,51))
    Cl = np.empty((5000,51))

    # matriz 3d p/las est. 1-51, las otras no tienen datos
    # t = array 3d. t[estacion, z(irregular), (*)]
    # (*) = pres,sal,temp,satox,ox,nitrato+nitrito(NN),fosfato(fos),silicato(sil),clorofila(clo)
    t = np.nan * np.ones((51,12,7))
    c = 0
    for st in range(1,52):
        c=0
        for i in range(0,len(a['Station'])):
            if a['Station'] [i] == st:
                if math.isnan(a['nitrato umol/kg'][i]) == False:
                    t[st-1,c,:] = [a['PrDM'][i],a['Sal00'][i],a['Potemp090C'][i],a['nitrato umol/kg'][i]+a['nitrito umol/kg'][i],a['fosfato umol/kg'][i],a['silicato umol/kg'][i],a['clorofila (mg/m3)'][i]]
                    c = c+1
    t[2,2,6] = np.nan # outlayer
    for st in range(1,52):
        points    = t[st-1,:,0]  #profs
        values_sal  = t[st-1,:,1]  #sal
        values_t  = t[st-1,:,2]  #temp
        values_NN = t[st-1,:,3]  #N+N
        values_F  = t[st-1,:,4]  #F
        values_S  = t[st-1,:,5]  #S
        values_C  = t[st-1,:,6]  #C
        # if np.isnan(points[0]) == True:
        #     T[:,st-1]   = values_t*np.ones(5000)
        #     Sal[:,st-1] = values_t*np.ones(5000)
        #     NN[:,st-1]  = values_t*np.ones(5000)
        #     F[:,st-1]   = values_t*np.ones(5000)
        #     S[:,st-1]   = values_t*np.ones(5000)
            # Cl[:,st-1]   = values_t*np.ones(5000)
        if np.isnan(points[0]) == False:
            T[:,st-1]   = griddata(points,values_t,z,method='linear')
            Sal[:,st-1] = griddata(points,values_sal,z,method='linear')
            NN[:,st-1]  = griddata(points,values_NN,z,method='linear')
            F[:,st-1]   = griddata(points,values_F,z,method='linear')
            S[:,st-1]   = griddata(points,values_S,z,method='linear')
            Cl[:,st-1]   = griddata(points,values_C,z,method='linear')
            if np.isnan(points[1]) == False:
                aux1 = Sal[:,st-1]
                aux2 = aux1[np.isnan(aux1) == False][0]
                aux = np.where(Sal == aux2)[0][0]
                T[:aux,st-1]    = values_t[0]
                Sal[:aux,st-1]  = values_sal[0]
                NN[:aux,st-1]   = values_NN[0]
                F[:aux,st-1]    = values_F[0]
                S[:aux,st-1]    = values_S[0]
                Cl[:aux,st-1]   = values_C[0]
            else:
                print('Estacion con un dato:', st)

    return Sal, T, NN, F, S, Cl

def mapa_O2(lon_CTD,lat_CTD,O2_CTD,prof):
    """
    Plot de O2 con circulos en las estaciones

    Parametros de entrada
    --------------------
    lon_CTD : 1D numpy.array. longitudes del CTD
    lat_CTD : 1D numpy.array. latitudes del CTD
    O2      : 2D numpy.array (5000,55):(prof,estacion) de Saturacion de Oxigeno disuelto(%) regrillado c/1metro
    prof    : int. Profundidad a la que se quiere hacer el ploteo

    Parametros de Salinidad
    ----------------------
    plot

    """
    #isobata 200m
    path_200 = '/media/giuliana/DE_BERDEN/Doctorado/Datos/200_modif.xlsx'
    iso200 = pd.read_excel(path_200)
    lon200,lat200 = iso200['lon'].tolist(), iso200['lat'].tolist()
    chao = []
    for i in range(len(lat200)):
        if lat200[i] < -40: chao.append(i)
    lon200,lat200 = np.delete(lon200,chao), np.delete(lat200,chao)

    #DOMINIO
    lon_min, lon_max = -60, -48
    lat_min, lat_max = -40, -30

    # # Figure
    fig = plt.figure()
    ax = fig.add_axes([0.05,0.05,0.8,0.9],projection=ccrs.PlateCarree())

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

    # --- isobata
    plt.plot(lon200,lat200,color='grey')

    cmap = plt.cm.get_cmap('YlGn', 4)
    levels = [100,110,120,130,140]
    cr = plt.scatter(lon_CTD,lat_CTD,c = O2_CTD[prof,:],cmap=cmap,vmin=100,vmax=140,edgecolor ='k')
    #barra de color
    cbar = plt.colorbar(cr, orientation='vertical', ticks = levels)
    cbar.set_label(label='%',fontsize=22)
    cbar.ax.tick_params(labelsize=24)
    plt.show()

def mapa_O2_vel(lon_CTD,lat_CTD,O2_CTD,prof):
    """
    Plot de O2 con circulos en las estaciones

    Parametros de entrada
    --------------------
    lon_CTD : 1D numpy.array. longitudes del CTD
    lat_CTD : 1D numpy.array. latitudes del CTD
    O2_CTD  : 2D numpy.array (5000,55):(prof,estacion) de Saturacion de Oxigeno disuelto(%) regrillado c/1metro
    prof    : int. Profundidad a la que se quiere hacer el ploteo

    Parametros de Salinidad
    ----------------------
    plot

    """

    U_10, V_10 = fADCP.datos_ADCP_cada_1m()

    #isobata 200m
    path_200 = '/media/giuliana/DE_BERDEN/Doctorado/Datos/200_modif.xlsx'
    iso200 = pd.read_excel(path_200)
    lon200,lat200 = iso200['lon'].tolist(), iso200['lat'].tolist()
    chao = []
    for i in range(len(lat200)):
        if lat200[i] < -40: chao.append(i)
    lon200,lat200 = np.delete(lon200,chao), np.delete(lat200,chao)

    #DOMINIO
    lon_min, lon_max = -60, -48
    lat_min, lat_max = -40, -30

    # # Figure
    fig = plt.figure()
    ax = fig.add_axes([0.05,0.05,0.8,0.9],projection=ccrs.PlateCarree())

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

    # --- isobata
    plt.plot(lon200,lat200,color='grey')

    cmap = plt.cm.get_cmap('Purples', 4)
    levels = [100,110,120,130,140]
    cr = plt.scatter(lon_CTD,lat_CTD,c = O2_CTD[prof,:],cmap=cmap,vmin=100,vmax=140,edgecolor ='k',s = 100)
    #barra de color
    cbar = plt.colorbar(cr, orientation='vertical', ticks = levels)
    cbar.set_label(label='%',fontsize=22)
    cbar.ax.tick_params(labelsize=24)

    q = plt.quiver(lon_CTD,lat_CTD,U_10[prof,:],V_10[prof,:],scale=1000,alpha = 0.5,width = 0.005)#,headwidth = 2,headlength = 3)
    plt.quiverkey(q, 0.2, 0.9, 50, r'$50 \frac{cm}{s}$', labelpos='E', coordinates='figure',fontproperties={'size': 24})
    plt.show()

def mapa_fluo_vel(lon_CTD,lat_CTD,Fluo_CTD,prof):
    """
    Plot de O2 con circulos en las estaciones

    Parametros de entrada
    --------------------
    lon_CTD : 1D numpy.array. longitudes del CTD
    lat_CTD : 1D numpy.array. latitudes del CTD
    Fluo_CTD  : 2D numpy.array (5000,55):(prof,estacion) de Fluorescencia regrillado c/1metro
    prof    : int. Profundidad a la que se quiere hacer el ploteo

    Parametros de Salinidad
    ----------------------
    plot

    """

    U_10, V_10 = fADCP.datos_ADCP_cada_1m()

    #isobata 200m
    path_200 = '/media/giuliana/DE_BERDEN/Doctorado/Datos/200_modif.xlsx'
    iso200 = pd.read_excel(path_200)
    lon200,lat200 = iso200['lon'].tolist(), iso200['lat'].tolist()
    chao = []
    for i in range(len(lat200)):
        if lat200[i] < -40: chao.append(i)
    lon200,lat200 = np.delete(lon200,chao), np.delete(lat200,chao)

    #DOMINIO
    lon_min, lon_max = -60, -48
    lat_min, lat_max = -40, -30

    # # Figure
    fig = plt.figure()
    ax = fig.add_axes([0.05,0.05,0.8,0.9],projection=ccrs.PlateCarree())

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

    # --- isobata
    plt.plot(lon200,lat200,color='grey')

    q = plt.quiver(lon_CTD,lat_CTD,U_10[prof,:],V_10[prof,:],scale=800,alpha = 0.5,width = 0.005)#,headwidth = 2,headlength = 3)
    plt.quiverkey(q, 0.2, 0.9, 50, r'$50 \frac{cm}{s}$', labelpos='E', coordinates='figure',fontproperties={'size': 24})

    cmap = plt.cm.get_cmap('YlGn', 10)

    levels = [0,1,2,3,4,5,6,7,8,9,10]

    cr = plt.scatter(lon_CTD,lat_CTD,c = Fluo_CTD[prof,:],cmap=cmap,vmin=0,vmax=10,edgecolor ='k',s = 100)
    #barra de color
    cbar = plt.colorbar(cr, orientation='vertical', ticks = levels)
    cbar.set_label(label='Fluorescence',fontsize=22)
    cbar.ax.tick_params(labelsize=24)


    plt.show()

def mapa_O2_fluo_vel(lon_CTD,lat_CTD,O2_CTD,Fluo_CTD,prof):
    """
    Plot de O2, Fluo (uno al lado del otro) con circulos en las estaciones y las veloc.

    Parametros de entrada
    --------------------
    lon_CTD : 1D numpy.array. longitudes del CTD
    lat_CTD : 1D numpy.array. latitudes del CTD
    Fluo_CTD  : 2D numpy.array (5000,55):(prof,estacion) de Fluorescencia regrillado c/1metro
    O2_CTD  : 2D numpy.array (5000,55):(prof,estacion) de Sat de Ox disuelto regrillado c/1metro
    prof    : int. Profundidad a la que se quiere hacer el ploteo

    Parametros de Salinidad
    ----------------------
    plot

    """

    U_10, V_10 = fADCP.datos_ADCP_cada_1m()

    #isobata 200m
    path_200 = '/media/giuliana/DE_BERDEN/Doctorado/Datos/200_modif.xlsx'
    iso200 = pd.read_excel(path_200)
    lon200,lat200 = iso200['lon'].tolist(), iso200['lat'].tolist()
    chao = []
    for i in range(len(lat200)):
        if lat200[i] < -40: chao.append(i)
    lon200,lat200 = np.delete(lon200,chao), np.delete(lat200,chao)

    #DOMINIO
    lon_min, lon_max = -60, -48
    lat_min, lat_max = -40, -30

    fig = plt.figure()
    ax = fig.add_axes([0.05,0.13,0.43,0.9],projection=ccrs.PlateCarree())
    ax_f = fig.add_axes([0.55,0.13,0.43,0.9],projection=ccrs.PlateCarree())
    cax = fig.add_axes([0.06,0.08,0.4,0.05])
    cax_f = fig.add_axes([0.57,0.08,0.4,0.05])

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
    ax.text(-59.5,-39.5,'A)', size = 24)
    # --- isobata
    ax.plot(lon200,lat200,color='grey')
    q = ax.quiver(lon_CTD,lat_CTD,U_10[prof,:],V_10[prof,:],scale=1000,alpha = 0.5,width = 0.005)#,headwidth = 2,headlength = 3)
    ax.quiverkey(q, 0.08, 0.88, 50, r'$50 \frac{cm}{s}$', labelpos='E', coordinates='figure',fontproperties={'size': 24})

    cmap = plt.cm.get_cmap('Purples', 4)
    levels = [100,110,120,130,140]
    cr = ax.scatter(lon_CTD,lat_CTD,c = O2_CTD[prof,:],cmap=cmap,vmin=100,vmax=140,edgecolor ='k',s = 100)
    #barra de color
    cbar = plt.colorbar(cr, orientation='horizontal', ticks = levels,cax = cax)
    cbar.set_label(label='%',fontsize=22)
    cbar.ax.tick_params(labelsize=24)

    ###############################33
    ax_f.set_extent([-60,-48,-40,-30])
    ax_f.add_feature(cfeature.LAND, color='#BDA973')
    ax_f.add_feature(cfeature.LAKES, color='lightcyan')
    ax_f.add_feature(cfeature.RIVERS, edgecolor='black')
    ax_f.coastlines(resolution='50m', color='black', linewidth=1)
    gl = ax_f.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='black', alpha=0.5, linestyle='-')
    gl.xlabels_top = False; gl.ylabels_right = False;gl.ylabels_left = False
    gl.xlocator = mticker.FixedLocator([-60,-58,-56,-54,-52,-50,-48])
    gl.ylocator = mticker.FixedLocator([-40,-38,-36,-34,-32,-30])
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 24, 'color': 'k'}
    gl.ylabel_style = {'size': 24, 'color': 'k'}
    ax_f.text(-59.5,-39.5,'B)', size = 24)

    # --- isobata
    ax_f.plot(lon200,lat200,color='grey')
    q_f = ax_f.quiver(lon_CTD,lat_CTD,U_10[prof,:],V_10[prof,:],scale=1000,alpha = 0.5,width = 0.005)#,headwidth = 2,headlength = 3)
    ax_f.quiverkey(q_f, 0.58, 0.88, 50, r'$50 \frac{cm}{s}$', labelpos='E', coordinates='figure',fontproperties={'size': 24})

    cmap_f = plt.cm.get_cmap('YlGn', 4)
    levels_f = [0,2.5,5,7.5,10]
    cr = ax_f.scatter(lon_CTD,lat_CTD,c = Fluo_CTD[prof,:],cmap=cmap_f,vmin=0,vmax=10,edgecolor ='k',s = 100)
    #barra de color
    cbar_f = plt.colorbar(cr, orientation='horizontal', ticks = levels_f,cax = cax_f)
    cbar_f.set_label(label='Fluorescence',fontsize=22)
    cbar_f.ax.tick_params(labelsize=24)
    ax_f.text

    plt.show()

























    ###

def mapa_fosfato_vel(lon_CTD,lat_CTD,Fluo_CTD,prof):
    """
    Plot de O2 con circulos en las estaciones

    Parametros de entrada
    --------------------
    lon_CTD : 1D numpy.array. longitudes del CTD
    lat_CTD : 1D numpy.array. latitudes del CTD
    F_CTD   : 2D numpy.array (5000,55):(prof,estacion) de Fosfato regrillado c/1metro
    prof    : int. Profundidad a la que se quiere hacer el ploteo

    Parametros de Salida
    ----------------------
    plot

    """

    U_10, V_10 = fADCP.datos_ADCP_cada_1m()

    #isobata 200m
    path_200 = '/media/giuliana/DE_BERDEN/Doctorado/Datos/200_modif.xlsx'
    iso200 = pd.read_excel(path_200)
    lon200,lat200 = iso200['lon'].tolist(), iso200['lat'].tolist()
    chao = []
    for i in range(len(lat200)):
        if lat200[i] < -40: chao.append(i)
    lon200,lat200 = np.delete(lon200,chao), np.delete(lat200,chao)

    #DOMINIO
    lon_min, lon_max = -60, -48
    lat_min, lat_max = -40, -30

    # # Figure
    fig = plt.figure()
    ax = fig.add_axes([0.05,0.05,0.8,0.9],projection=ccrs.PlateCarree())

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

    # --- isobata
    plt.plot(lon200,lat200,color='grey')

    q = plt.quiver(lon_CTD,lat_CTD,U_10[prof,:],V_10[prof,:],scale=800,alpha = 0.5,width = 0.005)#,headwidth = 2,headlength = 3)
    plt.quiverkey(q, 0.2, 0.9, 50, r'$50 \frac{cm}{s}$', labelpos='E', coordinates='figure',fontproperties={'size': 24})

    cmap = plt.cm.get_cmap('YlGn', 4)
    levels = [0,1,2,3,4]
    cr = plt.scatter(lon_CTD,lat_CTD,c = Fluo_CTD[prof,:],cmap=cmap,vmin=0,vmax=4,edgecolor ='k',s = 100)
    #barra de color
    cbar = plt.colorbar(cr, orientation='vertical', ticks = levels)
    cbar.set_label(label='Fluorescence',fontsize=22)
    cbar.ax.tick_params(labelsize=24)


    plt.show()
