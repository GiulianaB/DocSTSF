"""
Funciones para analisi Exploratorio
Cosas por ahcer para q funcione:
* Acomodar los archivos xls para que las variables tengan los mismos nombres que los cnv.

"""
path_gral = '/media/giuliana/DE_BERDEN/'
path_gral = '/media/giuliana/Disco1TB/'

# path_gral = '/home/california/Documents'
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




def isobata_200():
    """
    Descarga la isobada de 200m que me pasó Silvia Romero (GEBCO) y extrae la zona de DOMINIO
    Parametros de entrada
    --------------------
    Parametros de salida
    --------------------
    lon200,lat200
    """
    path_200 = path_gral+'Doctorado/Datos/200_modif.xlsx'
    iso200 = pd.read_excel(path_200)
    lon200,lat200 = iso200['lon'].tolist(), iso200['lat'].tolist()
    chao = []
    for i in range(len(lat200)):
        if lat200[i] < -40: chao.append(i)
    lon200,lat200 = np.delete(lon200,chao), np.delete(lat200,chao)

    return lon200,lat200

def lat_lon_CTD():
    """
    Guarda la posicion de las estaciones
    Parametros de salida
    -------------------
    lat_CTD: Latitudes de las estaciones de CTD de la campaña STSF2013
    lon_CTD: Longitud de las estaciones de CTD de la campaña STSF2013
    """
    lat_CTD = [-37.44883333333333, -37.5175, -37.5895, -37.653666666666666, -37.7345, -37.79683333333333, -37.86933333333333, -37.938833333333335, -37.056, -37.0545, -37.05866666666667, -37.08, -37.0565, -37.0585, -37.057, -37.0575, -37.062333333333335, -37.054, -37.05716666666667, -35.686166666666665, -35.635333333333335, -35.57816666666667, -35.521166666666666, -35.462666666666664, -35.4115, -35.356, -35.29683333333333, -35.24216666666667, -34.65116666666667, -34.6995, -34.734833333333334, -34.735166666666665, -34.501, -34.43233333333333, -34.330666666666666, -34.23766666666667, -34.109833333333334, -33.98166666666667, -33.8585, -33.7225, -33.59616666666667, -32.94683333333333, -32.9255, -32.89383333333333, -32.88366666666667, -32.3575, -32.26866666666667, -32.17316666666667, -32.088, -31.998, -31.9135, -31.8323, -31.7368333333, -31.64584, -31.558333333]
    lon_CTD = [-55.71366666666667, -55.5425, -55.38166666666667, -55.239333333333335, -55.0725, -54.916333333333334, -54.75983333333333, -54.599666666666664, -54.034333333333336, -53.86066666666667, -53.68716666666667, -53.519, -53.33283333333333, -53.15383333333333, -52.98016666666667, -52.81483333333333, -52.629666666666665, -52.455333333333336, -52.280833333333334, -52.40683333333333, -52.5445, -52.697833333333335, -52.844166666666666, -52.991166666666665, -53.1495, -53.285833333333336, -53.438833333333335, -53.58983333333333, -52.8815, -52.663666666666664, -52.402166666666666, -52.2235, -51.506, -51.592166666666664, -51.660833333333336, -51.730666666666664, -51.869, -51.97233333333333, -52.088166666666666, -52.21333333333333, -52.33233333333333, -51.61233333333333, -51.39983333333333, -51.14416666666666, -50.93233333333333, -49.87466666666667, -49.98166666666667, -50.105, -50.219166666666666, -50.33133333333333, -50.44716666666667, -50.5816666667, -50.68952, -50.7929, -50.915216667]

    return lat_CTD,lon_CTD

def carga_datos_CTD():
  """
  Abre datos de CTD de la campaña STSF2013

  Parámetros de entrada
  -------------------


  Parámetros de salida
  ------------------
  DATA: lista, cada elemento corresponde al archivo cnv de cada estacion.
  lat : 1D numpy.array (55):latitud de las estaciones de CTD
  lon : 1D numpy.array (55):longitud de las estaciones de CTD
  """
  path = path_gral+'Doctorado/Datos/STSF_data/'
  DATA = []

  for st in range(1,52):
      if st == 12:
          DATA.append(pd.read_excel(str(path)+'d0'+str(st)+'.xlsx'))
      else:
          DATA.append(fCNV(str(path) + 'd0'+str(st)+'.cnv'))
  for st in [52,53,54,55]:
      DATA.append(pd.read_excel(str(path)+'/EO_'+str(st)+'Down_cal_f.xlsx'))

  lat = [-37.44883333333333, -37.5175, -37.5895, -37.653666666666666, -37.7345, -37.79683333333333, -37.86933333333333, -37.938833333333335, -37.056, -37.0545,
 -37.05866666666667, -37.08, -37.0565, -37.0585, -37.057, -37.0575, -37.062333333333335, -37.054, -37.05716666666667, -35.686166666666665, -35.635333333333335, -35.57816666666667, -35.521166666666666, -35.462666666666664,
 -35.4115, -35.356, -35.29683333333333, -35.24216666666667, -34.65116666666667, -34.6995, -34.734833333333334, -34.735166666666665, -34.501, -34.43233333333333,
 -34.330666666666666, -34.23766666666667, -34.109833333333334, -33.98166666666667, -33.8585, -33.7225, -33.59616666666667, -32.94683333333333, -32.9255, -32.89383333333333, -32.88366666666667,
 -32.3575, -32.26866666666667, -32.17316666666667, -32.088, -31.998, -31.9135, -31.8323, -31.7368333333, -31.64584, -31.558333333]
  lon = [-55.71366666666667, -55.5425, -55.38166666666667, -55.239333333333335, -55.0725, -54.916333333333334, -54.75983333333333, -54.599666666666664, -54.034333333333336, -53.86066666666667, -53.68716666666667, -53.519, -53.33283333333333, -53.15383333333333, -52.98016666666667, -52.81483333333333,
 -52.629666666666665, -52.455333333333336, -52.280833333333334, -52.40683333333333, -52.5445, -52.697833333333335, -52.844166666666666, -52.991166666666665, -53.1495, -53.285833333333336,
 -53.438833333333335, -53.58983333333333, -52.8815, -52.663666666666664,
 -52.402166666666666, -52.2235, -51.506, -51.592166666666664, -51.660833333333336, -51.730666666666664, -51.869, -51.97233333333333, -52.088166666666666,
 -52.21333333333333, -52.33233333333333, -51.61233333333333, -51.39983333333333, -51.14416666666666, -50.93233333333333, -49.87466666666667, -49.98166666666667, -50.105, -50.219166666666666, -50.33133333333333, -50.44716666666667, -50.5816666667, -50.68952, -50.7929, -50.915216667]

  return DATA,lat,lon

def carga_datos_termosal():
    """
    Abre datos de termosal de la campaña STSF2013

    Parámetros de entrada
    -------------------
    Parámetros de salida
    ------------------
    lon_TS : 1D numpy.array longitudes del termosaligrafo de la campaña STSF2013
    lat_TS : 1D numpy.array latitudes del termosaligrafo de la campaña STSF2013
    temp_TS: 1D numpy.array temperatura del termosaligrafo de la campaña STSF2013
    sal_TS : 1D numpy.array salinidades del termosaligrafo de la campaña STSF2013

    """
    path = path_gral+'Doctorado/Datos/STSF_data/termosal_STSF/'
    TS1 = fCNV(path+'PD201304A.cnv')
    TS2 = fCNV(path+'PD201304B.cnv')
    TS3 = fCNV(path+'PD201304C.cnv')
    TS4 = fCNV(path+'PD201304D.cnv')
    TS5 = fCNV(path+'PD201304E.cnv')
    TS6 = fCNV(path+'PD201304F.cnv')
    TS7 = fCNV(path+'PD201304G.cnv')
    TS8 = fCNV(path+'PD201304H.cnv')
    lat_TS = np.concatenate([TS1['LATITUDE'],TS2['LATITUDE'],TS3['LATITUDE'],TS4['LATITUDE'],TS5['LATITUDE'],TS6['LATITUDE'],TS7['LATITUDE'],TS8['LATITUDE']])
    lon_TS = np.concatenate([TS1['LONGITUDE'],TS2['LONGITUDE'],TS3['LONGITUDE'],TS4['LONGITUDE'],TS5['LONGITUDE'],TS6['LONGITUDE'],TS7['LONGITUDE'],TS8['LONGITUDE']])
    temp_TS = np.concatenate([TS1['TEMP'],TS2['TEMP'],TS3['TEMP'],TS4['TEMP'],TS5['TEMP'],TS6['TEMP'],TS7['TEMP'],TS8['TEMP']])
    sal_TS = np.concatenate([TS1['PSAL'],TS2['PSAL'],TS3['PSAL'],TS4['PSAL'],TS5['PSAL'],TS6['PSAL'],TS7['PSAL'],TS8['PSAL']])
    #valores malos:
    val_malos=[]
    for med in range(len(lat_TS)):
        if temp_TS [med] == -9.99e-29:
            val_malos.append(med)
        if lat_TS[med] > -32.1 and lon_TS[med] < -51:
            sal_TS [med] = 29.8
        if lat_TS[med] > -33 and lat_TS[med] < -32.85 and lon_TS[med] < -51.1 and lon_TS[med] > -52.4:
            temp_TS [med] = 16.5
    lat_TS, lon_TS,temp_TS,sal_TS = np.delete(lat_TS,val_malos),np.delete(lon_TS,val_malos),np.delete(temp_TS,val_malos),np.delete(sal_TS,val_malos)

    return lon_TS,lat_TS,temp_TS, sal_TS

def regrillado_vertical(DATA):
    """
    Regrilla verticalmente los datos de T,S de cada est de la campaña STSF2013 c/1metro

    Parámetros de entrada
    --------------------
    DATA: lista, cada elemento corresponde al archivo cnv de cada estacion

    Parámetros de salida
    -------------------
    T   : 2D numpy.array (5000,55):(prof,estacion) de Temperatura regrillado c/1metro
    S   : 2D numpy.array (5000,55):(prof,estacion) de Salinidad   regrillado c/1metro
    O2  : 2D numpy.array (5000,55):(prof,estacion) de Saturacion de Oxigeno disuelto(%) regrillado c/1metro
    Fluo: 2D numpy.array (5000,55):(prof,estacion) de Fluorescencia regrillado c/1metro
    z   : 1D numpy.array (5000)   :(prof) profundidades en las cuales se regrilló.

    """
    z = np.linspace(0,4999,5000)
    T = np.nan*np.ones((5000,55))
    S = np.nan*np.ones((5000,55))
    O2 = np.nan*np.ones((5000,55))
    Fluo = np.nan*np.ones((5000,55))
    for st in range(1,56):
        data_est = DATA[st-1]
        points   = DATA[st-1]['PRES']
        values_t = DATA[st-1]['TEMP']
        values_s = DATA[st-1]['PSAL']
        values_f = DATA[st-1]['flSP']

        T[:,st-1]   = griddata(points,values_t,z,method='linear')
        S[:,st-1]   = griddata(points,values_s,z,method='linear')
        Fluo[:,st-1] = griddata(points,values_f,z,method='linear')
        aux = int(points[0]+1)
        T[:aux,st-1]    = values_t[0]
        S[:aux,st-1]    = values_s[0]
        Fluo[:aux,st-1] = values_f[0]
        if st < 51:
            values_O2 = DATA[st-1]['oxigen_ml_L']*100/DATA[st-1]['oxsolML/L']
            O2[:,st-1] = griddata(points,values_O2,z,method='linear')
            O2[:aux,st-1] = values_O2[0]
    Fluo[Fluo < 0] = 0.01
    return T,S,O2,Fluo,z

def suavizado(M):
    """
    Suaviza la matriz 'M' en la prof 'prof'.

    Parámetros de entrada
    --------------------
    M: 2D numpy.array (nlon,nlat) de variable regrillada

    Parámetros de salida
    -------------------
    M_s: 2D numpy.array ():variable suavizada a la 'prof'

    """
    nlon,nlat = M.shape
    M_s = np.empty(M.shape)
    M_s = np.copy(M)
    for i_lat in range(2,nlat-2):
        for i_lon in range(2,nlon-2):
            aux_lons = [M[i_lon-2,i_lat],M[i_lon-1,i_lat],M[i_lon,i_lat],M[i_lon+1,i_lat],M[i_lon,i_lat+2]]
            aux_lats = [M[i_lon,i_lat-2],M[i_lon,i_lat-1],M[i_lon,i_lat],M[i_lon,i_lat+1],M[i_lon,i_lat+2]]
            aux_esq1 = [M[i_lon-1,i_lat-1],M[i_lon+1,i_lat+1],M[i_lon-1,i_lat+1],M[i_lon+1,i_lat-1]]
            aux_esq2 = [M[i_lon-2,i_lat-2],M[i_lon+2,i_lat+2],M[i_lon-2,i_lat+2],M[i_lon+2,i_lat-2]]
            aux = np.concatenate((aux_lons,aux_lats,aux_esq1,aux_esq2))
            M_s[i_lon,i_lat] = np.nanmean(aux)
    return M_s

def mapa_temp(lat_CTD,lon_CTD,z0,T):
    """
    Plotea la temperatura a la profundidad deseada. Para ello interpola usando griddata de scipy

    Parámetros de entrada
    --------------------
    T: 2D numpy.array (5000,55):(prof,estacion) de Temperatura regrillado c/1metro
    z: int entre 0 y 5000, profundidad a la que se desea plotear
    lat : 1D numpy.array (55):latitudes de las estaciones de CTD
    lon : 1D numpy.array (55):longitudes de las estaciones de CTD

    Parámetros de salida
    -------------------
    plot
    data: matriz_grillada de la Temp a la prof z0

    """
    #DOMINIO
    lon_min, lon_max = -60, -48
    lat_min, lat_max = -40, -30
    x = np.linspace(lon_min,lon_max,500)
    y = np.linspace(lat_min,lat_max,500)
    x,y = np.meshgrid(x,y)

    if int(z0) == 0:
        #abrir archivos del termosal
        lon_TS,lat_TS,temp_TS,sal_TS = carga_datos_termosal()

        LONS, LATS = np.concatenate([lon_TS,lon_CTD]), np.concatenate([lat_TS,lat_CTD])
        Tsup_aux =  np.concatenate([temp_TS,T[int(z0),:]])

        """ Arreglo """
        lat_arreglo = [-38 ,-34.8 ,-34.8 ,-34.8 ,-37   ,-36   ,-34  ,-32 ,-35.2,-36 ,-36  ,-36  ,-32.8,-37.8,-37  ,-36.5,-34.9]
        lon_arreglo = [-56 ,-54   ,-54.5 ,-55   ,-57   ,-57   ,-51.5,-51 ,-51.8,-53 ,-52.9,-53.1,-51  ,-54  ,-55  ,-54.5,-52.6 ]
        t_arreglo =   [9.2 ,np.nan,np.nan,np.nan,np.nan,np.nan,16.5 ,16.2,17.5 ,12  ,12.2 ,11.8 ,16   ,14.8 ,12   ,12   , 14.5]

        lat_arreglo2 = [-37.45 ,-37.35,-37.35   ,-36.8 ,-38 ,  -33.04  ]
        lon_arreglo2 = [-52.7,-52.75  ,-52.8    ,-52.7 ,-57 ,  -50.16]
        t_arreglo2 =   [14  ,14.2     ,14.4     ,16    , 10 ,  18]

        lat_arreglo = np.concatenate([lat_arreglo,lat_arreglo2])
        lon_arreglo = np.concatenate([lon_arreglo,lon_arreglo2])
        t_arreglo = np.concatenate([t_arreglo,t_arreglo2])

        LONS_con_arreglo, LATS_con_arreglo = np.concatenate([LONS,lon_arreglo]), np.concatenate([LATS,lat_arreglo])
        Tnivel = np.concatenate([Tsup_aux,t_arreglo])

        lon_nivel,lat_nivel = LONS_con_arreglo, LATS_con_arreglo
    else:
        lon_nivel,lat_nivel = lon_CTD,lat_CTD
        Tnivel = T[int(z0),:]


    #Interpolación de T en el dominio:
    points = (lon_nivel,lat_nivel)
    values = Tnivel
    T_grillada = griddata(points,values,(x,y))
    data = suavizado(T_grillada)

    # # Figure
    cmap_t = plt.cm.get_cmap('coolwarm', 10)

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
    lon200,lat200 = isobata_200()
    plt.plot(lon200,lat200,color='grey', transform=ccrs.PlateCarree())

    levels = [5,7,9,11,13,15,17,19,21,23,25]
    cr = plt.contourf(x,y,data,levels,cmap=cmap_t, transform=ccrs.PlateCarree())
    plt.contour(x,y, data,levels, colors=('k'),linewidths=1, transform=ccrs.PlateCarree())
    #barra de color
    cbar = plt.colorbar(cr, orientation='vertical', ticks = levels)
    cbar.set_label(label='°C',fontsize=22)
    cbar.ax.tick_params(labelsize=24)

    plt.scatter(lon_CTD,lat_CTD,s = 20,color = 'k', transform=ccrs.PlateCarree())
    plt.show()

    return data

def mapa_sal(lat_CTD,lon_CTD,z0,S):
    """
    Plotea la salinidad a la profundidad deseada. Para ello interpola usando griddata de scipy y suaviza la imagen

    Parámetros de entrada
    --------------------
    S: 2D numpy.array (5000,55):(prof,estacion) de Salinidad regrillado c/1metro
    z: int entre 0 y 5000, profundidad a la que se desea plotear
    lat : 1D numpy.array (55):latitudes de las estaciones de CTD
    lon : 1D numpy.array (55):longitudes de las estaciones de CTD

    Parámetros de salida
    -------------------
    plot
    data: matriz_grillada de la Sal a la prof z0

    """
    #DOMINIO
    lon_min, lon_max = -60, -48
    lat_min, lat_max = -40, -30
    x = np.linspace(lon_min,lon_max,500)
    y = np.linspace(lat_min,lat_max,500)
    x,y = np.meshgrid(x,y)

    #Agrego Termosal si estoy en superficie
    if int(z0) == 0:
        #abrir archivos del termosal
        lon_TS,lat_TS,temp_TS,sal_TS = carga_datos_termosal()

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
    else:
        lon,lat = lon_CTD,lat_CTD
        Ssup = S[int(z0),:]

    #Interpolación de T en el dominio:
    points = (lon,lat)
    values = Ssup
    S_grillada = griddata(points,values,(x,y))

    data = suavizado(S_grillada)
    # # Figure
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
    # termosal
    if int(z0) == 0:
        plt.scatter(lon_TS[::150],lat_TS[::150],s = 10,color = 'gray',transform=ccrs.PlateCarree(),alpha = 0.5,zorder = 1)


    # --- isobata
    lon200,lat200 = isobata_200()
    plt.plot(lon200,lat200,color='grey', transform=ccrs.PlateCarree())

    levels = [30,31,32,33,34,35,36,37]
    cr = plt.contourf(x,y,data,levels,cmap=cmap_s, transform=ccrs.PlateCarree())
    plt.contour(x,y,data,levels, colors=('k'),linewidths=1, transform=ccrs.PlateCarree())
    #barra de color
    cbar = plt.colorbar(cr, orientation='vertical', ticks = levels)
    cbar.ax.tick_params(labelsize=24)
    plt.scatter(lon_CTD,lat_CTD,s = 20,color = 'k', transform=ccrs.PlateCarree())

    fecha = 'SSS'
    plt.text(-59.5,-31.2,fecha,size=24,bbox = dict(boxstyle='round', facecolor='white', alpha=0.8),transform=ccrs.PlateCarree())


    return data

def mapa_batimetrico(lat_min,lat_max,lon_min,lon_max,OSCAR = 'si'):
    """
    Plot el mapa batimetrico usando ETOPO entre los limites dados

    Parametros de entrada
    --------------------
    lat_min: 1D numpy.array limite del mapa
    lat_max: 1D numpy.array limite del mapa
    lon_min: 1D numpy.array limite del mapa
    lon_max: 1D numpy.array limite del mapa
    OSCAR  : 'si' o 'no'. Velocidades de OSCAR?

    Parametros de salinidad
    --------------------
    plot

    """

    LATS,LONS = lat_lon_CTD()
    path = '/media/california/DE_BERDEN/Doctorado/Datos/batimetria/ETOPO1_Bed_g_gmt4.grd'

    DATA = Dataset(path,'r')
    DATA.set_auto_mask(False)

    x = DATA['x'][:]
    y = DATA['y'][:]
    z = DATA['z'][:]

    x_STSF = x[7200:7921]   #lons=[-60,-48]
    y_STSF = y[3000:3601]   #lats=[-40:-30]
    z_STSF = z[3000:3601,7200:7921]
    interior = z_STSF > 0
    z_STSF[interior] = 3000

    plt.figure()
    cmap=plt.cm.get_cmap('cmo.tempo_r', 40)
    ax = plt.axes(projection=ccrs.Mercator())
    levels = [-4000,-3500,-3000,-2500,-2000,-1500,-1000,-500,-200,-100,0]
    levels_iso = [-2000,-200]
    BATH = ax.contourf(x_STSF,y_STSF,z_STSF,levels, cmap=cmap,vmin=-4000, vmax=500,extend='both',transform=ccrs.PlateCarree(),alpha = 0.8)
    BATH_ = ax.contour(x_STSF,y_STSF,z_STSF,levels_iso,cmap='Greys',vmin=-50000,vmax=-180,linewidths = 1)
    cbar = plt.colorbar(BATH,ticks=levels)
    cbar.ax.set_yticklabels(-1*np.array(levels))
    cbar.ax.tick_params(labelsize=18)
    cbar.ax.set_ylabel('Meters',fontsize=20)
    #######################################
    #Agrego veloc de oscar
    if OSCAR == 'si':
        path = '/media/california/DE_BERDEN/Doctorado/Datos/satelital/oscar_vel2013.nc.gz.nc4'
        data_VEL = Dataset(path)
        #Recorto sobre mi dominio
        lon_vel = data_VEL['longitude'][840:877]
        lat_vel = data_VEL['latitude'][330:361]
        time = data_VEL['time'][:]
        #Fecha de la campania: time[55] (2013-10-06)
        # dates = []
        # for t in range(len(time)):
        #     dates.append(netCDF4.num2date(time[t], data_VEL['time'].units,'gregorian'))
        t0 = 55
        u = data_VEL['u'][:,0,330:361,840:877]
        v = data_VEL['v'][:,0,330:361,840:877]

        x_vel,y_vel = np.meshgrid(lon_vel,lat_vel)
        cs = ax.quiver(x_vel,y_vel,(u[t0-1,:,:]+u[t0,:,:])/2,(v[t0-1,:,:]+v[t0,:,:])/2,color='0.2',units='width',scale=12,transform = ccrs.PlateCarree(),zorder = 3)
        ax.quiverkey(cs, 0.4, 0.81, 0.5, r'$ 50\frac{cm}{s}$', labelpos='E',coordinates='figure',fontproperties={'size' :24})
        # ax.text(15000,1000000,'Currents (OSCAR)\nMean between 3 and 10 October',size=12,bbox=dict(facecolor='white'))
    #######################################
    for Nest in range(1):
        ax.scatter(LONS,LATS,s=25,color='r',transform=ccrs.PlateCarree(),zorder = 4)
        # Todas las estaciones:
        # est = ['1',        '8', '9',  '19', '28','20', '29', '32', '33', '41', '42','45',   '46'   ,'55']
        # lon_est = [-55.8,-54.3,-54.2,-52.2,-53.9,-52.2,-53.2,  -52.2,-51.3,-52.6,-51.9,-51 ,  -49.7,-51.2]
        # lat_est = [-37.2,-37.8,-36.8,-36.8,-35.0,-35.6,-34.5,-34.6,-34.3,-33.3,-32.7,-32.65,-32.1  ,-31.2]
        # Algunas:
        est = ['1',        '8', '9',  '19', '28','20',  '33', '41',  '46'   ,'55']
        lon_est = [-55.8,-54.3,-54.2,-52.2,-54.1,-52.0,-51.1,-52.6, -49.5,-51.2]
        lat_est = [-37.2,-37.8,-36.8,-36.8,-35.0,-35.6,-34.4,-33.3,-32.1  ,-31.2]

        for st in range (len(est)):
            ax.text(lon_est[st],lat_est[st],est[st],fontsize=22,bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round'),transform=ccrs.PlateCarree())
        # Todas los nombres de las secciones:
        # T = [   'S1',   'S2', 'S3', 'S4', 'S5','S6','S7']
        # lon_T = [-55.6,-53.1,-53.2,-51.0,-51.8,-49.7,-50.3]
        # lat_T = [-38.2,-37.5,-35.9,-35.6,-33.8,-33.6,-31.7]
        # Solo algunos:
        T = [   'S1',   'S2', 'S3',  'S5','S7']
        lon_T = [-55.6,-53.1,-53.2,-51.8,-50.3]
        lat_T = [-38.2,-37.7,-36.1,-33.8,-31.7]

        for ti in range(len(T)):
            ax.text(lon_T[ti],lat_T[ti],T[ti],color='darkred',fontsize=18,bbox=dict(facecolor='white', edgecolor='r', boxstyle='circle'),transform=ccrs.PlateCarree(),zorder=4)
        # ax.arrow(-52.2,-34.8,1.2,-0.7,color='r',transform=ccrs.PlateCarree())
        # ax.arrow(-50.9,-32.95,1.2,-0.5,color='r',transform=ccrs.PlateCarree())
    ax.set_extent([lon_min,lon_max,lat_min,lat_max])
    ax.coastlines(resolution='50m', color='black', linewidth=1)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='black', alpha=0.3, linestyle='-')
    gl.xlabels_top = False; gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator([-60,-58,-56,-54,-52,-50,-48])
    gl.ylocator = mticker.FixedLocator([-40,-38,-36,-34,-32,-30])
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 18, 'color': 'k'}
    gl.ylabel_style = {'size': 18, 'color': 'k'}
    #
    ax.add_feature(cfeature.LAKES, color='lightcyan')
    ax.add_feature(cfeature.RIVERS, edgecolor='black')
    ax.add_feature(cfeature.LAND, color='#BDA975')

def corte_seccion(seccion, variable,matriz_grillada, DATA):
    """
    Plotea el corte vertical de la variable.

    Parámetros de entrada
    --------------------
    seccion: 1,2,3,4,5,6 o 7 (int). Seccion a plotear
    Variable: 'T' o 'S' o 'O2' o  'Fluo' (string) variable a plotear
    matriz_grillada: T o S o O2 o Fluo: 2D numpy.array (5000,55):(prof,estacion) de Temp/Sal/SatO2/Fluo regrillado c/1metro
    DATA: lista, cada elemento corresponde al archivo cnv de cada estacion.

    Parámetros de salida
    -------------------
    plot

    """
    #ingreso los datos de cada seccion
    if seccion == 1:
        prof = [DATA[0]['PRES'][-1], DATA[1]['PRES'][-1], DATA[2]['PRES'][-1], DATA[3]['PRES'][-1],DATA[4]['PRES'][-1],  DATA[5]['PRES'][-1], DATA[6]['PRES'][-1], DATA[7]['PRES'][-1]]
        lat = [DATA[0]['LATITUDE'][1], DATA[1]['LATITUDE'][1], DATA[2]['LATITUDE'][1],  DATA[3]['LATITUDE'][1], DATA[4]['LATITUDE'][1], DATA[5]['LATITUDE'][1], DATA[6]['LATITUDE'][1], DATA[7]['LATITUDE'][1]]
        lon = [DATA[0]['LONGITUDE'][1], DATA[1]['LONGITUDE'][1], DATA[2]['LONGITUDE'][1], DATA[3]['LONGITUDE'][1],DATA[4]['LONGITUDE'][1], DATA[5]['LONGITUDE'][1], DATA[6]['LONGITUDE'][1], DATA[7]['LONGITUDE'][1]]
        ind_i,ind_f = 0,8   #indices dentro de la matriz_grillada
        estaciones=['1' ,'2' ,'3' ,'4' ,'5' ,'6' ,'7' ,'8']

        #Arreglo en la est 5 para graficar bonito el corte vertical
        if variable == 'T':
            T5_somero = matriz_grillada[:75,4]
            T5 = matriz_grillada[75:,4]
            T5[T5 > 5.8] = 5.5
            T5_completo = np.concatenate((T5_somero,T5))
            matriz_grillada[:,4] = T5_completo
        if variable == 'S':
            S5_somero = matriz_grillada[:75,4]
            S5 = matriz_grillada[75:,4]
            S5[S5 > 34] = 33.99
            S5_completo = np.concatenate((S5_somero,S5))
            matriz_grillada[:,4] = S5_completo
    if seccion == 2:
        prof = [DATA[8]['PRES'][-1], DATA[9]['PRES'][-1], DATA[10]['PRES'][-1], 891,DATA[12]['PRES'][-1],  DATA[13]['PRES'][-1], DATA[14]['PRES'][-1], DATA[15]['PRES'][-1],DATA[16]['PRES'][-1],DATA[17]['PRES'][-1],DATA[18]['PRES'][-1]]
        lat = [DATA[8]['LATITUDE'][1], DATA[9]['LATITUDE'][1], DATA[10]['LATITUDE'][1],  DATA[11]['LATITUDE'][1], DATA[12]['LATITUDE'][1], DATA[13]['LATITUDE'][1], DATA[14]['LATITUDE'][1], DATA[15]['LATITUDE'][1],DATA[16]['LATITUDE'][1],DATA[17]['LATITUDE'][1],DATA[18]['LATITUDE'][1]]
        lon = [DATA[8]['LONGITUDE'][1], DATA[9]['LONGITUDE'][1], DATA[10]['LONGITUDE'][1], DATA[11]['LONGITUDE'][1],DATA[12]['LONGITUDE'][1], DATA[13]['LONGITUDE'][1], DATA[14]['LONGITUDE'][1], DATA[15]['LONGITUDE'][1],DATA[16]['LONGITUDE'][1],DATA[17]['LONGITUDE'][1],DATA[18]['LONGITUDE'][1]]
        ind_i,ind_f = 8,19   #indices dentro de la matriz_grillada
        estaciones=['9' ,'10' ,'11' ,'12' ,'13' ,'14' ,'15' ,'16','17','18','19']
    if seccion == 3:
        prof = [DATA[19]['PRES'][-1], DATA[20]['PRES'][-1], DATA[21]['PRES'][-1], DATA[22]['PRES'][-1],DATA[23]['PRES'][-1],  DATA[24]['PRES'][-1], DATA[25]['PRES'][-1],DATA[26]['PRES'][-1],DATA[27]['PRES'][-1]][::-1]
        lat = [DATA[19]['LATITUDE'][1], DATA[20]['LATITUDE'][1], DATA[21]['LATITUDE'][1],  DATA[22]['LATITUDE'][1], DATA[23]['LATITUDE'][1], DATA[24]['LATITUDE'][1], DATA[25]['LATITUDE'][1], DATA[26]['LATITUDE'][1],DATA[27]['LATITUDE'][1]][::-1]
        lon = [DATA[19]['LONGITUDE'][1], DATA[20]['LONGITUDE'][1], DATA[21]['LONGITUDE'][1], DATA[22]['LONGITUDE'][1],DATA[23]['LONGITUDE'][1], DATA[24]['LONGITUDE'][1], DATA[25]['LONGITUDE'][1], DATA[26]['LONGITUDE'][1], DATA[27]['LONGITUDE'][1]][::-1]
        ind_i,ind_f = 19,28   #indices dentro de la matriz_grillada
        estaciones=['20','21','22','23','24','25','26','27','28'][::-1]
    if seccion == 4:
        prof = [DATA[28]['PRES'][-1], DATA[29]['PRES'][-1], DATA[30]['PRES'][-1], DATA[31]['PRES'][-1]]
        lat = [DATA[28]['LATITUDE'][1], DATA[29]['LATITUDE'][1], DATA[30]['LATITUDE'][1],  DATA[31]['LATITUDE'][1]]
        lon = [DATA[28]['LONGITUDE'][1], DATA[29]['LONGITUDE'][1], DATA[30]['LONGITUDE'][1], DATA[31]['LONGITUDE'][1]]
        ind_i,ind_f = 28,32   #indices dentro de la matriz_grillada
        estaciones=['29' ,'30' ,'31' ,'32']
    if seccion == 5:
        prof = [DATA[32]['PRES'][-1], DATA[33]['PRES'][-1], DATA[34]['PRES'][-1], DATA[35]['PRES'][-1],DATA[36]['PRES'][-1],  DATA[37]['PRES'][-1], DATA[38]['PRES'][-1], DATA[39]['PRES'][-1], DATA[40]['PRES'][-1]][::-1]
        lat = [DATA[32]['LATITUDE'][1], DATA[33]['LATITUDE'][1], DATA[34]['LATITUDE'][1],  DATA[35]['LATITUDE'][1], DATA[36]['LATITUDE'][1], DATA[37]['LATITUDE'][1], DATA[38]['LATITUDE'][1], DATA[39]['LATITUDE'][1], DATA[40]['LATITUDE'][1]][::-1]
        lon = [DATA[32]['LONGITUDE'][1], DATA[33]['LONGITUDE'][1], DATA[34]['LONGITUDE'][1], DATA[35]['LONGITUDE'][1],DATA[36]['LONGITUDE'][1], DATA[37]['LONGITUDE'][1], DATA[38]['LONGITUDE'][1], DATA[39]['LONGITUDE'][1], DATA[40]['LONGITUDE'][1]][::-1]
        ind_i,ind_f = 32,41   #indices dentro de la matriz_grillada
        estaciones=['33','34','35','36','37','38','39','40','41'][::-1]
    if seccion == 6:
        prof = [DATA[41]['PRES'][-1], DATA[42]['PRES'][-1], DATA[43]['PRES'][-1], DATA[44]['PRES'][-1]]
        lat = [DATA[41]['LATITUDE'][1], DATA[42]['LATITUDE'][1], DATA[43]['LATITUDE'][1],  DATA[44]['LATITUDE'][1]]
        lon = [DATA[41]['LONGITUDE'][1], DATA[42]['LONGITUDE'][1], DATA[43]['LONGITUDE'][1], DATA[44]['LONGITUDE'][1]]
        ind_i,ind_f = 41,45   #indices dentro de la matriz_grillada
        estaciones=['42','43','44','45']
    if seccion == 7:
        prof = [DATA[45]['PRES'][-1],DATA[46]['PRES'][-1], DATA[47]['PRES'][-1], DATA[48]['PRES'][-1], DATA[49]['PRES'][-1],DATA[50]['PRES'][-1],   77,78,51,40][::-1]
        lat = [DATA[45]['LATITUDE'][1],DATA[46]['LATITUDE'][1], DATA[47]['LATITUDE'][1], DATA[48]['LATITUDE'][1],  DATA[49]['LATITUDE'][1], DATA[50]['LATITUDE'][1], -31.8323, -31.7368333333, -31.64584, -31.558333333][::-1]
        lon = [DATA[45]['LONGITUDE'][1],DATA[46]['LONGITUDE'][1], DATA[47]['LONGITUDE'][1], DATA[48]['LONGITUDE'][1], DATA[49]['LONGITUDE'][1],DATA[50]['LONGITUDE'][1], -50.5816666667, -50.68952, -50.7929, -50.915216667][::-1]
        ind_i,ind_f = 45,55   #indices dentro de la matriz_grillada
        estaciones=['46','47','48','49','50','51','52','53','54','55'][::-1]

    #extension de los datos al fondo:
    AA = matriz_grillada[:,ind_i:ind_f]
    for st in range(0,len(prof)):
        primero_con_nan = len(AA[:,st][np.isnan(AA[:,st]) == False])
        AA[primero_con_nan:,st] = AA[primero_con_nan-1,st]
    if seccion in [3,5,7]:
        AA = AA[:,::-1]

    #distancia entre estaciones dentro de la seccion
    dist_entre_est = [0]*len(prof)
    dist = [0]*len(prof)
    for i in range(1,len(prof)):
        a = 90-lat[i-1]
        b = 90-lat[i]
        phi = lon[i-1]-lon[i]
        cosp = math.cos(math.radians(a))*math.cos(math.radians(b))+math.sin(math.radians(a))*math.sin(math.radians(b))*math.cos(math.radians(phi))
        p = math.degrees(math.acos(cosp))
        dist_entre_est[i] = p*6371*math.pi/180  #dist km
        dist [0] = 0
        dist [i] = dist_entre_est[i]+dist[i-1]
    #array de profundidades:
    z = np.linspace(0,4999,5000)

    #Interpolacion entre estaciones con una resolucion de 100ptos por seccion:
    dist_aux,z_aux = np.meshgrid(dist,z)
    points = (dist_aux.flatten(),z_aux.flatten())
    values = AA.flatten()
    x,p = np.meshgrid(np.linspace(0,dist[-1],100),z)
    data = griddata(points,values,(x,p),method='linear')
    if variable == 'T':
        levels = [5,7,9,11,13,15,17,19,21,23,25]
        levels_label = [5,7,9,11,13,15,17,19,21,23,25]
        cmap = plt.cm.get_cmap('coolwarm',10)
        txt = 'Temperature - Section '+ str(seccion)
        txt2 = 'C)'
        txt3 = '°C'
        vmin, vmax = 5,25
    elif variable == 'S':
        levels = [32,32.5,33,33.5,34,34.5,35,35.5,36,36.5,37]
        levels_label = [32,33,34,35,36]
        cmap =  plt.cm.get_cmap('Reds', 20)
        txt = 'Salinity - Section '+ str(seccion)
        txt2 = 'D)'
        txt3 = ''
        vmin, vmax = 32,37
    elif variable == 'O2':
        levels      = [60,80,100,120,140]
        levels_label= [60,70,80,90,100,110,120,130,140]
        cmap = plt.cm.get_cmap('Purples',4)
        txt = 'Dissolved Oxygen Saturation - Section ' + str(seccion)
        txt2 = ''
        txt3 = '%'
        vmin,vmax = 60,140
    elif variable == 'Fluo':
        levels = [0,1,2,3,4,5]
        levels_label = levels
        cmap = plt.cm.get_cmap('Greens',5)
        txt = 'Fluorescence - Section '+str(seccion)
        txt2 = ''
        txt3 = ''
        vmin,vmax = 0,5

    #Suavizado
    sigma = 0.99 #this depends on how noisy your data is, play with it!
    data = suavizado(data)

    fig1 = plt.figure(figsize=(16,8))
    CF = plt.contourf(x[:1000,:],p[:1000,:],data[:1000,:],levels,cmap=cmap,vmin=vmin,vmax=vmax,extend = 'max')
    if variable == 'O2':
        CC = plt.contour(x[:1000,:],p[:1000,:],data[:1000,:],levels_label, colors=('k'),linewidths=1.5)
    else:
        CC = plt.contour(x[:1000,:],p[:1000,:],data[:1000,:],levels, colors=('k'),linewidths=1.5)
    plt.clabel(CC,levels_label,inline=1, fmt='%1.1f', fontsize=24)
    #Arreglos particulares:
    if seccion == 1 and variable == 'T':
        c6 = plt.contour(x[:1000,:],p[:1000,:],data[:1000,:],[6],colors=('k'),linewidths = 1,alpha=0.8)
        plt.clabel(c6,[6],inline=1, fmt='%1.1f', fontsize=24)
    if seccion == 1 and variable == 'S':
        c33_8 = plt.contour(x[:1000,:],p[:1000,:],data[:1000,:],[33.5,33.8,34],colors=('k'),linewidths = 1,alpha=0.8)
        plt.clabel(c33_8,[33.8],inline=1, fmt='%1.1f', fontsize=24)

    for est in range(0,len(prof)):
        plt.text(dist[est]-0.2,-4,estaciones[est],size = 30)
    plt.scatter(dist,np.zeros(len(dist)),marker = "v",color = 'k',s=200)

    # plt.text(5,180,txt,size = 30, bbox=dict(facecolor='white'))
    plt.plot(dist,prof,'*-')
    plt.axis([0,dist[-1],200,0])
    plt.xlabel('Distance (km)', size=30)
    plt.ylabel('Pressure (db)', size=30)
    plt.xticks(size = 30)
    plt.yticks([0,50,100,150,200],size = 30)
    plt.text(3,195,txt2,size = 30)

    cbar = fig1.colorbar(CF,orientation='vertical')
    cbar.set_label(label=txt3,fontsize=30)
    cbar.ax.tick_params(labelsize=24)

    plt.fill_between(dist, 1000, prof, facecolor='white',zorder = 2)
    plt.show()

def TS_plot(T,S, seccion, estaciones_a_plotear,letra):
    """
    Plotea el Diagrama TS de la seccion.

    Parámetros de entrada
    --------------------
    T: 2D numpy.array (5000,55):(prof,estacion) de Temp regrillado c/1metro
    S: 2D numpy.array (5000,55):(prof,estacion) de Sal regrillado c/1metro
    seccion: 1,2,3,4,5,6 o 7 (int). Seccion a plotear
    estaciones_a_plotear: 1D numpy.array con los numeros de estaciones a plotear. ['1','2','3']
    letra: string.
    Parámetros de salida
    -------------------
    plot del diagrama TS

    """
    #ingreso los datos de cada seccion
    if seccion == 1:
        ind_i,ind_f = 0,8   #indices dentro de la matriz_grillada
        estaciones=['1' ,'2' ,'3' ,'4' ,'5' ,'6' ,'7' ,'8']
    if seccion == 2:
        ind_i,ind_f = 8,19   #indices dentro de la matriz_grillada
        estaciones=['9' ,'10' ,'11' ,'12' ,'13' ,'14' ,'15' ,'16','17','18','19']
    if seccion == 3:
        ind_i,ind_f = 19,28   #indices dentro de la matriz_grillada
        estaciones=['20','21','22','23','24','25','26','27','28']
    if seccion == 4:
        ind_i,ind_f = 28,32   #indices dentro de la matriz_grillada
        estaciones=['29' ,'30' ,'31' ,'32']
    if seccion == 5:
        ind_i,ind_f = 32,41   #indices dentro de la matriz_grillada
        estaciones=['33','34','35','36','37','38','39','40','41']
    if seccion == 6:
        ind_i,ind_f = 41,45   #indices dentro de la matriz_grillada
        estaciones=['42','43','44','45']
    if seccion == 7:
        ind_i,ind_f = 45,55   #indices dentro de la matriz_grillada
        estaciones=['46','47','48','49','50','51','52','53','54','55']

    T_seccion = T[:,ind_i:ind_f]
    S_seccion = S[:,ind_i:ind_f]
    #check si las estaciones a plotear estan dentro de la seccion:

    ################PREPARO LA FIG###########################
    # Bordes de la figura
    smin, smax = 30, 37
    tmin, tmax = 3, 23

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

    colores = ['blue','k','tomato','green']
    labels = estaciones_a_plotear

    fig = plt.figure()
    ax = fig.add_axes([0.1,0.12,0.65,0.85])
    if seccion == 7:
        ax.set_xlabel('Salinity', size=34)
    ax.set_ylabel('Temperature (°C)', size=34)
    plt.xticks(size = 30)
    plt.yticks([8,13,18,23],size = 34)
    ax.tick_params(which='both', width=2)
    plt.xticks([31,32,33,34,35,36],size = 34)
    ax.axis([30,37,3,23])
       #contornos de densidad
    CS = ax.contour(si,ti,densidad, linestyles='dashed', colors='k',alpha=0.5)
    ax.clabel(CS, fontsize=22, inline=1, fmt='%1.0f') # Label every second level

    for k in range(0,len(estaciones_a_plotear)):
        ax.scatter(S_seccion[:,estaciones.index(estaciones_a_plotear[k])],T_seccion[:,estaciones.index(estaciones_a_plotear[k])],c=colores[k],s=1,label=labels[k])
    plt.legend(fontsize = 26,markerscale =16,loc=2)

    #nombre de las masas de agua
    ax.text(31.2, 18.5, 'RDP',size=34, color='k')
    ax.text(36.5, 19.7, 'TW',size=34,color='k')
    ax.text(32.4, 9, 'SASW',size=34,color='k')
    ax.arrow(33.1, 9.2, 0.3, 0, width=0.05,head_width=0.3, head_length=0.1,color='k' )
    ax.text(35.6, 12, 'SACW',size=34,color='k')
    ax.scatter(34.2,3.5, c='k',s=50,alpha=0.6)
    ax.text(33.5,3.4,'AAIW',size=34,color='k')

    ax.text(30.2,4,letra+')',size=30)

    ax.scatter(S,T,s=1,c='grey',alpha=0.2)
    #ploteo las estaciones:
    # for k in range(0,len(estaciones_a_plotear)):
    for k in range(len(estaciones_a_plotear)):
        ax.scatter(S_seccion[:,estaciones.index(estaciones_a_plotear[k])],T_seccion[:,estaciones.index(estaciones_a_plotear[k])],c=colores[k],s=50,label=labels[k])
        ax.plot(S_seccion[:,estaciones.index(estaciones_a_plotear[k])],T_seccion[:,estaciones.index(estaciones_a_plotear[k])],c=colores[k],lw= 1)

    #PARA FIG PAPER
    if seccion == 2 and estaciones_a_plotear == ['12','17']:
        c2 = patches.Ellipse((33.8,14.6), width=0.7, height=2.2,edgecolor = 'g',fill = False,linewidth= 2.8,linestyle='-')
        c3 = patches.Ellipse((33.8,10.4), width=0.7, height=2.2,edgecolor = 'b',fill = False,linewidth= 2.8,linestyle='-')
        c1 = patches.Ellipse((34.2,7.1),   width=0.5, height=2.2,edgecolor = 'r',fill = False,linewidth= 2.8,linestyle='-')
        ax.add_artist(c1)
        ax.add_artist(c2)
        ax.add_artist(c3)
    if seccion == 2 and estaciones_a_plotear == ['9','15','19']:
        c2 = patches.Ellipse((33.9,14.8), width=0.7, height=2.2,edgecolor = 'g',fill = False,linewidth= 2.8,linestyle='-')     #EST 12/17
        c3 = patches.Ellipse((33.75,10.1), width=0.7, height=2.2,edgecolor = 'b',fill = False,linewidth= 2.8,linestyle='-')
        ax.add_artist(c2)
        ax.add_artist(c3)
    elif seccion == 1:
        c1 = patches.Ellipse((33.9,6.2),   width=0.5, height=2.2,edgecolor = 'r',fill = False,linewidth= 2.8,linestyle='-')   #st 9/15/19
        ax.add_artist(c1)
    elif seccion == 3:
        c1 = patches.Ellipse((33.1,13.1),   width=0.5, height=2.2,edgecolor = 'g',fill = False,linewidth= 2.8,linestyle='-')   #st 9/15/19
        c2 = patches.Ellipse((34.3,13), width=0.7, height=2.2,edgecolor = 'k',fill = False,linewidth= 2.8,linestyle='-')     #EST 12/17
        c3 = patches.Ellipse((35.4,14.2), width=0.7, height=2.2,edgecolor = 'k',fill = False,linewidth= 2.8,linestyle='-')
        ax.add_artist(c1)
        ax.add_artist(c2)
        ax.add_artist(c3)
    elif seccion == 5:
        c1 = patches.Ellipse((35,13.5),width=0.5, height=2.2,edgecolor = 'r',fill = False,linewidth= 2.8,linestyle='-')   #st 9/15/19
        ax.add_artist(c1)
        ax.text(30.1,11.5,'<-- $15^\circ$C, < 28',fontsize = 24)

    plt.show()

def mapa_sal_vel(lat_CTD,lon_CTD,z0,S,U,V,seccion):
    """
    Plotea la salinidad a la profundidad deseada con las velocidades sobre las estaciones.
     Para ello interpola usando griddata de scipy y suaviza la imagen.

    Parámetros de entrada
    --------------------
    S: 2D numpy.array (5000,55):(prof,estacion) de Salinidad regrillado c/1metro
    z: int entre 0 y 5000, profundidad a la que se desea plotear
    lat : 1D numpy.array (55):latitudes de las estaciones de CTD
    lon : 1D numpy.array (55):longitudes de las estaciones de CTD
    U: 2D numpy.array(5000,55):(prof,estacion) de velocidad zonal regrillado c/1metro
    V: 2D numpy.array(5000,55):(prof,estacion) de velocidad meridional regrillado c/1metro
    seccion: 1,2,3,4,5,6 o 7, int numero de seccion en donde quiero plotear las veloc

    Parámetros de salida
    -------------------
    plot
    data: matriz_grillada de la Sal con la veloc de ADCP sobre las estaciones a la prof z0

    """
    #DOMINIO
    lon_min, lon_max = -60, -48
    lat_min, lat_max = -40, -30
    x = np.linspace(lon_min,lon_max,500)
    y = np.linspace(lat_min,lat_max,500)
    x,y = np.meshgrid(x,y)

    #Agrego Termosal si estoy en superficie
    if int(z0) == 0:
        #abrir archivos del termosal
        lon_TS,lat_TS,temp_TS,sal_TS = carga_datos_termosal()

        yy,yy_f = 25180-1700,25184+800
        sal_TS[yy:yy_f] = 26

        LONS, LATS = np.concatenate([lon_TS,lon_CTD]), np.concatenate([lat_TS,lat_CTD])
        Ssup_aux =  np.concatenate([sal_TS,S[int(z0),:]])

        """ Arreglo """
        lat_arreglo1 = [-38 ,-34.8 ,-34.8 ,-34.8 ,-37   ,-36   ,-34  ,-32 ,-35.2,-36 ,-36  ,-36  ,-32.8,-37.8,-37  ,-36.5,-34.9]
        lon_arreglo1 = [-56 ,-54   ,-54.5 ,-55   ,-57   ,-57   ,-51.5,-51 ,-51.8,-53 ,-52.9,-53.1,-51  ,-54  ,-55  ,-54.5,-52.6 ]
        s_arreglo1 =   [33.8,28    ,28    ,28    ,np.nan,np.nan,32.5 ,25  ,35   ,32.5,33   ,33   ,28   ,34.8 ,33.8 ,33.8 , 31.5]

        lat_arreglo2 = [-37.45 ,-37.35,-37.35   ,-36.8 ,-38   ]
        lon_arreglo2 = [-52.7,-52.75  ,-52.8    ,-52.7 ,-57   ]
        s_arreglo2 =   [34.5  ,33     ,34.5     ,33.5  , 33.8 ]

        lat_arreglo = np.concatenate([lat_arreglo1,lat_arreglo2])
        lon_arreglo = np.concatenate([lon_arreglo1,lon_arreglo2])
        s_arreglo = np.concatenate([s_arreglo1,s_arreglo2])

        LONS, LATS = np.concatenate([LONS,lon_arreglo]), np.concatenate([LATS,lat_arreglo])
        Ssup = np.concatenate([Ssup_aux,s_arreglo])

        lon,lat = LONS, LATS
    else:
        lon,lat = lon_CTD,lat_CTD
        Ssup = S[int(z0),:]

    #Interpolación de T en el dominio:
    points = (lon,lat)
    values = Ssup
    S_grillada = griddata(points,values,(x,y))
    # Suavizado
    data = suavizado(S_grillada)
    ## Figure
    cmap_s = plt.cm.get_cmap('Reds', 7)
    fig = plt.figure()
    ax = fig.add_axes([0.05,0.05,0.8,0.9],projection=ccrs.PlateCarree())

    ax.set_extent([-60,-48,-40,-30])
    ax.add_feature(cfeature.LAND, color='#BDA973')
    ax.add_feature(cfeature.LAKES, color='lightcyan')
    ax.add_feature(cfeature.RIVERS, edgecolor='black')
    ax.coastlines(resolution='50m', color='black', linewidth=1)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='black', alpha=0.2, linestyle='-')
    gl.xlabels_top = False; gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator([-60,-58,-56,-54,-52,-50,-48])
    gl.ylocator = mticker.FixedLocator([-40,-38,-36,-34,-32,-30])
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 24, 'color': 'k'}
    gl.ylabel_style = {'size': 24, 'color': 'k'}

    #isobata 200m
    lon200,lat200 = isobata_200()
    plt.plot(lon200,lat200,color='grey',alpha = 0.5)

    levels = [30,31,32,33,34,35,36,37]
    cr = plt.contourf(x,y,data,levels,cmap=cmap_s)
    plt.contour(x,y, data,levels, colors=('k'),linewidths=1)
    #barra de color
    cbar = plt.colorbar(cr, orientation='vertical', ticks = levels)
    cbar.ax.tick_params(labelsize=24)
    plt.scatter(lon_CTD,lat_CTD,s = 20,color = 'k')

    if seccion == 3:        ind_i,ind_f = 19,28   #indices dentro de la matriz_grillada

    #Esr 21,23,26
    lon3 = np.array([lon_CTD[ind_i+6],lon_CTD[ind_i+3],lon_CTD[ind_i+1]])
    lat3 = np.array([lat_CTD[ind_i+6],lat_CTD[ind_i+3],lat_CTD[ind_i+1]])
    U3 = np.array([U[z0,ind_i+6],U[z0,ind_i+3],U[z0,ind_i+1]])
    V3 = np.array([V[z0,ind_i+6],V[z0,ind_i+3],V[z0,ind_i+1]])


    q21 = plt.quiver(lon3[0],lat3[0],U3[0],V3[0],scale=800,width = 0.005,color = 'b',zorder = 5,headwidth = 2,headlength = 3)
    q23 = plt.quiver(lon3[1],lat3[1],U3[1],V3[1],scale=600,width = 0.005,color = 'g',zorder = 5,headwidth = 2,headlength = 3)
    q26 = plt.quiver(lon3[2],lat3[2],U3[2],V3[2],scale=1000,width = 0.005,color = 'r',zorder = 5,headwidth = 2,headlength = 3)

    plt.show()



def mapa_batimetrico_paper(lat_min,lat_max,lon_min,lon_max,OSCAR = 'si'):
    """
    Plot el mapa batimetrico usando ETOPO entre los limites dados

    Parametros de entrada
    --------------------
    lat_min: 1D numpy.array limite del mapa
    lat_max: 1D numpy.array limite del mapa
    lon_min: 1D numpy.array limite del mapa
    lon_max: 1D numpy.array limite del mapa
    OSCAR  : 'si' o 'no'. Velocidades de OSCAR?

    Parametros de salinidad
    --------------------
    plot

    """

    LATS,LONS = lat_lon_CTD()
    path = path_gral+'Doctorado/Datos/batimetria/ETOPO1_Bed_g_gmt4.grd'

    DATA = Dataset(path,'r')
    DATA.set_auto_mask(False)

    x = DATA['x'][:]
    y = DATA['y'][:]
    z = DATA['z'][:]

    x_STSF = x[7200:7921]   #lons=[-60,-48]
    y_STSF = y[3000:3601]   #lats=[-40:-30]
    z_STSF = z[3000:3601,7200:7921]
    interior = z_STSF > 0
    z_STSF[interior] = 3000

    fig = plt.figure()
    ax = plt.axes(projection=ccrs.Mercator())
    ax.set_extent([-60,-48,-40,-30],crs = ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, color='#BDA973',zorder = 3)
    ax.add_feature(cfeature.LAKES, color='lightcyan')
    ax.add_feature(cfeature.RIVERS, edgecolor='black')
    ax.coastlines(resolution='50m', color='black', linewidth=1,zorder = 4)
    gl = ax.gridlines(crs = ccrs.PlateCarree(),draw_labels=True,
    linewidth=1, color='black', alpha=0.5, linestyle='-')
    gl.xlabels_top = False; gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator([-60,-58,-56,-54,-52,-50,-48])
    gl.ylocator = mticker.FixedLocator([-40,-38,-36,-34,-32,-30])
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 24, 'color': 'k'}
    gl.ylabel_style = {'size': 24, 'color': 'k'}

    cmap=plt.cm.get_cmap('cmo.tempo_r', 40)
    levels = [-4000,-3500,-3000,-2500,-2000,-1500,-1000,-500,-200,-100,0]
    levels_iso = [-2000,-200]
    BATH = ax.contourf(x_STSF,y_STSF,z_STSF,levels, cmap=cmap,vmin=-4000, vmax=500,extend='both',transform=ccrs.PlateCarree(),alpha = 0.8)
    BATH_ = ax.contour(x_STSF,y_STSF,z_STSF,levels_iso,cmap='Greys',vmin=-50000,vmax=-180,linewidths = 1)
    cbar = plt.colorbar(BATH,ticks=levels)
    cbar.ax.set_yticklabels(-1*np.array(levels))
    cbar.ax.tick_params(labelsize=18)
    cbar.ax.set_ylabel('Meters',fontsize=20)

    #Termosal:
    LONS_TS,LATS_TS,temp_TS, sal_TS = carga_datos_termosal()
    plt.scatter(LONS_TS[::50],LATS_TS[::50],s = 10,color = 'gray',transform=ccrs.PlateCarree(),alpha = 0.5,zorder = 2)

    #######################################
    #Agrego veloc de oscar
    if OSCAR == 'si':
        path = path_gral+'Doctorado/Datos/satelital/oscar_vel2013.nc.gz.nc4'
        data_VEL = Dataset(path)
        #Recorto sobre mi dominio
        lon_vel = data_VEL['longitude'][840:877]
        lat_vel = data_VEL['latitude'][330:361]
        time = data_VEL['time'][:]
        #Fecha de la campania: time[55] (2013-10-06)
        # dates = []
        # for t in range(len(time)):
        #     dates.append(netCDF4.num2date(time[t], data_VEL['time'].units,'gregorian'))
        t0 = 55
        u = data_VEL['u'][:,0,330:361,840:877]
        v = data_VEL['v'][:,0,330:361,840:877]

        x_vel,y_vel = np.meshgrid(lon_vel,lat_vel)
        cs = ax.quiver(x_vel,y_vel,(u[t0-1,:,:]+u[t0,:,:])/2,(v[t0-1,:,:]+v[t0,:,:])/2,color='0.2',units='width',scale=12,transform = ccrs.PlateCarree(),zorder = 3)
        ax.quiverkey(cs, 0.4, 0.81, 0.5, r'$ 50\frac{cm}{s}$', labelpos='E',coordinates='figure',fontproperties={'size' :24})
        # ax.text(15000,1000000,'Currents (OSCAR)\nMean between 3 and 10 October',size=12,bbox=dict(facecolor='white'))
    #######################################
    # Estaciones CTD:
    for Nest in range(1):
        ax.scatter(LONS,LATS,s=25,color='k',transform=ccrs.PlateCarree(),zorder = 4)
        est = ['1',        '8', '9',  '19', '28','20',  '33', '41',  '46'   ,'55']
        lon_est = [-55.8,-54.3,-54.2,-52.2,-54.1,-52.0,-51.1,-52.6, -49.5,-51.2]
        lat_est = [-37.2,-37.8,-36.8,-36.8,-35.0,-35.6,-34.4,-33.3,-32.1  ,-31.2]

        for st in range (len(est)):
            ax.text(lon_est[st],lat_est[st],est[st],fontsize=16,bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round'),transform=ccrs.PlateCarree())
        T = [   'T1',   'T2', 'T3',  'T5','T7']
        lon_T = [-55.6,-53.1,-53.2,-51.8,-50.3]
        lat_T = [-38.2,-37.7,-36.1,-33.8,-31.7]

        for ti in range(len(T)):
            ax.text(lon_T[ti],lat_T[ti],T[ti],color='darkred',fontsize=14,bbox=dict(facecolor='white', edgecolor='r', boxstyle='circle'),transform=ccrs.PlateCarree(),zorder=4)

    ## Lugares:
    # Rio grande
    lat_rio_grande = -32-1/60-6/3600
    lon_rio_grande = -52-5/60-45/3600
    ax.scatter(lon_rio_grande,lat_rio_grande,s=50,color='b',transform=ccrs.PlateCarree(),zorder=5)
    ax.text(lon_rio_grande-3.4,lat_rio_grande,'Rio Grande',color ='darkblue',fontsize=24,zorder=5,transform=ccrs.PlateCarree())
    #RDP
    lat_RDP = -35.5
    lon_RDP = -56.8
    ax.text(lon_RDP,lat_RDP,'RDP',color ='k',fontsize=24,zorder=5,transform=ccrs.PlateCarree())

def corte_seccion_paper(seccion,variable,T,S,DATA,letra):
    """
    Plotea el corte vertical de la variable.

    Parámetros de entrada
    --------------------
    seccion: 1,2,3,4,5,6 o 7 (int). Seccion a plotear
    Variable: 'T' o 'S' o 'O2' o  'Fluo' (string) variable a plotear
    T: 2D numpy.array (5000,55):(prof,estacion) de Temp regrillado c/1metro
    s: 2D numpy.array (5000,55):(prof,estacion) de Sal regrillado c/1metro
    DATA: lista, cada elemento corresponde al archivo cnv de cada estacion.
    letra:int

    Parámetros de salida
    -------------------
    plot

    """
    #ingreso los datos de cada seccion
    if seccion == 1:
        prof = [DATA[0]['PRES'][-1], DATA[1]['PRES'][-1], DATA[2]['PRES'][-1], DATA[3]['PRES'][-1],DATA[4]['PRES'][-1],  DATA[5]['PRES'][-1], DATA[6]['PRES'][-1], DATA[7]['PRES'][-1]]
        lat = [DATA[0]['LATITUDE'][1], DATA[1]['LATITUDE'][1], DATA[2]['LATITUDE'][1],  DATA[3]['LATITUDE'][1], DATA[4]['LATITUDE'][1], DATA[5]['LATITUDE'][1], DATA[6]['LATITUDE'][1], DATA[7]['LATITUDE'][1]]
        lon = [DATA[0]['LONGITUDE'][1], DATA[1]['LONGITUDE'][1], DATA[2]['LONGITUDE'][1], DATA[3]['LONGITUDE'][1],DATA[4]['LONGITUDE'][1], DATA[5]['LONGITUDE'][1], DATA[6]['LONGITUDE'][1], DATA[7]['LONGITUDE'][1]]
        ind_i,ind_f = 0,8   #indices dentro de la matriz
        estaciones=['1' ,'2' ,'3' ,'4' ,'5' ,'6' ,'7' ,'8']
        #Arreglo en la est 5 para graficar bonito el corte vertical
        T5_somero = T[:75,4]
        T5 = T[75:,4]
        T5[T5 > 5.8] = 5.5
        T5_completo = np.concatenate((T5_somero,T5))
        T[:,4] = T5_completo
        S5_somero = S[:75,4]
        S5 = S[75:,4]
        S5[S5 > 34] = 33.99
        S5_completo = np.concatenate((S5_somero,S5))
        S[:,4] = S5_completo
    if seccion == 2:
        prof = [DATA[8]['PRES'][-1], DATA[9]['PRES'][-1], DATA[10]['PRES'][-1], 891,DATA[12]['PRES'][-1],  DATA[13]['PRES'][-1], DATA[14]['PRES'][-1], DATA[15]['PRES'][-1],DATA[16]['PRES'][-1],DATA[17]['PRES'][-1],DATA[18]['PRES'][-1]]
        lat = [DATA[8]['LATITUDE'][1], DATA[9]['LATITUDE'][1], DATA[10]['LATITUDE'][1],  DATA[11]['LATITUDE'][1], DATA[12]['LATITUDE'][1], DATA[13]['LATITUDE'][1], DATA[14]['LATITUDE'][1], DATA[15]['LATITUDE'][1],DATA[16]['LATITUDE'][1],DATA[17]['LATITUDE'][1],DATA[18]['LATITUDE'][1]]
        lon = [DATA[8]['LONGITUDE'][1], DATA[9]['LONGITUDE'][1], DATA[10]['LONGITUDE'][1], DATA[11]['LONGITUDE'][1],DATA[12]['LONGITUDE'][1], DATA[13]['LONGITUDE'][1], DATA[14]['LONGITUDE'][1], DATA[15]['LONGITUDE'][1],DATA[16]['LONGITUDE'][1],DATA[17]['LONGITUDE'][1],DATA[18]['LONGITUDE'][1]]
        ind_i,ind_f = 8,19   #indices dentro de la matriz
        estaciones=['9' ,'10' ,'11' ,'12' ,'13' ,'14' ,'15' ,'16','17','18','19']
    if seccion == 3:
        prof = [DATA[19]['PRES'][-1], DATA[20]['PRES'][-1], DATA[21]['PRES'][-1], DATA[22]['PRES'][-1],DATA[23]['PRES'][-1],  DATA[24]['PRES'][-1], DATA[25]['PRES'][-1],DATA[26]['PRES'][-1],DATA[27]['PRES'][-1]][::-1]
        lat = [DATA[19]['LATITUDE'][1], DATA[20]['LATITUDE'][1], DATA[21]['LATITUDE'][1],  DATA[22]['LATITUDE'][1], DATA[23]['LATITUDE'][1], DATA[24]['LATITUDE'][1], DATA[25]['LATITUDE'][1], DATA[26]['LATITUDE'][1],DATA[27]['LATITUDE'][1]][::-1]
        lon = [DATA[19]['LONGITUDE'][1], DATA[20]['LONGITUDE'][1], DATA[21]['LONGITUDE'][1], DATA[22]['LONGITUDE'][1],DATA[23]['LONGITUDE'][1], DATA[24]['LONGITUDE'][1], DATA[25]['LONGITUDE'][1], DATA[26]['LONGITUDE'][1], DATA[27]['LONGITUDE'][1]][::-1]
        ind_i,ind_f = 19,28   #indices dentro de la matriz
        estaciones=['20','21','22','23','24','25','26','27','28'][::-1]
    if seccion == 4:
        prof = [DATA[28]['PRES'][-1], DATA[29]['PRES'][-1], DATA[30]['PRES'][-1], DATA[31]['PRES'][-1]]
        lat = [DATA[28]['LATITUDE'][1], DATA[29]['LATITUDE'][1], DATA[30]['LATITUDE'][1],  DATA[31]['LATITUDE'][1]]
        lon = [DATA[28]['LONGITUDE'][1], DATA[29]['LONGITUDE'][1], DATA[30]['LONGITUDE'][1], DATA[31]['LONGITUDE'][1]]
        ind_i,ind_f = 28,32   #indices dentro de la matriz
        estaciones=['29' ,'30' ,'31' ,'32']
    if seccion == 5:
        prof = [DATA[32]['PRES'][-1], DATA[33]['PRES'][-1], DATA[34]['PRES'][-1], DATA[35]['PRES'][-1],DATA[36]['PRES'][-1],  DATA[37]['PRES'][-1], DATA[38]['PRES'][-1], DATA[39]['PRES'][-1], DATA[40]['PRES'][-1]][::-1]
        lat = [DATA[32]['LATITUDE'][1], DATA[33]['LATITUDE'][1], DATA[34]['LATITUDE'][1],  DATA[35]['LATITUDE'][1], DATA[36]['LATITUDE'][1], DATA[37]['LATITUDE'][1], DATA[38]['LATITUDE'][1], DATA[39]['LATITUDE'][1], DATA[40]['LATITUDE'][1]][::-1]
        lon = [DATA[32]['LONGITUDE'][1], DATA[33]['LONGITUDE'][1], DATA[34]['LONGITUDE'][1], DATA[35]['LONGITUDE'][1],DATA[36]['LONGITUDE'][1], DATA[37]['LONGITUDE'][1], DATA[38]['LONGITUDE'][1], DATA[39]['LONGITUDE'][1], DATA[40]['LONGITUDE'][1]][::-1]
        ind_i,ind_f = 32,41   #indices dentro de la matriz
        estaciones=['33','34','35','36','37','38','39','40','41'][::-1]
    if seccion == 6:
        prof = [DATA[41]['PRES'][-1], DATA[42]['PRES'][-1], DATA[43]['PRES'][-1], DATA[44]['PRES'][-1]]
        lat = [DATA[41]['LATITUDE'][1], DATA[42]['LATITUDE'][1], DATA[43]['LATITUDE'][1],  DATA[44]['LATITUDE'][1]]
        lon = [DATA[41]['LONGITUDE'][1], DATA[42]['LONGITUDE'][1], DATA[43]['LONGITUDE'][1], DATA[44]['LONGITUDE'][1]]
        ind_i,ind_f = 41,45   #indices dentro de la matriz
        estaciones=['42','43','44','45']
    if seccion == 7:
        prof = [DATA[45]['PRES'][-1],DATA[46]['PRES'][-1], DATA[47]['PRES'][-1], DATA[48]['PRES'][-1], DATA[49]['PRES'][-1],DATA[50]['PRES'][-1],   77,78,51,40][::-1]
        lat = [DATA[45]['LATITUDE'][1],DATA[46]['LATITUDE'][1], DATA[47]['LATITUDE'][1], DATA[48]['LATITUDE'][1],  DATA[49]['LATITUDE'][1], DATA[50]['LATITUDE'][1], -31.8323, -31.7368333333, -31.64584, -31.558333333][::-1]
        lon = [DATA[45]['LONGITUDE'][1],DATA[46]['LONGITUDE'][1], DATA[47]['LONGITUDE'][1], DATA[48]['LONGITUDE'][1], DATA[49]['LONGITUDE'][1],DATA[50]['LONGITUDE'][1], -50.5816666667, -50.68952, -50.7929, -50.915216667][::-1]
        ind_i,ind_f = 45,55   #indices dentro de la matriz
        estaciones=['46','47','48','49','50','51','52','53','54','55'][::-1]

    #extension de los datos al fondo:
    AA_T = T[:,ind_i:ind_f]
    AA_S = S[:,ind_i:ind_f]
    for st in range(0,len(prof)):
        primero_con_nan_T = len(AA_T[:,st][np.isnan(AA_T[:,st]) == False])
        primero_con_nan_S = len(AA_S[:,st][np.isnan(AA_S[:,st]) == False])
        AA_T[primero_con_nan_T:,st] = AA_T[primero_con_nan_T-1,st]
        AA_S[primero_con_nan_S:,st] = AA_S[primero_con_nan_S-1,st]
    if seccion in [3,5,7]:
        AA_T = AA_T[:,::-1]
        AA_S = AA_S[:,::-1]

    #distancia entre estaciones dentro de la seccion
    dist_entre_est = [0]*len(prof)
    dist = [0]*len(prof)
    for i in range(1,len(prof)):
        a = 90-lat[i-1]
        b = 90-lat[i]
        phi = lon[i-1]-lon[i]
        cosp = math.cos(math.radians(a))*math.cos(math.radians(b))+math.sin(math.radians(a))*math.sin(math.radians(b))*math.cos(math.radians(phi))
        p = math.degrees(math.acos(cosp))
        dist_entre_est[i] = p*6371*math.pi/180  #dist km
        dist [0] = 0
        dist [i] = dist_entre_est[i]+dist[i-1]
    #array de profundidades:
    z = np.linspace(0,4999,5000)

    #Interpolacion entre estaciones con una resolucion de 100ptos por seccion:
    dist_aux,z_aux = np.meshgrid(dist,z)
    points = (dist_aux.flatten(),z_aux.flatten())
    values_T = AA_T.flatten()
    values_S = AA_S.flatten()
    x,p = np.meshgrid(np.linspace(0,dist[-1],100),z)
    data_T = griddata(points,values_T,(x,p),method='linear')
    data_S = griddata(points,values_S,(x,p),method='linear')
    #Suavizado
    sigma = 0.99 #this depends on how noisy your data is, play with it!
    data_T = suavizado(data_T)
    data_S = suavizado(data_S)
    #Densidad: sigma kg/m^3
    data_D = gsw.dens(data_S,data_T,0)-1000
    if variable == 'T':
        levels = [5,7,9,11,13,15,17,19,21,23,25]
        levels_label = [5,7,9,11,13,15,17,19,21,23,25]
        cmap = plt.cm.get_cmap('coolwarm',10)
        txt = 'Temperature - Section '+ str(seccion)
        txt2 = str(letra)+')'
        txt3 = '°C'
        vmin, vmax = 5,25
        data = data_T
    elif variable == 'S':
        levels = [32,32.5,33,33.5,34,34.5,35,35.5,36,36.5,37]
        levels_label = [32,33,34,35,36]
        cmap =  plt.cm.get_cmap('Reds', 20)
        txt = 'Salinity - Section '+ str(seccion)
        txt2 = str(letra)+')'
        txt3 = ''
        vmin, vmax = 32,37
        data = data_S

    fig1 = plt.figure(figsize=(16,8))
    CF = plt.contourf(x[:1000,:],p[:1000,:],data[:1000,:],levels,cmap=cmap,vmin=vmin,vmax=vmax,extend = 'max')
    CC = plt.contour(x[:1000,:],p[:1000,:],data[:1000,:],levels, colors=('k'),linewidths=1.5)
    CD = plt.contour(x[:1000,:],p[:1000,:],data_D[:1000,:],levels = [25,26],colors=('k'),linestyles = 'dashed', linewidths=2)
    plt.clabel(CC,levels_label,inline=1, fmt='%1.1f', fontsize=24)
    #Arreglos particulares:
    if seccion == 1 and variable == 'T':
        c6 = plt.contour(x[:1000,:],p[:1000,:],data[:1000,:],[6],colors=('k'),linewidths = 1,alpha=0.8)
        plt.clabel(c6,[6],inline=1, fmt='%1.1f', fontsize=24)
    if seccion == 1 and variable == 'S':
        c33_8 = plt.contour(x[:1000,:],p[:1000,:],data[:1000,:],[33.5,33.8,34],colors=('k'),linewidths = 1,alpha=0.8)
        plt.clabel(c33_8,[33.8],inline=1, fmt='%1.1f', fontsize=24)

    for est in range(0,len(prof)):
        plt.text(dist[est]-0.2,-4,estaciones[est],size = 30)
    plt.scatter(dist,np.zeros(len(dist)),marker = "v",color = 'k',s=200)

    # plt.text(5,180,txt,size = 30, bbox=dict(facecolor='white'))
    plt.plot(dist,prof,'*-')
    plt.axis([0,132.7,200,0])
    plt.xlabel('Distance (km)', size=30)
    plt.ylabel('Pressure (db)', size=30)
    plt.xticks(size = 30)
    plt.yticks([0,50,100,150,200],size = 30)
    plt.text(3,195,txt2,size = 30)

    cbar = fig1.colorbar(CF,orientation='vertical')
    cbar.set_label(label=txt3,fontsize=30)
    cbar.ax.tick_params(labelsize=24)

    plt.fill_between(dist, 1000, prof, facecolor='white',zorder = 2)
    plt.show()








##
