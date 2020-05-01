"""
Funciones para el calculo del transporte sobre la isob de 200 usando
los datos de la campaña STSF2013
"""
# path_gral = '/home/california/Documents/'
path_gral = '/media/giuliana/DE_BERDEN/'
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
from seabird import fCNV
import math
import numpy.matlib
import matplotlib.pyplot as plt
from haversine import haversine
import f_CTD_exploratorio as fCTD
import f_ADCP_exploratorio as fADCP
import seawater as sw
import netCDF4
import xarray as xr

""" Funciones p/calculo con ADCP"""
def batimetria_sobre_transecta(lon_T,lat_T,fuente):
    """
    Obtiene la batiemtria sobre una transecta segun la batimetria GEBCO

    Parámetros de entrada
    -------------------
    lon_T : 1D numpy.array. Longitude de la transecta
    lat_T : 1D numpy.array. Latitudes de la transecta
    fuente: 'STSF' o 'NEMO' o 'GEBCO' o 'ETOPO' o 'Mati' o 'Palma' o 'Topex', string. Fuente de la base de batimetriaself.
    #### Mati. Batimetria hecha pr Deigo y Moira en su tesis interpolada por Matias Dinapoli (Quizas en el Dinapoli et al., 2019)

    Parametros de salida
    -------------------
    z_T: 1D numpy.array. Batimetria sobre T basado en 'fuente'.

    """
    if fuente == 'NEMO':
        anio = 2013
        DATA = Dataset(path_gral+'Doctorado/Datos/Copernicus/0.083degree/reanalisis/Anual_hasta643/global-reanalysis-phy-001-030-daily_'+str(anio)+'_hasta643m.nc')
        lat = DATA.variables['latitude'][:]
        lon = DATA.variables['longitude'][:]      #va de 0:360
        u = DATA.variables['uo'][0,:,:,:]    # [time, depth, lat, lon]
        depth = DATA.variables['depth'][:]
        # 1. Batimetria
        bat = np.nan*np.ones((len(lat),len(lon)))
        for i_lat in range(len(lat)):
            for i_lon in range(len(lon)):
                aux = u[:,i_lat,i_lon].mask
                for i_depth in range(len(depth)):
                    if aux[i_depth] == False:
                        bat[i_lat,i_lon] = depth[i_depth]

                        #ptos de grilla:
                        xi, yi = np.meshgrid(np.linspace(-60,-48,len(lon)),np.linspace(-40,-30,len(lat)))
                        points = np.array( (xi.flatten(), yi.flatten()) ).T
                        values = bat.flatten()
                        z_T = griddata(points, values, (lon_T,lat_T))

    if fuente == 'GEBCO':
        DATA = Dataset(path_gral+'Doctorado/Datos/batimetria/GEBCO_2019/gebco_2019_n-30.0_s-40.0_w-60.0_e-48.0.nc')
        lon = DATA.variables['lon'][:]
        lat = DATA.variables['lat'][:]
        #limites de la transecta:
        #longitude
        lon_T_sur   = np.min(lon_T)
        lon_T_norte = np.max(lon_T)

        dlon_sur = lon-lon_T_sur
        i_lon_gebco_sur  = np.where(abs(dlon_sur) == np.min(abs(dlon_sur)))[0][0]

        dlon_norte = lon-lon_T_norte
        i_lon_gebco_norte  = np.where(abs(dlon_norte) == np.min(abs(dlon_norte)))[0][0]

        lon_gebco_T = lon[i_lon_gebco_sur-10:i_lon_gebco_norte+10]      #Tomo un margen de +-10 para que los bordes de la linea de control no qeuden en el limite

        #latitude
        lat_T_sur   = np.min(lat_T)
        lat_T_norte = np.max(lat_T)

        dlat_sur = lat-lat_T_sur
        i_lat_gebco_sur  = np.where(abs(dlat_sur) == np.min(abs(dlat_sur)))[0][0]

        dlat_norte = lat-lat_T_norte
        i_lat_gebco_norte  = np.where(abs(dlat_norte) == np.min(abs(dlat_norte)))[0][0]

        lat_gebco_T = lat[i_lat_gebco_sur-10:i_lat_gebco_norte+10]      #Tomo un margen de +-10 para que los bordes de la linea de control no qeuden en el limite

        #interpolacion
        xi, yi = np.meshgrid(lon_gebco_T,lat_gebco_T)
        points = np.array( (yi.flatten(), xi.flatten()) ).T
        #solo tomo los datos de prof cercanos a la linea de control (+-10, para tener margen) para que no pese tanto la matriz
        depth = DATA.variables['elevation'][i_lat_gebco_sur-10:i_lat_gebco_norte+10,i_lon_gebco_sur-10:i_lon_gebco_norte+10]            #lat, lon
        values = depth.flatten()


        z_T = -griddata(points, values, (lat_T,lon_T))

    if fuente == 'ETOPO':
        DATA = Dataset(path_gral+'Doctorado/Datos/batimetria/ETOPO1_Bed_g_gmt4.grd','r')
        DATA.set_auto_mask(False)

        lon = DATA.variables['x'][:]
        lat = DATA.variables['y'][:]
        #limites de la transecta:
        #longitude
        lon_T_sur   = np.min(lon_T)
        lon_T_norte = np.max(lon_T)

        dlon_sur = lon-lon_T_sur
        i_lon_etopo_sur  = np.where(abs(dlon_sur) == np.min(abs(dlon_sur)))[0][0]

        dlon_norte = lon-lon_T_norte
        i_lon_etopo_norte  = np.where(abs(dlon_norte) == np.min(abs(dlon_norte)))[0][0]

        lon_etopo_T = lon[i_lon_etopo_sur-10:i_lon_etopo_norte+10]      #Tomo un margen de +-10 para que los bordes de la linea de control no qeuden en el limite

        #latitude
        lat_T_sur   = np.min(lat_T)
        lat_T_norte = np.max(lat_T)

        dlat_sur = lat-lat_T_sur
        i_lat_etopo_sur  = np.where(abs(dlat_sur) == np.min(abs(dlat_sur)))[0][0]

        dlat_norte = lat-lat_T_norte
        i_lat_etopo_norte  = np.where(abs(dlat_norte) == np.min(abs(dlat_norte)))[0][0]

        lat_etopo_T = lat[i_lat_etopo_sur-10:i_lat_etopo_norte+10]      #Tomo un margen de +-10 para que los bordes de la linea de control no qeuden en el limite

        #interpolacion
        xi, yi = np.meshgrid(lon_etopo_T,lat_etopo_T)
        points = np.array( (yi.flatten(), xi.flatten()) ).T
        #solo tomo los datos de prof cercanos a la linea de control (+-10, para tener margen) para que no pese tanto la matriz
        depth = DATA.variables['z'][i_lat_etopo_sur-10:i_lat_etopo_norte+10,i_lon_etopo_sur-10:i_lon_etopo_norte+10]            #lat, lon
        values = depth.flatten()


        z_T = -griddata(points, values, (lat_T,lon_T))

    if fuente == 'Mati':
        path = path_gral+'Doctorado/Datos/batimetria/dominio0_batimetria_MoirayDiego_tesis_y_Mati_interpolacion.nc'
        DATA = Dataset(path)
        lon = DATA['lon_rho'][0,:]
        lat = DATA['lat_rho'][:,0]

        #limites de la transecta:
        #longitude
        lon_T_sur   = np.min(lon_T)
        lon_T_norte = np.max(lon_T)

        dlon_sur = lon-lon_T_sur
        i_lon_mati_sur  = np.where(abs(dlon_sur) == np.min(abs(dlon_sur)))[0][0]

        dlon_norte = lon-lon_T_norte
        i_lon_mati_norte  = np.where(abs(dlon_norte) == np.min(abs(dlon_norte)))[0][0]

        lon_mati_T = lon[i_lon_mati_sur-10:i_lon_mati_norte+10]      #Tomo un margen de +-10 para que los bordes de la linea de control no qeuden en el limite

        #latitude
        lat_T_sur   = np.min(lat_T)
        lat_T_norte = np.max(lat_T)

        dlat_sur = lat-lat_T_sur
        i_lat_mati_sur  = np.where(abs(dlat_sur) == np.min(abs(dlat_sur)))[0][0]

        dlat_norte = lat-lat_T_norte
        i_lat_mati_norte  = np.where(abs(dlat_norte) == np.min(abs(dlat_norte)))[0][0]

        lat_mati_T = lat[i_lat_mati_sur-10:i_lat_mati_norte+10]      #Tomo un margen de +-10 para que los bordes de la linea de control no qeuden en el limite

        #interpolacion
        xi, yi = np.meshgrid(lon_mati_T,lat_mati_T)
        points = np.array( (yi.flatten(), xi.flatten()) ).T
        #solo tomo los datos de prof cercanos a la linea de control (+-10, para tener margen) para que no pese tanto la matriz
        depth = DATA.variables['h'][i_lat_mati_sur-10:i_lat_mati_norte+10,i_lon_mati_sur-10:i_lon_mati_norte+10]            #lat, lon
        values = depth.flatten()


        z_T = griddata(points, values, (lat_T,lon_T))

    if fuente == 'STSF':
        path = path_gral+'Doctorado/Datos/batimetria/Batimetria_STSF_Trucha.xlsx'
        DATA = pd.read_excel(path)
        lon_STSF = DATA['Long_trucha'].values
        lat_STSF = DATA['Latitud'].values

        #interpolacion
        points = (lat_STSF,lon_STSF)
        #solo tomo los datos de prof cercanos a la linea de control (+-10, para tener margen) para que no pese tanto la matriz
        depth = DATA['DepthNeg'].values
        values = depth.flatten()
        z_T = -griddata(points, values, (lat_T,lon_T))

    if fuente == 'Palma':
        path = path_gral+'Doctorado/Datos/batimetria/usada_en_Palma_2008_2004/htotmodif2.xlsx'
        DATA = pd.read_excel(path)
        lon_Palma = DATA['lon'].values
        lat_Palma = DATA['lat'].values
        #interpolacion
        points = (lat_Palma,lon_Palma)
        depth = DATA['z'].values
        values = depth.flatten()
        z_T = -griddata(points, values, (lat_T,lon_T))

    if fuente == 'Topex':
        path = path_gral+'Doctorado/Datos/batimetria/TOPEX_topo_stsf.nc'
        DATA = Dataset(path)
        lat_Topex = DATA.variables['lat'][:]
        lon_Topex = DATA.variables['lon'][:]      #va de 0:360
        depth = DATA.variables['z'][:,:]

        #interpolacion
        xi, yi = np.meshgrid(np.linspace(-60,-48,len(lon_Topex)),np.linspace(-40,-30,len(lat_Topex)))
        points = np.array( (yi.flatten(), xi.flatten()) ).T
        values = depth.flatten()
        z_T = -griddata(points, values, (lat_T,lon_T))
    return z_T

def buscar_limites_linea_control(a):
    """
    Busca los limites de la linea control usando los datos de CTD, GEBCO, NEMO, ETOPO o Mati.

    Parametros de entrada
    -------------------
    a: 'STSF' o  'GEBCO' o 'ETOPO' o 'Mati' o 'NEMO' o 'Palma' O 'Topex '. string. En base a que se buscan los limites de la linea control.

    Parametros de salida
    -------------------
    lon_sur: 1D numpy.array. Longitude del limite sur
    lat_sur: 1D numpy.array. Latitud del limite sur
    lon_norte: 1D numpy.array. Longitude del limite norte
    lat_norte: 1D numpy.array. Latitud del limite norte

    """
    #Limite sur:
    [A01,A08] = [fCNV(path_gral+'Doctorado/Datos/STSF_data/d01.cnv'),fCNV(path_gral+'Doctorado/Datos/STSF_data/d08.cnv')]
    lon_s = np.linspace(A01['LONGITUDE'][0], A08['LONGITUDE'][0],50)
    lat_s = np.linspace(A01['LATITUDE'] [0], A08['LATITUDE'] [0],50)


    #Limite norte:
    A046 = fCNV(path_gral+'Doctorado/Datos/STSF_data/d046.cnv')
    #Est55= mas cerca de la costa
    lat_n = np.linspace(-31.558333333, A046['LATITUDE'] [0],50)
    lon_n = np.linspace(-50.915216667, A046['LONGITUDE'][0],50)

    prof_s = batimetria_sobre_transecta(lon_s,lat_s,a)
    prof_n = batimetria_sobre_transecta(lon_n,lat_n,a)

    prof_s[np.isnan(prof_s) == True]= 0
    prof_n[np.isnan(prof_n) == True]= 0

    #limite sur
    i = len(abs(prof_s[abs(prof_s)<200])) -1 #indice de ultima est con prof < 200m

    m_lat = (lat_s[i+1]-lat_s[i])/(prof_s[i+1]-prof_s[i])
    b_lat = lat_s[i]-m_lat*prof_s[i]
    lat_sur = m_lat*200+b_lat

    m_lon = (lon_s[i+1]-lon_s[i])/(prof_s[i+1]-prof_s[i])
    b_lon = lon_s[i]-m_lon*prof_s[i]
    lon_sur  = m_lon*200+b_lon

    #limite norte
    i = len(abs(prof_n[abs(prof_n)<200])) -1 #ultima est con prof < 200m

    m_lat = (lat_n[i+1]-lat_n[i])/(prof_n[i+1]-prof_n[i])
    b_lat = lat_n[i]-m_lat*prof_n[i]
    lat_norte = m_lat*200+b_lat

    m_lon = (lon_n[i+1]-lon_n[i])/(prof_n[i+1]-prof_n[i])
    b_lon = lon_n[i]-m_lon*prof_n[i]
    lon_norte  = m_lon*200+b_lon


    return lon_sur,lat_sur, lon_norte, lat_norte

def interpolacion_por_nivel_considerando_nans(LONS,LATS,U_10,V_10,lon_T,lat_T):
    """
        Interpola U y V por nivel (c/10m) sobre la transecta considerando los nans.

        Parametros de entrada
        -------------------
        LONS : 1D numpy.array longitudes de las estaciones
        LATS : 1D numpy.array latitudes de las estaciones
        U_10 : 2D numpy.array (prof, estaciones): (20,55) velocidad zonal c/10m
        V_10 : 2D numpy.array (prof, estaciones): (20,55) velocidad meridional c/10m
        lon_T: 1D numpy.array longitudes de la transecta a interpolar
        lat_T: 1D numpy.array latitudes de la transecta a interpolar
        z_T  : 1D numpy.array Batimetria sobre la transecta a interpolar

        Parametros de salida
        -------------------
        u_T: 2D numpy.array (prof, pto de transecta) vel. zonal sobre la transecta
        v_T: 2D numpy.array (prof, pto de transecta) vel. meridional sobre la transecta

    """
    ndepth = U_10.shape[0]
    nptos = len(lon_T)
    u_g = np.empty((ndepth,nptos))
    v_g = np.empty((ndepth,nptos))
    for i in range(ndepth):
        u_g[i,:] = griddata((LONS,LATS), U_10[i,:], (lon_T,lat_T))
        v_g[i,:] = griddata((LONS,LATS), V_10[i,:], (lon_T,lat_T))

    return u_g,v_g

def extrapolacion_lineal_hacia_el_fondo(u_g,v_g,lon_T,lat_T,bat, metodo):
    """
    Como u_g, v_g sobre la transecta tienen muchos nans entre
    el ultimo dato interpolado y el fondo segun gebco, esta funcion extrapola
    linealmente la veloc.zonal y meridional hasta el fondo, deifnido por gebco
    discretizado c/10m

    Parámetros de entrada
    --------------------
    u_g   : 2D numpy.array (20,200):(prof,pto). Vel zonal interpolada con interpolacion por nivel considerando nans
    v_g   : 2D numpy.array (20,200):(prof,pto). Vel meridional interpolada con interpolacion por nivel considerando nans
    lon_T : 1D numpy.array. Longitudes de la Transecta
    lat_T : 1D numpy.array. Latitudes de la transecta
    bat: Batimetria sobre la transecta
    metodo : 'lineal', 'cte', 'ceros'.

    Parámetros de salida
    --------------------
    u_g_lineal: 2D numpy.array (20,200):(prof,pto). Vel zonal extrapolada linealmente hasta el fondo (definido por gebco) discretizado c/10m
    v_g_lineal: 2D numpy.array (20,200):(prof,pto). Vel meridional extrapolada linealmente hasta el fondo (definido por gebco) discretizado c/10m


    """
    nptos = len(lon_T)
    zz = np.linspace(0,4999,5000) #Profs
    #regrillado de batimetria en zz:
    bat = np.round(bat/10)*10-5
    bat[bat>200] = 200  #max prof 195mm

    if nptos != u_g.shape[1]:
        print('Transecta con diferentes dimensiones que el campo de velocidades')

    if metodo == 'lineal':
        u_g_lineal = np.copy(u_g)
        v_g_lineal = np.copy(v_g)
        for pto in range(nptos):
            u,v = u_g[:,pto],v_g[:,pto]
            i_int = len(u[np.isnan(u) == False])-1      #hasta este nivel tienen datos
            Z_int = zz[i_int]                           #ultima prof con datos interpolados
            U_int,V_int = u[i_int],v[i_int]                            #veloc. interpolada en la ultima prof

            Z_fondo = bat[pto]                                           # batimetria segun gebco
            i_z_fondo = np.where(zz == Z_fondo)[0][0]                    # indice asociado en zz
            U_fondo,V_fondo = 0,0                                                  # sup: vel nula en el fondo

            if Z_fondo > Z_int:
                #tengo que hacer la extrapolacion lineal hacia el fondo
                m_u, m_v = (U_int-U_fondo)/(Z_int-Z_fondo), (V_int-V_fondo)/(Z_int-Z_fondo)
                b_u, b_v = U_fondo-Z_fondo*m_u            ,  V_fondo-Z_fondo*m_v
                for i in range(i_int,i_z_fondo+1):
                    u_g_lineal[i,pto] = m_u*zz[i]+b_u
                    v_g_lineal[i,pto] = m_v*zz[i]+b_v

        u_g_extrapolada = u_g_lineal
        v_g_extrapolada = v_g_lineal

    if metodo == 'cte':
        u_g_cte = np.copy(u_g)
        v_g_cte = np.copy(v_g)
        for pto in range(nptos):
            u,v = u_g[:,pto],v_g[:,pto]
            i_int = len(u[np.isnan(u) == False])-1      #hasta este nivel tienen datos
            Z_int = zz[i_int]                           #ultima prof con datos interpolados
            U_int,V_int = u[i_int],v[i_int]                            #veloc. interpolada en la ultima prof

            Z_fondo = bat[pto]                                           # batimetria segun gebco
            i_z_fondo = np.where(zz == Z_fondo)[0][0]                    # indice asociado en zz
            U_fondo,V_fondo = 0,0                                                  # sup: vel nula en el fondo

            if Z_fondo > Z_int:
                #tengo que hacer la extrapolacion cte hacia el fondo
                for i in range(i_int,i_z_fondo+1):
                    u_g_cte[i,pto] = U_int
                    v_g_cte[i,pto] = V_int

        u_g_extrapolada = u_g_cte
        v_g_extrapolada = v_g_cte

    if metodo == 'ceros':
        u_g_ceros = np.copy(u_g)
        v_g_ceros = np.copy(v_g)
        for pto in range(nptos):
            u,v = u_g[:,pto],v_g[:,pto]
            i_int = len(u[np.isnan(u) == False])-1      #hasta este nivel tienen datos
            Z_int = zz[i_int]                           #ultima prof con datos interpolados
            U_int,V_int = u[i_int],v[i_int]                            #veloc. interpolada en la ultima prof

            Z_fondo = bat[pto]                                           # batimetria segun gebco
            i_z_fondo = np.where(zz == Z_fondo)[0][0]                    # indice asociado en zz
            U_fondo,V_fondo = 0,0                                                  # sup: vel nula en el fondo

            if Z_fondo > Z_int:
                #tengo que hacer la extrapolacion cte hacia el fondo
                for i in range(i_int,i_z_fondo+1):
                    u_g_ceros[i,pto] = 0
                    v_g_ceros[i,pto] = 0

        u_g_extrapolada = u_g_ceros
        v_g_extrapolada = v_g_ceros


    return u_g_extrapolada, v_g_extrapolada

def proyeccion_sobre_recta(lon_r,lat_r,u,v):
    """
    Proyecta las velocidades u,v sobre la normal de la recta definida
    por lon_r,lat_r. Considera positivo hacia el este y el norte

    Parametros de entrada
    --------------------
    lon_r: 1D numpy.array de longitudes de la recta sobre la cual se quiere proyectar
    lat_r: 1D numpy.array de latitudes de la recta sobre la cual se quiere proyectar
    u    : 2D numpy.array (prof,pto) de velocidades zonales
    v    : 2D numpy.array (prof,pto) de velocidades meridionales

    Parametros de Salidas
    --------------------
    Vf: 2D numpy.array (prof,pto) de velocidades proyectadas sobre la recta

    """
    x1,y1   = lon_r[0] ,lat_r[0]
    x2,y2   = lon_r[len(lon_r)-1],lat_r[len(lat_r)-1]
    alpha_r = math.atan2((x2-x1),(y2-y1))   # ° de la isobata respecto de norte (42° aprox)
    alpha_p = alpha_r + math.pi/2

    beta = np.arctan2(u,v)          #angulos de cada nodo respecto al norte
    gamma = beta - alpha_p              #angulo entre el vector de velocidad y el vector perpendicular a la isobata
    Vf = np.sqrt(u**2+v**2)*np.cos(gamma)   #intensidad de la vel perpendicular

    return Vf

def f_proyeccion_paralela_sobre_recta(lon_r,lat_r,u,v):
    """
    Proyecta las velocidades u,v sobre la paralela de la recta definida
    por lon_r,lat_r. Considera positivo hacia el este y el norte

    Parametros de entrada
    --------------------
    lon_r: 1D numpy.array de longitudes de la recta sobre la cual se quiere proyectar
    lat_r: 1D numpy.array de latitudes de la recta sobre la cual se quiere proyectar
    u    : 2D numpy.array (prof,pto) de velocidades zonales
    v    : 2D numpy.array (prof,pto) de velocidades meridionales

    Parametros de Salidas
    --------------------
    V_paralela: 2D numpy.array (prof,pto) de velocidades proyectadas sobre la recta

    """
    x1,y1   = lon_r[0] ,lat_r[0]
    x2,y2   = lon_r[len(lon_r)-1],lat_r[len(lat_r)-1]
    alpha_r = math.atan2((x2-x1),(y2-y1))   # ° de la isobata respecto de norte (42° aprox)
    alpha_p = alpha_r + math.pi/2

    beta = np.arctan2(u,v)          #angulos de cada nodo respecto al norte
    gamma = beta - alpha_p              #angulo entre el vector de velocidad y el vector perpendicular a la isobata
    V_paralela = np.sqrt(u**2+v**2)*np.sin(gamma)   #intensidad de la vel paralela

    return V_paralela

def gen_dist_2d(lon_r,lat_r,depth):
    """
    Genera la matriz 2D de distancias entre los puntos de la linea control repetidas en la profundidad

    Parametros de entrada
    --------------------
    lon_r: 1D numpy.array (ptos) de longitudes de la recta
    lat_r: 1D numpy.array (ptos) de latitudes de la recta
    depth: 1D numpy.array (prof) de profundidades

    Parametros de salida
    -------------------
    dist2: 2D numpy.array (prof,ptos) de la distancia entre puntos repetidas en todas las profs.
    """

    dist = []
    for k in range(len(lon_r)-1):
        p1 = ( lat_r[k],lon_r[k])
        p2 = (lat_r[k+1],lon_r[k+1])
        dist.append(1000*haversine(p1,p2))         # (m)
    dist.append(dist[-1]/2)
    dist[0] = dist[0]/2
    dist2 = np.matlib.repmat(dist,len(depth),1)

    return dist2

def gen_dz_2d(lon_r,lat_r,depth):
    """
    Genera la matriz 2D de distancias entre los puntos de la linea control repetidas en la profundidad

    Parametros de entrada
    --------------------
    lon_r: 1D numpy.array (ptos) de longitudes de la recta
    lat_r: 1D numpy.array (ptos) de latitudes de la recta
    depth: 1D numpy.array (prof) de profundidades

    Parametros de salida
    -------------------
    dz2: 2D numpy.array (prof,ptos) de la profundidad de cada nivel repetidas en todos los puntos.
    """

    dz = np.nan*np.ones((len(depth)))
    dif_prof=[depth[0]]
    for z in range(0,len(depth)-1):
        dif_prof.append(depth[z+1]-depth[z])
    for z in range(len(depth)-1):
        if z == 0:
            dz[z] = dif_prof[z]+dif_prof[z+1]/2
        else:
            dz[z] = dif_prof[z]/2 + dif_prof[z+1]/2
    dz2 = np.matlib.repmat(dz,len(lon_r),1).T

    return dz2

def plot_recta_fuente(lon200,lat200,fuente, T_total,V_integrada, est_CTD = True,isob_200=False):
    """
    Plotea el mapa entre [30-40]S y [48-60]N, con:

    Parametros de entrada:
    --------------------
    lon200  : 1D numpy.array (ptos). Longitud de la linea de control (isobata de 200 simplificada por una recta)
    lat200  : 1D numpy.array (ptos). Latitud de la linea de control (isobata de 200 simplificada por una recta)
    fuente  : string. Fuente con la que se calculo el transporte
    T_total : 1D numpy.array. Transporte total hacia oceano abierto  (m**3/s)
    V_integrada: 1D numpy.array (ptos). Velocidad proyectada sobre la recta hacia afuera integrada en la vertical
    est_CTD : True or False
    isob_200: True or False
    """
    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    import cartopy.feature as cfeature
    import matplotlib.ticker as mticker
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


    xi, yi = np.meshgrid(np.linspace(-60,-48,500),np.linspace(-40,-30,500))
    cmap_vel =  plt.cm.get_cmap('seismic', 12)
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


    txt = 'T ('+str(fuente)+'): ' + str(T_total/1000000)[:4]+ ' Sv'
    c_vel = ax.scatter(lon200,lat200,c=V_integrada, cmap=cmap_vel,vmin=-30,vmax=30, label = txt)
    plt.legend(fontsize = 24)

    cax_vel = fig.add_axes([0.78, 0.1,0.02,0.8])
    levels_vel = [-30,-20,-10,0,10,20,30]
    cbar_vel = plt.colorbar(c_vel, orientation='vertical', ticks = levels_vel, cax=cax_vel)
    cbar_vel.ax.tick_params(labelsize=24)
    cbar_vel.ax.set_xlabel('$ cm.s^{-1}$', size=26,rotation=0)
    cbar_vel.ax.set_ylabel('$V_{offshore}$', size=26)

    if est_CTD == True:
        lat_CTD,lon_CTD = fCTD.lat_lon_CTD()
        ax.scatter(lon_CTD,lat_CTD,color='k',edgecolors ='white')
        if isob_200 == True:
            lon200_isob,lat200_isob = fCTD.isobata_200()
            ax.plot(lon200_isob,lat200_isob,color='grey')

def plot_recta_fuentes(fuentes,Transporte = 'neto', est_CTD = True,isob_200=False, vel_ADCP = False):
    """
    Plotea el mapa entre [30-40]S y [48-60]N, con las lineas de control para cada fuente:

    Parametros de entrada:
    --------------------
    fuentes  : string. Fuentes con la que se calculo el transporte
    Transporte: 'neto','offshore','inshore'
    est_CTD : True or False. Estaciones de CTD, STSF Survey
    isob_200: True or False. Isobata de 200m (GEBCO)
    vel_ADCP: True or False. Campo de velocidad superficial basado en datos ADCP.
    """
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

    #Transporte sobre linea de control:
    T_integrada = np.zeros((len(fuentes),200))
    for count, fuente in enumerate(fuentes):
        Bat_linea_control = pd.read_csv(path_gral+'Doctorado/Scripts/Salidas_npy/Bat_linea_control_fuente_'+str(fuente)+'.csv')
        lat200 = Bat_linea_control['lat200']
        lon200 = Bat_linea_control['lon200']
        T = np.load(path_gral+'Doctorado/Scripts/Salidas_npy/Transporte_'+str(fuente)+'.npy')
        T_neto,T_offshore,T_inshore = np.nansum(T)/1000000,np.nansum(T[T>0])/1000000,np.nansum(T[T<0])/1000000
        if Transporte == 'neto': T_tot = T_neto
        if Transporte == 'offshore': T_tot = T_offshore
        if Transporte == 'inshore': T_tot = T_inshore
        T_integrada[count,:] = np.nanmean(T,axis = 0)
        txt = 'T ('+str(fuente)+'): ' + str(T_tot)[:4]+ ' Sv'
        c_vel = ax.scatter(lon200,lat200,c=T_integrada[count,:],cmap = 'seismic',vmin=-700,vmax=700,s=5,label = txt)
    T_integrada_media = np.nanmean(T_integrada,axis =0)
    plt.legend(fontsize = 24,markerscale = 5)

    lat_CTD,lon_CTD = fCTD.lat_lon_CTD()
    if est_CTD == True:
        ax.scatter(lon_CTD,lat_CTD,color='k',edgecolors ='white')
    if isob_200 == True:
        lon200_isob,lat200_isob = fCTD.isobata_200()
        ax.plot(lon200_isob,lat200_isob,color='grey')
    if vel_ADCP == True:
        U_1, V_1 = fADCP.datos_ADCP_cada_1m()
        U_integrada,V_integrada = np.nanmean(U_1,axis = 0),np.nanmean(V_1,axis = 0)
        u_grid = griddata((lon_CTD,lat_CTD), U_integrada, (xi,yi))
        v_grid = griddata((lon_CTD,lat_CTD), V_integrada, (xi,yi))
        Q = ax.quiver(xi[::10,::10],yi[::10,::10],u_grid[::10,::10],v_grid[::10,::10],alpha = 0.5)
        QK = ax.quiverkey(Q,0.92,0.1,50,label = '$50cm.s^{-1}$',fontproperties={'size':24})

    plt.show()



""" Funciones p/calculo de veloc gesotrofica"""
# Baroclinico
def f_abro_CTD_st(st):
    """
    Abre el .CNV del CTD de la estacion indicada
    Parametro de entrada:
    --------------------
    st: 1D np.array con el numero de estacion a abrir

    Parametro de salida:
    --------------------
    DATA: .cnv abierto
    """

    path = path_gral+'Doctorado/Datos/STSF_data/'
    if st == 12:
        DATA = pd.read_excel(str(path)+'d0'+str(st)+'.xlsx')
    elif st in [52,53,54,55]:
        DATA = pd.read_excel(str(path)+'/EO_'+str(st)+'Down_cal_f.xlsx')
    else:
        DATA = fCNV(str(path) + 'd0'+str(st)+'.cnv')
    return DATA

def f_vel_geo_baroclinica_entre_st(estaciones):
    """
    Calcula la veloc geostrofica referido a la superficie entre las estaciones

    Parametros de entrada:
    ---------------------
    st: 1D np.array con las estaciones elegidas (solo 2) [5,22]

    Parametros de salida:
    ---------------------
    vg_rel: Velocidad geostrofica entre las estaciones.

    """
    #Matriz de Z,S y T cada 1metro.
    z = np.linspace(0,4999,5000)
    T = np.nan*np.ones((5000,2))
    S = np.nan*np.ones((5000,2))
    lat = []
    lon = []
    for st in [0,1]:
        DATA = f_abro_CTD_st(estaciones[st])
        lon.append(np.nanmean(DATA['LONGITUDE']))
        lat.append(np.nanmean(DATA['LATITUDE']))
        points   = DATA['PRES']
        values_t = DATA['TEMP']
        values_s = DATA['PSAL']

        T[:,st-1]   = griddata(points,values_t,z,method='linear')
        S[:,st-1]   = griddata(points,values_s,z,method='linear')
        aux = int(points[0]+1)
        T[:aux,st-1]    = values_t[0]
        S[:aux,st-1]    = values_s[0]
    # Calculo de geopotential anomaly [m 3 kg -1 Pa = m 2 s -2 = J kg -1] (referido a la sup)
    ga_A = sw.gpan(S[:,0],T[:,0],z)
    ga_B = sw.gpan(S[:,1],T[:,1],z)
    ga = np.array([ga_A,ga_B]).T
    vg_rel = sw.gvel(ga, lat, lon)

    return vg_rel

# Barotropico
def batimetria_recta_entre_st(sts,fuente):
    """
    Busca la batimetra sobre la recta entre estaciones de CTD de la campaña STSF2013

    Parametros de entrada
    --------------------
    sts: 1D np.array con el numero de estaciones entre las cuales quiero calcular
    fuente: 'NEMO' o 'GEBCO' o 'ETOPO' o 'Palma' o 'Topex', string. Fuente de la base de batimetria.

    Parametro de salida:
    -------------------
    lat_T:1D numpy.array. latitud de la recta entre estaciones
    lon_T:1D numpy.array. longitud de la recta entre estaciones
    bat: 1D numpy.array. Batimetria sobre T basado en 'fuente'.
    """
    lat_CTD,lon_CTD = fCTD.lat_lon_CTD()
    lat1,lon1 = lat_CTD[sts[0]-1],lon_CTD[sts[0]-1]
    lat2,lon2 = lat_CTD[sts[1]-1],lon_CTD[sts[1]-1]
    lat_T =  np.linspace(lat1,lat2,50)
    lon_T =  np.linspace(lon1,lon2,50)
    bat = batimetria_sobre_transecta(lon_T,lat_T,fuente)
    return lat_T,lon_T,bat

def f_vel_barotropica_sup_entre_st(sts,fecha_inicial,fecha_final):
    """
    Obtiene la vel barotropica superficial media entre las sts obtenida a partir de
    datos satelitales de altimetria de Copernicus Marine
    (http://marine.copernicus.eu/services-portfolio/access-to-products/?option=com_csw&view=details&product_id=SEALEVEL_GLO_PHY_L4_REP_OBSERVATIONS_008_047 )

    Parametro de entrada
    -------------------
    sts: 1D np.array con el numero de estaciones entre las cuales quiero calcular
    la velocidad

    Parametro de salida
    -------------------
    v_p: (int) velocidad normal barotropica promedio entre las estaciones  (m/s)
    sla_0: (int) sts[0]
    sla_1: (int) sts[1]

    sla: The sea level anomaly is the sea surface height above mean sea surface; it is referenced to the [1993, 2012] period
    DUDA: se usa adt q es la altura sobre el nuvel del geoide
    """

    lat_CTD,lon_CTD = fCTD.lat_lon_CTD()
    lat1,lon1 = lat_CTD[sts[0]-1],lon_CTD[sts[0]-1]
    lat2,lon2 = lat_CTD[sts[1]-1],lon_CTD[sts[1]-1]
    # Recta entre estaciones
    recta_lat =  np.linspace(lat1,lat2,50)
    recta_lon =  np.linspace(lon1,lon2,50)
    # Archivo de Copernicus con datos c/3hs para 01-2013 a 10-2013
    path = path_gral+'Doctorado/Datos/Copernicus/UV_geostrofica/DATASET-DUACS-REP-GLOBAL-MERGED-ALLSAT-PHY-L4_octubre_2013.nc'
    # U,V media entre tiempo_inicial y tiempo_final
    DATA = xr.open_dataset(path).sel(time = slice(fecha_inicial,fecha_final)).mean(dim = 'time')
    # Interpolacion a la recta:
    xi,yi = np.meshgrid(DATA.longitude.values-360,DATA.latitude.values)
    u,v,sla = DATA.ugos.values,DATA.vgos.values,DATA.adt.values

    #interpolo sobre la recta entre estaciones
    u_g_recta = griddata((xi.flatten(),yi.flatten()),u.flatten(), (recta_lon,recta_lat))
    v_g_recta = griddata((xi.flatten(),yi.flatten()),v.flatten(), (recta_lon,recta_lat))
    sla_0 = griddata((xi.flatten(),yi.flatten()),sla.flatten(), (recta_lon[0],recta_lat[0]))
    sla_1 = griddata((xi.flatten(),yi.flatten()),sla.flatten(), (recta_lon[-1],recta_lat[-1]))
    #parA INTERṔOLAR OCN XARRAY
    # u_g_recta = DATA.ugos.interp(longitude = reca_lon + 360,latitude = recta_lat)
    #
    #

    # media de la recta entre estacioens
    u_g,v_g = np.nanmean(u_g_recta),np.nanmean(v_g_recta)
    # proyeccion perpendicular a la recta
    v_p = proyeccion_sobre_recta(recta_lon,recta_lat,u_g,v_g)
    return v_p,sla_0,sla_1

def f_transporte_entre_st(sts,v):
    """
    Calculo de tranporte  la veloc media entre las estaciones

    Parametro de entrada:
    -------------------
    sts: 1D np.array (est1,est2) con el numero de estaciones entre las cuales quiero calcular
    v: (int) (m/s) velocidad  media entre las estaciones

    Parametro de salida:
    -------------------
    T: (int)(Sv=m**3/s) Transporte
    """

    lat_CTD,lon_CTD = fCTD.lat_lon_CTD()
    lat1,lon1 = lat_CTD[sts[0]-1],lon_CTD[sts[0]-1]
    lat2,lon2 = lat_CTD[sts[1]-1],lon_CTD[sts[1]-1]
    # Distancia entre estaciones (m)
    p1 = (lat1,lon1)
    p2 = (lat2,lon2)
    dist = 1000*haversine(p1,p2)
    # Profundidad (m)
    prof1 = int(f_abro_CTD_st(sts[0])['PRES'][-1])
    prof2 = int(f_abro_CTD_st(sts[1])['PRES'][-1])
    z = np.nanmin([prof1,prof2])

    T = dist*z*v
    return T/1000000

"""
Transporte de Ekman (Me)
"""
def f_W_paralela_a_recta(fecha_inicial,fecha_final ,recta_lon,recta_lat):
    """
    Velocidad del viento paralela a la recta

    Parametro de entrada
    -------------------
    fecha_inicial:   2013/10/03
    fecha_final:     2013/10/06
    recta_lon:  1D np.array longitudes de la recta
    recta_lat:  1D np.array latitudes de la recta

    Parametro de salida:
    -------------------
    W: viento paralelo a la recta. positivo hacia el sudoeste (m/s)
    u_mean: viento zonal medio (m/s)
    v_mean: viento meridional medio (m/s)
    """

    path_u = path_gral+'Doctorado/Datos/satelital/ERA5_2013/VIENTO_10/U10/326827.VAR_10U.e5.oper.an.sfc.128_165_10u.regn320sc.2013100100_2013103123.nc'
    path_v = path_gral+'Doctorado/Datos/satelital/ERA5_2013/VIENTO_10/V10/326827.VAR_10V.e5.oper.an.sfc.128_166_10v.regn320sc.2013100100_2013103123.nc'
    # U,V media entre tiempo_inicial y tiempo_final
    DATA_u = xr.open_dataset(path_u).sel(time = slice(fecha_inicial,fecha_final)).mean(dim = 'time')
    DATA_v = xr.open_dataset(path_v).sel(time = slice(fecha_inicial,fecha_final)).mean(dim = 'time')
    # Interpolacion a la recta:
    xi,yi = np.meshgrid(DATA_u.longitude.values-360,DATA_u.latitude.values)
    values_u,values_v = DATA_u.VAR_10U.values,DATA_v.VAR_10V.values
    u_R = griddata((yi.flatten(),xi.flatten()),values_u.flatten(),(recta_lat,recta_lon))
    v_R = griddata((yi.flatten(),xi.flatten()),values_v.flatten(),(recta_lat,recta_lon))
    # Promedio en toda la recta:
    u_mean = np.nanmean(u_R)
    v_mean = np.nanmean(v_R)
    # Proyecto p/calcular la velocidad paralela a la recta:
    W = f_proyeccion_paralela_sobre_recta(recta_lon,recta_lat,u_mean,v_mean)
    return W, u_mean,v_mean

def densidad_estacion(st):
    """
    Calculo de densidad media de la estacion st de la campaña STSF2013
    Parametro de entrada
    --------------------
    st: (int) numero de estacion de STSF2013

    Paramtro de salida
    ------------------
    rho_agua: 1D np.array densidad de la estacion (kg/m**3))
    """
    DATA = f_abro_CTD_st(st)
    t = DATA['TEMP']
    s = DATA['PSAL']
    rho_agua = sw.dens(s,t,0)

    return rho_agua

from airsea import windstress as ws
def f_Me_vol(W,recta_lat,recta_lon,sts,rho_air = 1.225,Cd = 10**-3):
    """
    Calculo del transporte de volumen de Ekman a 90° a la derecha
    de la direccion del viento por metro

    Unidades:
        [rho_air] = kg/m3
        [omega] = 1/s
        [f] = 1/s
        [W] = m/s
        [Cd] = adimensional
        [Tau] = N/m**2
        [Me_x_metro] = Kg/(m.s)
        [Me_masa] = Kg/s
        [Me_vol] = m**3/s

    Parametro de entrada:
    --------------------
    W       : (int). Velocidad del viento paralela a la costa en la superficie (m/s)
    lat_recta: (int). Latitud de la recta sobre la cual se calcula Me
    lon_recta: (int). Longitud de la recta sobre la cual se calcula Me
    rho_air : (int). Densidad del viento (kg/m**3)
    Cd      : (int). Coeficiente de arrastre
    sts: 1D np.array numeros de estaciones entre las cuales se calcula Me

    Parametro de salida:
    --------------------
    Me_vol : (int) Transporte de volumen por metro [Sv]=(m**3/s)/1000000

    """
    lat_media = np.nanmean(recta_lat)
    #Parametro de Coriolis
    omega = 2*np.pi/86400
    f = 2*omega*np.sin(lat_media)
    ## Stress del viento usando el metodo de Smith, tmb se puede poner 'largepond' para cambiar de metodo por large and pond 1981
    Tau = ws.stress(W,10,'smith')
    # Transporte x metro a la derecha de la direccion de Tau
    Me_x_metro = Tau*(1/f)
    #Transporte total de masa
    p1 = (recta_lat[0],recta_lon[0])
    p2 = (recta_lat[1],recta_lon[1])
    dist = 1000*haversine(p1,p2)  #[m]
    if W < 0 : a =-1
    if W > 0 : a = 1

    Me_masa = a*np.abs(Me_x_metro)*dist          # [kg/s]
    #Transporte total de vol considernado la densidad media entre las estaciones.
    rho_sts = np.nanmean(densidad_estacion(sts[0])),np.nanmean(densidad_estacion(sts[1]))
    Me_vol = Me_masa/np.mean(rho_sts)        # [m**3/s]
    return Me_vol/1000000      #[Sv]




"""
Figura paper
"""
def plot_transporte_paper(fuentes, est_CTD = True,isob_200=False, vel_ADCP = False,vel_OSCAR = False):
    """
    Plotea el mapa entre [30-40]S y [48-60]N, con las lineas de control para cada fuente:

    Parametros de entrada:
    --------------------
    fuentes  : string. Fuentes con la que se calculo el transporte
    Transporte: 'neto','offshore','inshore'
    est_CTD : True or False. Estaciones de CTD, STSF Survey
    isob_200: True or False. Isobata de 200m (GEBCO)
    vel_ADCP: True or False. Campo de velocidad superficial basado en datos ADCP.
    vel_OSCAR: True or False. Campo de velocidad superficial basado en oscar.
    """
    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    import cartopy.feature as cfeature
    import matplotlib.ticker as mticker
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


    xi, yi = np.meshgrid(np.linspace(-60,-48,500),np.linspace(-40,-30,500))
    fig = plt.figure()
    ax = fig.add_axes([0.05,0.05,0.8,0.9],projection=ccrs.Mercator())
    ax.set_extent([-60,-48,-40,-30],crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, color='#BDA973')
    ax.add_feature(cfeature.LAKES, color='lightcyan')
    ax.add_feature(cfeature.RIVERS, edgecolor='black')
    ax.coastlines(resolution='50m', color='black', linewidth=1)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='black', alpha=0.5, linestyle='-')
    gl.xlabels_top = False; gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator([-60,-58,-56,-54,-52,-50,-48])
    gl.ylocator = mticker.FixedLocator([-40,-38,-36,-34,-32,-30])
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 24, 'color': 'k'}
    gl.ylabel_style = {'size': 24, 'color': 'k'}
    #Transporte sobre linea de control:
    # T_integrada = np.zeros((len(fuentes),200))
    # for count, fuente in enumerate(fuentes):
    #     Bat_linea_control = pd.read_csv(path_gral+'Doctorado/Scripts/Salidas_npy/Bat_linea_control_fuente_'+str(fuente)+'.csv')
    #     lat200 = Bat_linea_control['lat200']
    #     lon200 = Bat_linea_control['lon200']
    #     T = np.load(path_gral+'Doctorado/Scripts/Salidas_npy/Transporte_'+str(fuente)+'.npy')
    #     T_neto,T_offshore,T_inshore = np.nansum(T)/1000000,np.nansum(T[T>0])/1000000,np.nansum(T[T<0])/1000000
    #     T_integrada[count,:] = np.nanmean(T,axis = 0)
    #     c_vel = ax.scatter(lon200,lat200,c=T_integrada[count,:],cmap = 'seismic',vmin=-700,vmax=700,s=5)
    # T_integrada_media = np.nanmean(T_integrada,axis =0)
    lat_CTD,lon_CTD = fCTD.lat_lon_CTD()



    # Segmento transporte geostrofico + Ekman
    # A: 5 y 22
    ax.plot([lon_CTD[4],lon_CTD[21]],[lat_CTD[4],lat_CTD[21]],c='darkgreen',lw=3,ls='-',marker = '*',markersize = 20,zorder = 5,transform=ccrs.PlateCarree())
    # B: 22 y 32
    ax.plot([lon_CTD[21],lon_CTD[31]],[lat_CTD[21],lat_CTD[31]],c='darkgreen',lw=3,ls='-',marker = '*',markersize = 20,zorder = 5,transform=ccrs.PlateCarree())
    # C: 32 y 36
    ax.plot([lon_CTD[31],lon_CTD[35]],[lat_CTD[31],lat_CTD[35]],c='darkgreen',lw=3,ls='-',marker = '*',markersize = 20,zorder = 5,transform=ccrs.PlateCarree())
    # D: 36 y 45
    ax.plot([lon_CTD[35],lon_CTD[44]],[lat_CTD[35],lat_CTD[44]],c='darkgreen',lw=3,ls='-',marker = '*',markersize = 20,zorder = 5,transform=ccrs.PlateCarree())
    # E: 45 y 49
    ax.plot([lon_CTD[44],lon_CTD[48]],[lat_CTD[44],lat_CTD[48]],c='darkgreen',lw=3,ls='-',marker = '*',markersize = 20,zorder = 5,transform=ccrs.PlateCarree())
    #Flechas verdes:
    xf = np.array([np.mean([lon_CTD[4],lon_CTD[21]]), np.mean([lon_CTD[21],lon_CTD[31]]),np.mean([lon_CTD[31],lon_CTD[35]]),np.mean([lon_CTD[35],lon_CTD[44]]),np.mean([lon_CTD[44],lon_CTD[48]])])
    yf = np.array([np.mean([lat_CTD[4],lat_CTD[21]]), np.mean([lat_CTD[21],lat_CTD[31]]),np.mean([lat_CTD[31],lat_CTD[35]]),np.mean([lat_CTD[35],lat_CTD[44]]),np.mean([lat_CTD[44],lat_CTD[48]])])
    uf = np.array([10,10,-5,5,2])
    vf = np.array([-8,-4,4,-2,-1])
    ax.quiver(xf,yf,uf,vf,color = 'darkgreen',scale =150,zorder = 5,transform=ccrs.PlateCarree())

    if est_CTD == True:
        ax.scatter(lon_CTD,lat_CTD,color='k',edgecolors ='white',transform=ccrs.PlateCarree())
    if isob_200 == True:
        lon200_isob,lat200_isob = fCTD.isobata_200()
        ax.plot(lon200_isob,lat200_isob,color='grey',transform=ccrs.PlateCarree())
    if vel_ADCP == True:
        U_1, V_1 = fADCP.datos_ADCP_cada_1m()
        U_integrada,V_integrada = np.nanmean(U_1,axis = 0),np.nanmean(V_1,axis = 0)
        u_grid = griddata((lon_CTD,lat_CTD), U_integrada, (xi,yi))
        v_grid = griddata((lon_CTD,lat_CTD), V_integrada, (xi,yi))
        Q = ax.quiver(xi[::10,::10],yi[::10,::10],u_grid[::10,::10],v_grid[::10,::10],alpha = 0.5,transform=ccrs.PlateCarree())
        QK = ax.quiverkey(Q,0.92,0.1,50,label = '$50cm.s^{-1}$',fontproperties={'size':24})

    if vel_ADCP == False and vel_OSCAR == True:
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
        cs = ax.quiver(x_vel,y_vel,(u[t0-1,:,:]+u[t0,:,:])/2,(v[t0-1,:,:]+v[t0,:,:])/2,color='0.2',units='width',scale=12,transform = ccrs.PlateCarree(),zorder = 3,alpha = 0.6)
        ax.quiverkey(cs, 0.72, 0.9, 0.5, r'$ 50\frac{cm}{s}$', labelpos='E',coordinates='figure',fontproperties={'size' :24})
        # ax.text(15000,1000000,'Currents (OSCAR)\nMean between 3 and 10 October',size=12,bbox=dict(facecolor='white'))

    txt = 'Section   Tg+Me [Sv]  \nSt.  5-22:  1.85 \nSt.22-32:  1.56 \nSt.32-36: -0.39 \nSt.36-45:  0.28 \nSt.45-48:  0.03 \n Total:      3.33'
    ax.text(-59.5,-33.75,txt,size=26,bbox =  dict(boxstyle='round', facecolor='white', alpha=0.8),transform=ccrs.PlateCarree())
    plt.show()




















###
###
