"""
Funciones para explorar los datos de ADCP de la Campaña STSF2013
"""
# path_gral = '/media/giuliana/DE_BERDEN/'
# path_gral = 'home/california/Documents/'
path_gral = '/media/giuliana/Disco1TB/'
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import math
import f_CTD_exploratorio as fCTD
from haversine import haversine


def datos_ADCP_cada_10m():
    """
    1. Abre datos de ADCP ubicados en path:/media/california/TOSHIBA EXT/Datos/STSF_data/Datos-ADCP
    2. Los regrilla verticalmente cada dz metros

    Paramatros de entrada
    --------------------
    ADCP: no significa nada. NO lo usa

    Parametros de salida
    --------------------
    U_10: 2D numpy.array. Velocidad zonal c/10m. (prof, estacion) = (20,55)
    V_10: 2D numpy.array. Velocidad meridional c/10m. (prof, estacion) = (20,55)
    """

    DATA = []
    z = np.linspace(5,195,20) #Profs
    U_10 = np.nan*np.ones((20,55))
    V_10 = np.nan*np.ones((20,55))
    for l in range(0,55):
        est = l+1
        DATA.append(pd.read_excel(path_gral+'Doctorado/Datos/STSF_data/Datos-ADCP/L-ADCP-'+str(est)+'.xlsx'))
        points   = DATA[l]['Pressure (dbar)']
        values_u = DATA[l]['U (cm/s)']
        values_v = DATA[l]['V (cm/s)']
        U_10[:,l] = griddata(points,values_u,z,method='linear')
        V_10[:,l] = griddata(points,values_v,z,method='linear')
        U_10[0,l] = values_u[0]  #Correccion en el primer dato
        V_10[0,l] = values_v[0]  #Correccion en el primer dato
    for est in [3,22,39]:
        l=est-1
        U_10[:,l] = (U_10[:,l-1]+U_10[:,l+1])/2
        V_10[:,l] = (V_10[:,l-1]+V_10[:,l+1])/2
    for est in [28,41,52,53,54,55]:
        l=est-1
        U_10[:,l] =  U_10[:,l-1]
        V_10[:,l] =  V_10[:,l-1]
            #Datos faltantes
    U_10[:,28] =  U_10[:,29]
    V_10[:,28] =  V_10[:,29]
    U_10[1,2]  = (U_10[0,2]+U_10[2,2])/2
    V_10[7,21] = (V_10[6,21]+V_10[8,21])/2

    return U_10, V_10

def datos_ADCP_cada_1m():
    """
    1. Abre datos de ADCP ubicados en path:/media/california/TOSHIBA EXT/Datos/STSF_data/Datos-ADCP
    2. Los regrilla verticalmente cada 1 metro

    Paramatros de entrada
    --------------------

    Parametros de salida
    --------------------
    U_10: 2D numpy.array. Velocidad zonal c/1m [cm/s]. (prof, estacion) = (5000,55)
    V_10: 2D numpy.array. Velocidad meridional c/1m [cm/s]. (prof, estacion) = (5000,55)
    """

    DATA = []
    z = np.linspace(0,4999,5000) #Profs
    U_1 = np.nan*np.ones((5000,55))
    V_1 = np.nan*np.ones((5000,55))
    for l in range(0,55):
        est = l+1
        DATA.append(pd.read_excel(path_gral+'Doctorado/Datos/STSF_data/Datos-ADCP/L-ADCP-'+str(est)+'.xlsx'))
        points   = DATA[l]['Pressure (dbar)']
        values_u = DATA[l]['U (cm/s)']
        values_v = DATA[l]['V (cm/s)']
        U_1[:,l] = griddata(points,values_u,z,method='linear')
        V_1[:,l] = griddata(points,values_v,z,method='linear')
    for est in [3,22,39]:
        l=est-1
        U_1[:,l] = (U_1[:,l-1]+U_1[:,l+1])/2
        V_1[:,l] = (V_1[:,l-1]+V_1[:,l+1])/2
    for est in [28,41]:
        l=est-1
        U_1[:,l] =  U_1[:,l-1]
        V_1[:,l] =  V_1[:,l-1]
        #arreglo de la est 51, falta la vl meridional:
    # U_1[39:45,50] = 9
            #Datos faltantes
    U_1[:,28] =  U_1[:,29]
    V_1[:,28] =  V_1[:,29]
    #Extrapolo hasta superficie el primer dato.
    for l in range(0,51):
        aux1_u,aux1_v = U_1[:,l],V_1[:,l]
        aux2_u,aux2_v = aux1_u[np.isnan(aux1_u) == False][0],aux1_v[np.isnan(aux1_v) == False][0]
        aux_u,aux_v = np.where(aux1_u == aux2_u)[0][0],np.where(aux1_v == aux2_v)[0][0]
        U_1[:aux_u,l]    = aux2_u
        V_1[:aux_v,l]    = aux2_v
    # FALTA
    # U_10[1,2]  = (U_10[0,2]+U_10[2,2])/2
    # V_10[7,21] = (V_10[6,21]+V_10[8,21])/2

    return U_1, V_1

def reproyeccion_sobre_seccion(U_seccion,V_seccion,seccion,lat,lon):
    """
    Reproyeccion de U,V sobre las secciones de la campania STSF2013

    Parametros de entrada
    --------------------
    U_seccion: 2D numpy.array (prof,estacion) de Velocidad Zonal de la seccion regrillada c/ 1 o 10 metros
    V_seccion: 2D numpy.array (prof,estacion) de Velocidad Meridional de la seccion regrillada c/ 1 o 10 metros
    seccion: 1,2,3,4,5,6 o 7 (int). Seccion a plotear
    lat: 1D numpy.array de latitudes de las estaciones de la seccion
    lon: 1D numpy.array de longitudes de las estaciones de la seccion

    Parametros de salida
    -------------------
    v_along: 2D numpy.array (prof,estacion) de Velocidad perpendicular de la seccion regrillada c/ 1 o 10 metros
    v_cross: 2D numpy.array (prof,estacion) de Velocidad paralela de la seccion regrillada c/ 1 o 10 metros

    """

    #NOTA: Se considera positivo hacia el norte y el este.


        # Proyeccion sobre seccion
    x1,y1   = lon[0] ,lat[0]
    x2,y2   = lon[len(lon)-1],lat[len(lat)-1]
    alpha_r = math.atan2((x2-x1),(y2-y1))   # ° de la seccion respecto de norte
    alpha_p = alpha_r - math.pi/2           # ° de la normal a la seccion respecto de norte

    beta = np.arctan2(U_seccion,V_seccion)          #angulos de cada nodo respecto al norte
    gamma = beta - alpha_p              #angulo entre el vector de velocidad y el vector normal a la seccion
    valong = np.sqrt(U_seccion**2+V_seccion**2)*np.cos(gamma)   # vel perpendicular
    vcross = np.sqrt(U_seccion**2+V_seccion**2)*np.sin(gamma)   # vel paralela

    return valong,vcross

def corte_seccion_along(seccion, U,V, DATA, suavizado = 'si'):
    """
    Plotea el corte vertical de la variable.

    Parámetros de entrada
    --------------------
    seccion: 1,2,3,4,5,6 o 7 (int). Seccion a plotear
    U: 2D numpy.array (prof,estacion) de Velocidad Zonal regrillada c/ 1 o 10 metros
    V: 2D numpy.array (prof,estacion) de Velocidad Meridional regrillada c/ 1 o 10 metros
    DATA: lista, cada elemento corresponde al archivo cnv de cada estacion.

    Parámetros de salida
    -------------------
    plot
    valong: 2D np.array (prof,ptos) de la vel alongshore regrillada.

    """
    #ingreso los datos de cada seccion
    if seccion == 1:
        prof = [DATA[0]['PRES'][-1], DATA[1]['PRES'][-1], DATA[2]['PRES'][-1], DATA[3]['PRES'][-1],DATA[4]['PRES'][-1],  DATA[5]['PRES'][-1], DATA[6]['PRES'][-1], DATA[7]['PRES'][-1]]
        lat = [DATA[0]['LATITUDE'][1], DATA[1]['LATITUDE'][1], DATA[2]['LATITUDE'][1],  DATA[3]['LATITUDE'][1], DATA[4]['LATITUDE'][1], DATA[5]['LATITUDE'][1], DATA[6]['LATITUDE'][1], DATA[7]['LATITUDE'][1]]
        lon = [DATA[0]['LONGITUDE'][1], DATA[1]['LONGITUDE'][1], DATA[2]['LONGITUDE'][1], DATA[3]['LONGITUDE'][1],DATA[4]['LONGITUDE'][1], DATA[5]['LONGITUDE'][1], DATA[6]['LONGITUDE'][1], DATA[7]['LONGITUDE'][1]]
        ind_i,ind_f = 0,8   #indices dentro de la matriz_grillada
        estaciones=['1' ,'2' ,'3' ,'4' ,'5' ,'6' ,'7' ,'8']
        letra = 'A'
    if seccion == 2:
        prof = [DATA[8]['PRES'][-1], DATA[9]['PRES'][-1], DATA[10]['PRES'][-1], 891,DATA[12]['PRES'][-1],  DATA[13]['PRES'][-1], DATA[14]['PRES'][-1], DATA[15]['PRES'][-1],DATA[16]['PRES'][-1],DATA[17]['PRES'][-1],DATA[18]['PRES'][-1]]
        lat = [DATA[8]['LATITUDE'][1], DATA[9]['LATITUDE'][1], DATA[10]['LATITUDE'][1],  DATA[11]['LATITUDE'][1], DATA[12]['LATITUDE'][1], DATA[13]['LATITUDE'][1], DATA[14]['LATITUDE'][1], DATA[15]['LATITUDE'][1],DATA[16]['LATITUDE'][1],DATA[17]['LATITUDE'][1],DATA[18]['LATITUDE'][1]]
        lon = [DATA[8]['LONGITUDE'][1], DATA[9]['LONGITUDE'][1], DATA[10]['LONGITUDE'][1], DATA[11]['LONGITUDE'][1],DATA[12]['LONGITUDE'][1], DATA[13]['LONGITUDE'][1], DATA[14]['LONGITUDE'][1], DATA[15]['LONGITUDE'][1],DATA[16]['LONGITUDE'][1],DATA[17]['LONGITUDE'][1],DATA[18]['LONGITUDE'][1]]
        ind_i,ind_f = 8,19   #indices dentro de la matriz_grillada
        estaciones=['9' ,'10' ,'11' ,'12' ,'13' ,'14' ,'15' ,'16','17','18','19']
        letra = 'E'
    if seccion == 3:
        prof = [DATA[19]['PRES'][-1], DATA[20]['PRES'][-1], DATA[21]['PRES'][-1], DATA[22]['PRES'][-1],DATA[23]['PRES'][-1],  DATA[24]['PRES'][-1], DATA[25]['PRES'][-1],DATA[26]['PRES'][-1],DATA[27]['PRES'][-1]][::-1]
        lat = [DATA[19]['LATITUDE'][1], DATA[20]['LATITUDE'][1], DATA[21]['LATITUDE'][1],  DATA[22]['LATITUDE'][1], DATA[23]['LATITUDE'][1], DATA[24]['LATITUDE'][1], DATA[25]['LATITUDE'][1], DATA[26]['LATITUDE'][1],DATA[27]['LATITUDE'][1]][::-1]
        lon = [DATA[19]['LONGITUDE'][1], DATA[20]['LONGITUDE'][1], DATA[21]['LONGITUDE'][1], DATA[22]['LONGITUDE'][1],DATA[23]['LONGITUDE'][1], DATA[24]['LONGITUDE'][1], DATA[25]['LONGITUDE'][1], DATA[26]['LONGITUDE'][1], DATA[27]['LONGITUDE'][1]][::-1]
        ind_i,ind_f = 19,28   #indices dentro de la matriz_grillada
        estaciones=['20','21','22','23','24','25','26','27','28'][::-1]
        letra = 'B'
    if seccion == 4:
        prof = [DATA[28]['PRES'][-1], DATA[29]['PRES'][-1], DATA[30]['PRES'][-1], DATA[31]['PRES'][-1]]
        lat = [DATA[28]['LATITUDE'][1], DATA[29]['LATITUDE'][1], DATA[30]['LATITUDE'][1],  DATA[31]['LATITUDE'][1]]
        lon = [DATA[28]['LONGITUDE'][1], DATA[29]['LONGITUDE'][1], DATA[30]['LONGITUDE'][1], DATA[31]['LONGITUDE'][1]]
        ind_i,ind_f = 28,32   #indices dentro de la matriz_grillada
        estaciones=['29' ,'30' ,'31' ,'32']
        letra = ''
    if seccion == 5:
        prof = [DATA[32]['PRES'][-1], DATA[33]['PRES'][-1], DATA[34]['PRES'][-1], DATA[35]['PRES'][-1],DATA[36]['PRES'][-1],  DATA[37]['PRES'][-1], DATA[38]['PRES'][-1], DATA[39]['PRES'][-1], DATA[40]['PRES'][-1]][::-1]
        lat = [DATA[32]['LATITUDE'][1], DATA[33]['LATITUDE'][1], DATA[34]['LATITUDE'][1],  DATA[35]['LATITUDE'][1], DATA[36]['LATITUDE'][1], DATA[37]['LATITUDE'][1], DATA[38]['LATITUDE'][1], DATA[39]['LATITUDE'][1], DATA[40]['LATITUDE'][1]][::-1]
        lon = [DATA[32]['LONGITUDE'][1], DATA[33]['LONGITUDE'][1], DATA[34]['LONGITUDE'][1], DATA[35]['LONGITUDE'][1],DATA[36]['LONGITUDE'][1], DATA[37]['LONGITUDE'][1], DATA[38]['LONGITUDE'][1], DATA[39]['LONGITUDE'][1], DATA[40]['LONGITUDE'][1]][::-1]
        ind_i,ind_f = 32,41   #indices dentro de la matriz_grillada
        estaciones=['33','34','35','36','37','38','39','40','41'][::-1]
        letra = 'C'
    if seccion == 6:
        prof = [DATA[41]['PRES'][-1], DATA[42]['PRES'][-1], DATA[43]['PRES'][-1], DATA[44]['PRES'][-1]]
        lat = [DATA[41]['LATITUDE'][1], DATA[42]['LATITUDE'][1], DATA[43]['LATITUDE'][1],  DATA[44]['LATITUDE'][1]]
        lon = [DATA[41]['LONGITUDE'][1], DATA[42]['LONGITUDE'][1], DATA[43]['LONGITUDE'][1], DATA[44]['LONGITUDE'][1]]
        ind_i,ind_f = 41,45   #indices dentro de la matriz_grillada
        estaciones=['42','43','44','45']
        letra = ''
    if seccion == 7:
        prof = [DATA[45]['PRES'][-1],DATA[46]['PRES'][-1], DATA[47]['PRES'][-1], DATA[48]['PRES'][-1], DATA[49]['PRES'][-1],DATA[50]['PRES'][-1],   77,78,51,40][::-1]
        lat = [DATA[45]['LATITUDE'][1],DATA[46]['LATITUDE'][1], DATA[47]['LATITUDE'][1], DATA[48]['LATITUDE'][1],  DATA[49]['LATITUDE'][1], DATA[50]['LATITUDE'][1], -31.8323, -31.7368333333, -31.64584, -31.558333333][::-1]
        lon = [DATA[45]['LONGITUDE'][1],DATA[46]['LONGITUDE'][1], DATA[47]['LONGITUDE'][1], DATA[48]['LONGITUDE'][1], DATA[49]['LONGITUDE'][1],DATA[50]['LONGITUDE'][1], -50.5816666667, -50.68952, -50.7929, -50.915216667][::-1]
        ind_i,ind_f = 45,55   #indices dentro de la matriz_grillada
        estaciones=['46','47','48','49','50','51','52','53','54','55'][::-1]
        letra = 'D'
    #extension de los datos al fondo:
    U_seccion = U[:,ind_i:ind_f]
    V_seccion = V[:,ind_i:ind_f]
    for st in range(0,len(prof)):
        primero_con_nan = len(U_seccion[:,st][np.isnan(U_seccion[:,st]) == False])
        U_seccion[primero_con_nan:,st] = U_seccion[primero_con_nan-1,st]
        V_seccion[primero_con_nan:,st] = V_seccion[primero_con_nan-1,st]
    if seccion in [3,5,7]:
        U_seccion = U_seccion[:,::-1]
        V_seccion = V_seccion[:,::-1]
    if seccion == 1:
        U_seccion[:,2] = 0.75*U_seccion[:,1]+0.25*U_seccion[:,3]
        V_seccion[:,2] = 0.75*V_seccion[:,1]+0.25*V_seccion[:,3]

    #Reproyeccion sobre seccion:
    valong,vcross = reproyeccion_sobre_seccion(U_seccion,V_seccion,seccion,lat,lon)

    #arreglo seccion 7 p/plotear:
    if seccion == 7:
        valong[valong>-20]=-20
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
        if valong.shape[0] == 5000:
            z = np.linspace(0,4999,5000)
        elif valong.shape[0] == 20:
            z = np.linspace(5,195,20) #Profs

    #Interpolacion entre estaciones con una resolucion de 100ptos por seccion:
    dist_aux,z_aux = np.meshgrid(dist,z)
    points = (dist_aux.flatten(),z_aux.flatten())
    values = valong.flatten()
    x,p = np.meshgrid(np.linspace(0,dist[-1],100),z)
    data = griddata(points,values,(x,p),method='linear')

    levels = np.linspace(-60,60,13)
    levels_label = [-100,-80,-60,-40,-20,0,20,40,60,80,100]
    levels_contorno = np.linspace(-100,100,21)
    cmap = plt.cm.get_cmap('coolwarm',12)
    txt = 'Velocity alongshore - Section '+ str(seccion)
    txt2 = letra + ')'
    txt3 = r'$cm.s^{-1}$'
    vmin, vmax = -60,60

    if suavizado == 'si':    data = fCTD.suavizado(data)

    fig1 = plt.figure(figsize=(16,8))
    CF = plt.contourf(x[:210,:],p[:210,:],data[:210,:],levels,cmap=cmap,vmin=vmin,vmax=vmax, extend = 'both')
    CC = plt.contour(x[:210,:],p[:210,:],data[:210,:],levels_contorno, colors=('k'),linewidths=1.5)
    plt.clabel(CC,levels_label,inline=1, fmt='%1.f', fontsize=24)

    for est in range(0,len(prof)):
        plt.text(dist[est]-0.2,-4,estaciones[est],size = 24)
    plt.scatter(dist,np.zeros(len(dist)),marker = "v",color = 'k',s=200)

    plt.text(5,180,txt,size = 30, bbox=dict(facecolor='white'))
    plt.plot(dist,prof,'*-')
    plt.axis([0,156,200,0])
    plt.xlabel('Distance (km)', size=30)
    plt.ylabel('Pressure (db)', size=30)
    plt.xticks(size = 30)
    plt.yticks(size = 30)
    plt.text(3,195,txt2,size = 30)

    cbar = fig1.colorbar(CF,orientation='vertical')
    cbar.set_label(label=txt3,fontsize=30)
    cbar.ax.tick_params(labelsize=30)

    plt.fill_between(dist, 1000, prof, facecolor='white',zorder = 2)
    plt.show()
    return

def corte_seccion_cross(seccion, U,V, DATA, suavizado = 'si'):
    """
    Plotea el corte vertical de la variable.

    Parámetros de entrada
    --------------------
    seccion: 1,2,3,4,5,6 o 7 (int). Seccion a plotear
    U: 2D numpy.array (prof,estacion) de Velocidad Zonal regrillada c/ 1 o 10 metros
    V: 2D numpy.array (prof,estacion) de Velocidad Meridional regrillada c/ 1 o 10 metros
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
    U_seccion = U[:,ind_i:ind_f]
    V_seccion = V[:,ind_i:ind_f]
    for st in range(0,len(prof)):
        primero_con_nan = len(U_seccion[:,st][np.isnan(U_seccion[:,st]) == False])
        U_seccion[primero_con_nan:,st] = U_seccion[primero_con_nan-1,st]
        V_seccion[primero_con_nan:,st] = V_seccion[primero_con_nan-1,st]
    if seccion in [3,5,7]:
        U_seccion = U_seccion[:,::-1]
        V_seccion = V_seccion[:,::-1]
    if seccion == 1:
        U_seccion[:,2] = 0.75*U_seccion[:,1]+0.25*U_seccion[:,3]
        V_seccion[:,2] = 0.75*V_seccion[:,1]+0.25*V_seccion[:,3]

    #Reproyeccion sobre seccion:
    valong,vcross = reproyeccion_sobre_seccion(U_seccion,V_seccion,seccion,lat,lon)

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
        if valong.shape[0] == 5000:
            z = np.linspace(0,4999,5000)
        elif valong.shape[0] == 20:
            z = np.linspace(5,195,20) #Profs

    #Interpolacion entre estaciones con una resolucion de 100ptos por seccion:
    dist_aux,z_aux = np.meshgrid(dist,z)
    points = (dist_aux.flatten(),z_aux.flatten())
    values = vcross.flatten()
    x,p = np.meshgrid(np.linspace(0,dist[-1],100),z)
    data = griddata(points,values,(x,p),method='linear')

    levels = np.linspace(-60,60,13)
    levels_label = [-60,-40,-20,0,20,40,60]
    cmap = plt.cm.get_cmap('coolwarm',12)
    txt = 'Velocity cross-shore - Section '+ str(seccion)
    txt2 = 'A)'
    txt3 = r'$cm.s^{-1}$'
    vmin, vmax = -60,60

    if suavizado == 'si':    data = fCTD.suavizado(data)

    fig1 = plt.figure(figsize=(16,8))
    CF = plt.contourf(x[:210,:],p[:210,:],data[:210,:],levels,cmap=cmap,vmin=vmin,vmax=vmax,extend = 'both')
    CC = plt.contour(x[:210,:],p[:210,:],data[:210,:],levels, colors=('k'),linewidths=1.5)
    plt.clabel(CC,levels_label,inline=1, fmt='%1.f', fontsize=30)

    for est in range(0,len(prof)):
        plt.text(dist[est]-0.2,-4,estaciones[est],size = 30)
    plt.scatter(dist,np.zeros(len(dist)),marker = "v",color = 'k',s=200)

    plt.text(5,180,txt,size = 30, bbox=dict(facecolor='white'))
    plt.plot(dist,prof,'*-')
    plt.axis([0,dist[-1],200,0])
    plt.xlabel('Distance (km)', size=30)
    plt.ylabel('Depth (m)', size=30)
    plt.xticks(size = 30)
    plt.yticks(size = 30)
    plt.text(3,195,txt2,size = 30)

    cbar = fig1.colorbar(CF,orientation='vertical')
    cbar.set_label(label=txt3,fontsize=30)
    cbar.ax.tick_params(labelsize=30)

    plt.fill_between(dist, 1000, prof, facecolor='white')
    plt.show()

def gen_dist_2d(lon_r,lat_r,depth):
    """
    Genera la matriz 2D de distancias entre los puntos de la seccion
    repetidas en la profundidad. considerando que la distancia asociada al primer
    pto (estacion) es la mitad entre la primera y segunda. Analogo con la ultima.


    Parametros de entrada
    --------------------
    lon_r: 1D numpy.array (ptos) de longitudes de la recta
    lat_r: 1D numpy.array (ptos) de latitudes de la recta
    depth: 1D numpy.array (prof) de profundidades

    Parametros de salida
    -------------------
    dist2: 2D numpy.array (prof,ptos) de la distancia entre puntos repetidas en
     todas las profs.
    """

    dist = []
    for k in range(0,len(lon_r)):
        if k==0:
            p1 = ( lat_r[k],lon_r[k])
            p2 = (lat_r[k+1],lon_r[k+1])
            dist.append(1000*haversine(p1,p2)/2)
        elif k == len(lon_r)-1:
            p1 = ( lat_r[k],lon_r[k])
            p2 = (lat_r[k-1],lon_r[k-1])
            dist.append(1000*haversine(p1,p2)/2)
        else:
            p0 = ( lat_r[k-1],lon_r[k-1])
            p1 = ( lat_r[k],lon_r[k])
            p2 = (lat_r[k+1],lon_r[k+1])
            d1 = 1000*haversine(p1,p0)/2    # mitad de dist a la est. previa
            d2 = 1000*haversine(p1,p2)/2    # mitad de dist a la est. posterior
            dist.append(d1+d2)
    dist2 = np.matlib.repmat(dist,len(depth),1)
    return dist2

def transporte_seccion(DATA,seccion,U,V):
    """
    Calcula el transporte por seccion en los primeros 200 m

    Parametros de entrada
    -------------------
    seccion: 1,2,3,4,5,6 o 7 (int). Seccion a plotear
    U: 2D numpy.array (prof,estacion) de Velocidad Zonal regrillada c/ 1 metros
    V: 2D numpy.array (prof,estacion) de Velocidad Meridional regrillada c/ 1 metros
    DATA: lista, cada elemento corresponde al archivo cnv de cada estacion.

    Parametros de salida
    -------------------


    """
    depth = np.linspace(0,U.shape[0]-1,U.shape[0])
    if seccion == 1:
        prof = [DATA[0]['PRES'][-1], DATA[1]['PRES'][-1], DATA[2]['PRES'][-1], DATA[3]['PRES'][-1],DATA[4]['PRES'][-1],  DATA[5]['PRES'][-1], DATA[6]['PRES'][-1], DATA[7]['PRES'][-1]]
        lat = [DATA[0]['LATITUDE'][1], DATA[1]['LATITUDE'][1], DATA[2]['LATITUDE'][1],  DATA[3]['LATITUDE'][1], DATA[4]['LATITUDE'][1], DATA[5]['LATITUDE'][1], DATA[6]['LATITUDE'][1], DATA[7]['LATITUDE'][1]]
        lon = [DATA[0]['LONGITUDE'][1], DATA[1]['LONGITUDE'][1], DATA[2]['LONGITUDE'][1], DATA[3]['LONGITUDE'][1],DATA[4]['LONGITUDE'][1], DATA[5]['LONGITUDE'][1], DATA[6]['LONGITUDE'][1], DATA[7]['LONGITUDE'][1]]
        ind_i,ind_f = 0,8   #indices dentro de la matriz_grillada
        estaciones=['1' ,'2' ,'3' ,'4' ,'5' ,'6' ,'7' ,'8']
        letra = 'A'
    if seccion == 2:
        prof = [DATA[8]['PRES'][-1], DATA[9]['PRES'][-1], DATA[10]['PRES'][-1], 891,DATA[12]['PRES'][-1],  DATA[13]['PRES'][-1], DATA[14]['PRES'][-1], DATA[15]['PRES'][-1],DATA[16]['PRES'][-1],DATA[17]['PRES'][-1],DATA[18]['PRES'][-1]]
        lat = [DATA[8]['LATITUDE'][1], DATA[9]['LATITUDE'][1], DATA[10]['LATITUDE'][1],  DATA[11]['LATITUDE'][1], DATA[12]['LATITUDE'][1], DATA[13]['LATITUDE'][1], DATA[14]['LATITUDE'][1], DATA[15]['LATITUDE'][1],DATA[16]['LATITUDE'][1],DATA[17]['LATITUDE'][1],DATA[18]['LATITUDE'][1]]
        lon = [DATA[8]['LONGITUDE'][1], DATA[9]['LONGITUDE'][1], DATA[10]['LONGITUDE'][1], DATA[11]['LONGITUDE'][1],DATA[12]['LONGITUDE'][1], DATA[13]['LONGITUDE'][1], DATA[14]['LONGITUDE'][1], DATA[15]['LONGITUDE'][1],DATA[16]['LONGITUDE'][1],DATA[17]['LONGITUDE'][1],DATA[18]['LONGITUDE'][1]]
        ind_i,ind_f = 8,19   #indices dentro de la matriz_grillada
        estaciones=['9' ,'10' ,'11' ,'12' ,'13' ,'14' ,'15' ,'16','17','18','19']
        letra = 'E'
    if seccion == 3:
        prof = [DATA[19]['PRES'][-1], DATA[20]['PRES'][-1], DATA[21]['PRES'][-1], DATA[22]['PRES'][-1],DATA[23]['PRES'][-1],  DATA[24]['PRES'][-1], DATA[25]['PRES'][-1],DATA[26]['PRES'][-1],DATA[27]['PRES'][-1]][::-1]
        lat = [DATA[19]['LATITUDE'][1], DATA[20]['LATITUDE'][1], DATA[21]['LATITUDE'][1],  DATA[22]['LATITUDE'][1], DATA[23]['LATITUDE'][1], DATA[24]['LATITUDE'][1], DATA[25]['LATITUDE'][1], DATA[26]['LATITUDE'][1],DATA[27]['LATITUDE'][1]][::-1]
        lon = [DATA[19]['LONGITUDE'][1], DATA[20]['LONGITUDE'][1], DATA[21]['LONGITUDE'][1], DATA[22]['LONGITUDE'][1],DATA[23]['LONGITUDE'][1], DATA[24]['LONGITUDE'][1], DATA[25]['LONGITUDE'][1], DATA[26]['LONGITUDE'][1], DATA[27]['LONGITUDE'][1]][::-1]
        ind_i,ind_f = 19,28   #indices dentro de la matriz_grillada
        estaciones=['20','21','22','23','24','25','26','27','28'][::-1]
        letra = 'B'
    if seccion == 4:
        prof = [DATA[28]['PRES'][-1], DATA[29]['PRES'][-1], DATA[30]['PRES'][-1], DATA[31]['PRES'][-1]]
        lat = [DATA[28]['LATITUDE'][1], DATA[29]['LATITUDE'][1], DATA[30]['LATITUDE'][1],  DATA[31]['LATITUDE'][1]]
        lon = [DATA[28]['LONGITUDE'][1], DATA[29]['LONGITUDE'][1], DATA[30]['LONGITUDE'][1], DATA[31]['LONGITUDE'][1]]
        ind_i,ind_f = 28,32   #indices dentro de la matriz_grillada
        estaciones=['29' ,'30' ,'31' ,'32']
        letra = ''
    if seccion == 5:
        prof = [DATA[32]['PRES'][-1], DATA[33]['PRES'][-1], DATA[34]['PRES'][-1], DATA[35]['PRES'][-1],DATA[36]['PRES'][-1],  DATA[37]['PRES'][-1], DATA[38]['PRES'][-1], DATA[39]['PRES'][-1], DATA[40]['PRES'][-1]][::-1]
        lat = [DATA[32]['LATITUDE'][1], DATA[33]['LATITUDE'][1], DATA[34]['LATITUDE'][1],  DATA[35]['LATITUDE'][1], DATA[36]['LATITUDE'][1], DATA[37]['LATITUDE'][1], DATA[38]['LATITUDE'][1], DATA[39]['LATITUDE'][1], DATA[40]['LATITUDE'][1]][::-1]
        lon = [DATA[32]['LONGITUDE'][1], DATA[33]['LONGITUDE'][1], DATA[34]['LONGITUDE'][1], DATA[35]['LONGITUDE'][1],DATA[36]['LONGITUDE'][1], DATA[37]['LONGITUDE'][1], DATA[38]['LONGITUDE'][1], DATA[39]['LONGITUDE'][1], DATA[40]['LONGITUDE'][1]][::-1]
        ind_i,ind_f = 32,41   #indices dentro de la matriz_grillada
        estaciones=['33','34','35','36','37','38','39','40','41'][::-1]
        letra = 'C'
    if seccion == 6:
        prof = [DATA[41]['PRES'][-1], DATA[42]['PRES'][-1], DATA[43]['PRES'][-1], DATA[44]['PRES'][-1]]
        lat = [DATA[41]['LATITUDE'][1], DATA[42]['LATITUDE'][1], DATA[43]['LATITUDE'][1],  DATA[44]['LATITUDE'][1]]
        lon = [DATA[41]['LONGITUDE'][1], DATA[42]['LONGITUDE'][1], DATA[43]['LONGITUDE'][1], DATA[44]['LONGITUDE'][1]]
        ind_i,ind_f = 41,45   #indices dentro de la matriz_grillada
        estaciones=['42','43','44','45']
        letra = ''
    if seccion == 7:
        prof = [DATA[45]['PRES'][-1],DATA[46]['PRES'][-1], DATA[47]['PRES'][-1], DATA[48]['PRES'][-1], DATA[49]['PRES'][-1],DATA[50]['PRES'][-1],   77,78,51,40][::-1]
        lat = [DATA[45]['LATITUDE'][1],DATA[46]['LATITUDE'][1], DATA[47]['LATITUDE'][1], DATA[48]['LATITUDE'][1],  DATA[49]['LATITUDE'][1], DATA[50]['LATITUDE'][1], -31.8323, -31.7368333333, -31.64584, -31.558333333][::-1]
        lon = [DATA[45]['LONGITUDE'][1],DATA[46]['LONGITUDE'][1], DATA[47]['LONGITUDE'][1], DATA[48]['LONGITUDE'][1], DATA[49]['LONGITUDE'][1],DATA[50]['LONGITUDE'][1], -50.5816666667, -50.68952, -50.7929, -50.915216667][::-1]
        ind_i,ind_f = 45,55   #indices dentro de la matriz_grillada
        estaciones=['46','47','48','49','50','51','52','53','54','55'][::-1]
        letra = 'D'
    #extension de los datos al fondo:
    U_seccion = U[:,ind_i:ind_f]
    V_seccion = V[:,ind_i:ind_f]
    #Veloc alongshore [cm]
    valong,vcross = reproyeccion_sobre_seccion(U_seccion,V_seccion,seccion,lat,lon)

    dd2 = gen_dist_2d(lon,lat,depth)                # [m]
    dz2 = np.matlib.repmat(np.ones(len(depth)),len(lon),1).T      # [m]
    T = valong[:200,:]*dd2[:200,:]*dz2[:200,:]/100
    return T



















































###
