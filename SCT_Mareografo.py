# Datos Mareografo San Clemente del Tuyu

import pandas as pd
import datetime
path_gral = '/media/giuliana/Disco1TB/'

def altura_San_Clemente_ASO():
    """
    Halla la altura del mar en San Clemente para agosto septiembre y octubre del 2013
    Parametro de salida
    ------------------
    SLA: Dataframe con Altura del mar observada ('Obs') y predicha('predicha')
    con el tiempo como index
    """

    path_SCT = path_gral + 'Doctorado/Datos/Mareografos_arg/San_Clemente.xlsx'

    #Agosto
    A_O =  pd.read_excel(path_SCT,'agosto2')
    A_O.index = A_O['Fecha']
    A_O = A_O.resample('1H').mean()

    A_P = pd.read_excel(path_SCT,'predicha_agosto')
    fecha = []
    for i in range(len(A_P['fecha'])):
        if i >= 1152:
            a = str(A_P['fecha'][i][-4:])+str(A_P['fecha'][i][3:5])+str(A_P['fecha'][i][:2])+str(A_P['hora'][i])[0:2]+str(A_P['hora'][i])[3:5]
        else:
            a = str(A_P['fecha'][i])[:4]+str(A_P['fecha'][i])[8:10]+str(A_P['fecha'][i])[5:7]+ str(A_P['hora'][i])[0:2]+str(A_P['hora'][i])[3:5]
        fecha.append(datetime.datetime.strptime(a,'%Y%m%d%H%M'))
    A_P.index = fecha
    A_P = A_P.resample('1H').mean()

    #Septiembre
    S_O =  pd.read_excel(path_SCT,'septiembre')
    fecha = []
    for i in range(len(S_O['Fecha'])):
      a = S_O['Fecha'][i][-4:]+S_O['Fecha'][i][4:6]+S_O['Fecha'][i][1:3]+ str(S_O['Hora'][i])[0:2]+str(S_O['Hora'][i])[3:5]
      fecha.append(datetime.datetime.strptime(a,'%Y%m%d%H%M'))
    S_O.index = fecha
    S_O = S_O.resample('1H').mean()

    S_P = pd.read_excel(path_SCT,'predicha_septiembre')
    fecha = []
    for i in range(len(S_P['fecha'])):
        if i >= 1152:
            a = str(S_P['fecha'][i][-4:])+str(S_P['fecha'][i][3:5])+str(S_P['fecha'][i][:2])+str(S_P['hora'][i])[0:2]+str(S_P['hora'][i])[3:5]
        else:
            a = str(S_P['fecha'][i])[:4]+str(S_P['fecha'][i])[8:10]+str(S_P['fecha'][i])[5:7]+ str(S_P['hora'][i])[0:2]+str(S_P['hora'][i])[3:5]
        fecha.append(datetime.datetime.strptime(a,'%Y%m%d%H%M'))
    S_P.index = fecha
    S_P = S_P.resample('1H').mean()

    #Octubre
    O_O = pd.read_excel(path_SCT,'octubre')
    O_O.index = O_O['fecha']
    O_O = O_O.drop(['fecha'],axis = 1)

    O_P = pd.read_excel(path_SCT,'predicha_octubre')
    O_P.index = O_P['fecha']
    O_P = O_P.drop(['fecha'],axis = 1)


    lala = [A_O,S_O,O_O/100]
    lala_P = [A_P,S_P,O_P/100]
    ASO_O = pd.concat(lala,sort = False)
    ASO_P = pd.concat(lala_P,sort = False)

    ASO = pd.merge(ASO_O,ASO_P, left_index=True, right_index=True)
    SLA = ASO
    return SLA

##
