"""
SAT O2 y Fluorescencia:
"""
path_gral = '/media/giuliana/Disco1TB/'
import numpy as np
from seabird import fCNV
import pandas as pd

#arrays de sat1..52
for cargo_datos_CTD in range(1):
    [A01,A02] = [fCNV(path_gral + 'Doctorado/Datos/STSF_data/d01.cnv'),fCNV(path_gral + 'Doctorado/Datos/STSF_data/d02.cnv')]
    [A03,A04] = [fCNV(path_gral + 'Doctorado/Datos/STSF_data/d03.cnv'),fCNV(path_gral + 'Doctorado/Datos/STSF_data/d04.cnv')]
    [A05,A06] = [fCNV(path_gral + 'Doctorado/Datos/STSF_data/d05.cnv'),fCNV(path_gral + 'Doctorado/Datos/STSF_data/d06.cnv')]
    [A07,A08] = [fCNV(path_gral + 'Doctorado/Datos/STSF_data/d07.cnv'),fCNV(path_gral + 'Doctorado/Datos/STSF_data/d08.cnv')]
    [A09,A010] = [fCNV(path_gral + 'Doctorado/Datos/STSF_data/d09.cnv'),fCNV(path_gral + 'Doctorado/Datos/STSF_data/d010.cnv')]
    A011 = fCNV(path_gral + 'Doctorado/Datos/STSF_data/d011.cnv')
    [A013,A014] = [fCNV(path_gral + 'Doctorado/Datos/STSF_data/d013.cnv'),fCNV(path_gral + 'Doctorado/Datos/STSF_data/d014.cnv')]
    [A015,A016] = [fCNV(path_gral + 'Doctorado/Datos/STSF_data/d015.cnv'),fCNV(path_gral + 'Doctorado/Datos/STSF_data/d016.cnv')]
    [A017,A018] = [fCNV(path_gral + 'Doctorado/Datos/STSF_data/d017.cnv'),fCNV(path_gral + 'Doctorado/Datos/STSF_data/d018.cnv')]
    [A019,A020] = [fCNV(path_gral + 'Doctorado/Datos/STSF_data/d019.cnv'),fCNV(path_gral + 'Doctorado/Datos/STSF_data/d020.cnv')]
    [A021,A022] = [fCNV(path_gral + 'Doctorado/Datos/STSF_data/d021.cnv'),fCNV(path_gral + 'Doctorado/Datos/STSF_data/d022.cnv')]
    [A023,A024] = [fCNV(path_gral + 'Doctorado/Datos/STSF_data/d023.cnv'),fCNV(path_gral + 'Doctorado/Datos/STSF_data/d024.cnv')]
    [A025,A026] = [fCNV(path_gral + 'Doctorado/Datos/STSF_data/d025.cnv'),fCNV(path_gral + 'Doctorado/Datos/STSF_data/d026.cnv')]
    [A027,A028] = [fCNV(path_gral + 'Doctorado/Datos/STSF_data/d027.cnv'),fCNV(path_gral + 'Doctorado/Datos/STSF_data/d028.cnv')]
    [A029,A030] = [fCNV(path_gral + 'Doctorado/Datos/STSF_data/d029.cnv'),fCNV(path_gral + 'Doctorado/Datos/STSF_data/d030.cnv')]
    [A031,A032] = [fCNV(path_gral + 'Doctorado/Datos/STSF_data/d031.cnv'),fCNV(path_gral + 'Doctorado/Datos/STSF_data/d032.cnv')]
    [A033,A034] = [fCNV(path_gral + 'Doctorado/Datos/STSF_data/d033.cnv'),fCNV(path_gral + 'Doctorado/Datos/STSF_data/d034.cnv')]
    [A035,A036] = [fCNV(path_gral + 'Doctorado/Datos/STSF_data/d035.cnv'),fCNV(path_gral + 'Doctorado/Datos/STSF_data/d036.cnv')]
    [A037,A038] = [fCNV(path_gral + 'Doctorado/Datos/STSF_data/d037.cnv'),fCNV(path_gral + 'Doctorado/Datos/STSF_data/d038.cnv')]
    [A039,A040] = [fCNV(path_gral + 'Doctorado/Datos/STSF_data/d039.cnv'),fCNV(path_gral + 'Doctorado/Datos/STSF_data/d040.cnv')]
    [A041,A042] = [fCNV(path_gral + 'Doctorado/Datos/STSF_data/d041.cnv'),fCNV(path_gral + 'Doctorado/Datos/STSF_data/d042.cnv')]
    [A043,A044] = [fCNV(path_gral + 'Doctorado/Datos/STSF_data/d043.cnv'),fCNV(path_gral + 'Doctorado/Datos/STSF_data/d044.cnv')]
    [A045,A046] = [fCNV(path_gral + 'Doctorado/Datos/STSF_data/d045.cnv'),fCNV(path_gral + 'Doctorado/Datos/STSF_data/d046.cnv')]
    [A047,A048] = [fCNV(path_gral + 'Doctorado/Datos/STSF_data/d047.cnv'),fCNV(path_gral + 'Doctorado/Datos/STSF_data/d048.cnv')]
    [A049,A050] = [fCNV(path_gral + 'Doctorado/Datos/STSF_data/d049.cnv'),fCNV(path_gral + 'Doctorado/Datos/STSF_data/d050.cnv')]
    A051 = fCNV(path_gral + 'Doctorado/Datos/STSF_data/d051.cnv')
    A052=pd.read_excel(path_gral + 'Doctorado/Datos/STSF_data/EO_52Down_cal_f.xlsx')
    A053=pd.read_excel(path_gral + 'Doctorado/Datos/STSF_data/EO_53Down_cal_f.xlsx')
    A054=pd.read_excel(path_gral + 'Doctorado/Datos/STSF_data/EO_54Down_cal_f.xlsx')
    A055=pd.read_excel(path_gral + 'Doctorado/Datos/STSF_data/EO_55Down_cal_f.xlsx')
    A012=pd.read_excel(path_gral + 'Doctorado/Datos/STSF_data/d012.xlsx')

    sat1 = A01['oxigen_ml_L'].data*100/A01['oxsolML/L'].data
    sat2 = A02['oxigen_ml_L'].data*100/A02['oxsolML/L'].data
    sat3 = A03['oxigen_ml_L'].data*100/A03['oxsolML/L'].data
    sat4 = A04['oxigen_ml_L'].data*100/A04['oxsolML/L'].data  ; sat4[0:8] = 9.65*100/6.63  #sacado de las botellas
    sat5 = A05['oxigen_ml_L'].data*100/A05['oxsolML/L'].data
    sat6 = A06['oxigen_ml_L'].data*100/A06['oxsolML/L'].data
    sat7 = A07['oxigen_ml_L'].data*100/A07['oxsolML/L'].data
    sat8 = A08['oxigen_ml_L'].data*100/A08['oxsolML/L'].data
    sat9 = A09['oxigen_ml_L'].data*100/A09['oxsolML/L'].data
    sat10 = A010['oxigen_ml_L'].data*100/A010['oxsolML/L'].data
    sat11 = A011['oxigen_ml_L'].data*100/A011['oxsolML/L'].data
    sat12 = A012['oxigen_ml_L']*100/A012['oxsolML/L']
    sat13 = A013['oxigen_ml_L'].data*100/A013['oxsolML/L'].data
    sat14 = A014['oxigen_ml_L'].data*100/A014['oxsolML/L'].data
    sat15 = A015['oxigen_ml_L'].data*100/A015['oxsolML/L'].data
    sat16 = A016['oxigen_ml_L'].data*100/A016['oxsolML/L'].data
    sat17 = A017['oxigen_ml_L'].data*100/A017['oxsolML/L'].data
    sat18 = A018['oxigen_ml_L'].data*100/A018['oxsolML/L'].data
    sat19 = A019['oxigen_ml_L'].data*100/A019['oxsolML/L'].data
    sat20 = A020['oxigen_ml_L'].data*100/A020['oxsolML/L'].data
    sat21 = A021['oxigen_ml_L'].data*100/A021['oxsolML/L'].data
    sat22 = A022['oxigen_ml_L'].data*100/A022['oxsolML/L'].data
    sat23 = A023['oxigen_ml_L'].data*100/A023['oxsolML/L'].data
    sat24 = A024['oxigen_ml_L'].data*100/A024['oxsolML/L'].data
    sat25 = A025['oxigen_ml_L'].data*100/A025['oxsolML/L'].data
    sat26 = A026['oxigen_ml_L'].data*100/A026['oxsolML/L'].data
    sat27 = A027['oxigen_ml_L'].data*100/A027['oxsolML/L'].data
    sat28 = A028['oxigen_ml_L'].data*100/A028['oxsolML/L'].data
    sat29 = A029['oxigen_ml_L'].data*100/A029['oxsolML/L'].data
    sat30 = A030['oxigen_ml_L'].data*100/A030['oxsolML/L'].data
    sat31 = A031['oxigen_ml_L'].data*100/A031['oxsolML/L'].data
    sat32 = A032['oxigen_ml_L'].data*100/A032['oxsolML/L'].data
    sat33 = A033['oxigen_ml_L'].data*100/A033['oxsolML/L'].data
    sat34 = A034['oxigen_ml_L'].data*100/A034['oxsolML/L'].data
    sat35 = A035['oxigen_ml_L'].data*100/A035['oxsolML/L'].data
    sat36 = A036['oxigen_ml_L'].data*100/A036['oxsolML/L'].data
    sat37 = A037['oxigen_ml_L'].data*100/A037['oxsolML/L'].data
    sat38 = A038['oxigen_ml_L'].data*100/A038['oxsolML/L'].data
    sat39 = A039['oxigen_ml_L'].data*100/A039['oxsolML/L'].data
    sat40 = A040['oxigen_ml_L'].data*100/A040['oxsolML/L'].data
    sat41 = A041['oxigen_ml_L'].data*100/A041['oxsolML/L'].data
    sat42 = A042['oxigen_ml_L'].data*100/A042['oxsolML/L'].data
    sat43 = A043['oxigen_ml_L'].data*100/A043['oxsolML/L'].data
    sat44 = A044['oxigen_ml_L'].data*100/A044['oxsolML/L'].data
    sat45 = A045['oxigen_ml_L'].data*100/A045['oxsolML/L'].data
    sat46 = A046['oxigen_ml_L'].data*100/A046['oxsolML/L'].data
    sat47 = A047['oxigen_ml_L'].data*100/A047['oxsolML/L'].data
    sat48 = A048['oxigen_ml_L'].data*100/A048['oxsolML/L'].data
    sat49 = A049['oxigen_ml_L'].data*100/A049['oxsolML/L'].data
    sat50 = A050['oxigen_ml_L'].data*100/A050['oxsolML/L'].data
    sat51 = A051['oxigen_ml_L'].data*100/A051['oxsolML/L'].data

    c1 = A01['flSP'].data
    c2 = A02['flSP'].data
    c3 = A03['flSP'].data
    c4 = A04['flSP'].data
    c5 = A05['flSP'].data
    c6 = A06['flSP'].data
    c7 = A07['flSP'].data
    c8 = A08['flSP'].data
    c9 = A09['flSP'].data
    c10 = A010['flSP'].data
    c11 = A011['flSP'].data
    c12 = A012['flSP']
    c13 = A013['flSP'].data
    c14 = A014['flSP'].data
    c15 = A015['flSP'].data
    c16 = A016['flSP'].data
    c17 = A017['flSP'].data
    c18 = A018['flSP'].data
    c19 = A019['flSP'].data
    c20 = A020['flSP'].data
    c21 = A021['flSP'].data
    c22 = A022['flSP'].data
    c23 = A023['flSP'].data
    c24 = A024['flSP'].data
    c25 = A025['flSP'].data
    c26 = A026['flSP'].data
    c27 = A027['flSP'].data
    c28 = A028['flSP'].data
    c29 = A029['flSP'].data
    c30 = A030['flSP'].data
    c31 = A031['flSP'].data
    c32 = A032['flSP'].data
    c33 = A033['flSP'].data
    c34 = A034['flSP'].data
    c35 = A035['flSP'].data
    c36 = A036['flSP'].data
    c37 = A037['flSP'].data
    c38 = A038['flSP'].data
    c39 = A039['flSP'].data
    c40 = A040['flSP'].data
    c41 = A041['flSP'].data
    c42 = A042['flSP'].data
    c43 = A043['flSP'].data
    c44 = A044['flSP'].data
    c45 = A045['flSP'].data
    c46 = A046['flSP'].data
    c47 = A047['flSP'].data
    c48 = A048['flSP'].data
    c49 = A049['flSP'].data
    c50 = A050['flSP'].data
    c51 = A051['flSP'].data

    p1 = A01['PRES'].data
    p2 = A02['PRES'].data
    p3 = A03['PRES'].data
    p4 = A04['PRES'].data
    p5 = A05['PRES'].data
    p6 = A06['PRES'].data
    p7 = A07['PRES'].data
    p8 = A08['PRES'].data
    p9 = A09['PRES'].data
    p10 = A010['PRES'].data
    p11 = A011['PRES'].data
    p12 = A012['PRES']
    p13 = A013['PRES'].data
    p14 = A014['PRES'].data
    p15 = A015['PRES'].data
    p16 = A016['PRES'].data
    p17 = A017['PRES'].data
    p18 = A018['PRES'].data
    p19 = A019['PRES'].data
    p20 = A020['PRES'].data
    p21 = A021['PRES'].data
    p22 = A022['PRES'].data
    p23 = A023['PRES'].data
    p24 = A024['PRES'].data
    p25 = A025['PRES'].data
    p26 = A026['PRES'].data
    p27 = A027['PRES'].data
    p28 = A028['PRES'].data
    p29 = A029['PRES'].data
    p30 = A030['PRES'].data
    p31 = A031['PRES'].data
    p32 = A032['PRES'].data
    p33 = A033['PRES'].data
    p34 = A034['PRES'].data
    p35 = A035['PRES'].data
    p36 = A036['PRES'].data
    p37 = A037['PRES'].data
    p38 = A038['PRES'].data
    p39 = A039['PRES'].data
    p40 = A040['PRES'].data
    p41 = A041['PRES'].data
    p42 = A042['PRES'].data
    p43 = A043['PRES'].data
    p44 = A044['PRES'].data
    p45 = A045['PRES'].data
    p46 = A046['PRES'].data
    p47 = A047['PRES'].data
    p48 = A048['PRES'].data
    p49 = A049['PRES'].data
    p50 = A050['PRES'].data
    p51 = A051['PRES'].data


    s1,t1 = A01['PSAL'].data , A01['TEMP'].data
    s2,t2 = A02['PSAL'].data , A02['TEMP'].data
    s3,t3 = A03['PSAL'].data , A03['TEMP'].data
    s4,t4 = A04['PSAL'].data , A04['TEMP'].data
    s5,t5 = A05['PSAL'].data , A05['TEMP'].data
    s6,t6 = A06['PSAL'].data , A06['TEMP'].data
    s7,t7 = A07['PSAL'].data , A07['TEMP'].data
    s8,t8 = A08['PSAL'].data , A08['TEMP'].data
    s9,t9 = A09['PSAL'].data , A09['TEMP'].data
    s10,t = A010['PSAL'].data, A010['TEMP'].data
    s11,t11 = A011['PSAL'].data, A011['TEMP'].data
    s12,t12 = A012['PSAL']     , A012['TEMP']
    s13,t13 = A013['PSAL'].data, A013['TEMP'].data
    s14,t14 = A014['PSAL'].data, A014['TEMP'].data
    s15,t15 = A015['PSAL'].data, A015['TEMP'].data
    s16,t16 = A016['PSAL'].data, A016['TEMP'].data
    s17,t17 = A017['PSAL'].data, A017['TEMP'].data
    s18,t18 = A018['PSAL'].data, A018['TEMP'].data
    s19,t19 = A019['PSAL'].data, A019['TEMP'].data
    s20,t20 = A020['PSAL'].data, A020['TEMP'].data
    s21,t21 = A021['PSAL'].data, A021['TEMP'].data
    s22,t22 = A022['PSAL'].data, A022['TEMP'].data
    s23,t23 = A023['PSAL'].data, A023['TEMP'].data
    s24,t24 = A024['PSAL'].data, A024['TEMP'].data
    s25,t25 = A025['PSAL'].data, A025['TEMP'].data
    s26,t26 = A026['PSAL'].data, A026['TEMP'].data
    s27,t27 = A027['PSAL'].data, A027['TEMP'].data
    s28,t28 = A028['PSAL'].data, A028['TEMP'].data
    s29,t29 = A029['PSAL'].data, A029['TEMP'].data
    s30,t30 = A030['PSAL'].data, A030['TEMP'].data
    s31,t31 = A031['PSAL'].data, A031['TEMP'].data
    s32,t32 = A032['PSAL'].data, A032['TEMP'].data
    s33,t33 = A033['PSAL'].data, A033['TEMP'].data
    s34,t34 = A034['PSAL'].data, A034['TEMP'].data
    s35,t35 = A035['PSAL'].data, A035['TEMP'].data
    s36,t36 = A036['PSAL'].data, A036['TEMP'].data
    s37,t37 = A037['PSAL'].data, A037['TEMP'].data
    s38,t38 = A038['PSAL'].data, A038['TEMP'].data
    s39,t39 = A039['PSAL'].data, A039['TEMP'].data
    s40,t40 = A040['PSAL'].data, A040['TEMP'].data
    s41,t41 = A041['PSAL'].data, A041['TEMP'].data
    s42,t42 = A042['PSAL'].data, A042['TEMP'].data
    s43,t43 = A043['PSAL'].data, A043['TEMP'].data
    s44,t44 = A044['PSAL'].data, A044['TEMP'].data
    s45,t45 = A045['PSAL'].data, A045['TEMP'].data
    s46,t46 = A046['PSAL'].data, A046['TEMP'].data
    s47,t47 = A047['PSAL'].data, A047['TEMP'].data
    s48,t48 = A048['PSAL'].data, A048['TEMP'].data
    s49,t49 = A049['PSAL'].data, A049['TEMP'].data
    s50,t50 = A050['PSAL'].data, A050['TEMP'].data
    s51,t51 = A051['PSAL'].data, A051['TEMP'].data

    S, T, SAT, FLUO, P = np.nan*np.ones((200,51)), np.nan*np.ones((200,51)), np.nan*np.ones((200,51)),np.nan*np.ones((200,51)), np.nan*np.ones((200,51))
    for st in range(1,52):
        if st != 12 and 200 > eval('p'+str(st)+'[-1]'):
            for z in range(len(eval('p'+str(st)))):
                S[z,st-1] = eval('s'+str(st))[z]
                T[z,st-1] = eval('t'+str(st))[z]
                SAT[z,st-1] = eval('sat'+str(st))[z]
                FLUO[z,st-1] = eval('c'+str(st))[z]
                P[z,st-1] = eval('p'+str(st))[z]

import matplotlib.pyplot as plt
import cmocean as cmo

## fig sat O2 vs Fluo
for fig_satO2_vs_Fluo in range(1):
    cmap_depth = plt.cm.get_cmap('cmo.deep',6)
    fig = plt.figure()
    ax = fig.add_axes([0.05,0.1,0.85,0.85])
    for st in range(1,52):
        plt.scatter(eval('sat'+str(st)),eval('c'+str(st)),c = eval('p'+str(st)),cmap=cmap_depth,vmin=-20,vmax=100)
    plt.xlabel('% Sat O2',size=24)
    plt.ylabel('Fluorescence [ug/l]',size=24)

    cax = fig.add_axes([0.92, 0.18,0.02,0.73])
    cbar = plt.colorbar(ticks = [0,20,40,60,80,100], cax=cax)
    cbar.ax.set_xlabel('Meters', size=18,rotation=45)
    cbar.ax.tick_params(labelsize=24)

    plt.show()

# Diagrama TS
for cosas_necesarias_para_hacer_diagrama_TS in range(1):
    from matplotlib.pyplot import axvspan
    import seawater as gsw
    ################PREPARO LA FIG###########################

    # Bordes de la figura
    smin, smax, tmin, tmax = 27, 37, 3, 23

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

# Plot SAT O2 (TS) ***********************************************
for fig_sat_O2 in range(1):

    cmap_sat = plt.cm.get_cmap('Purples', 4)
    fig = plt.figure(figsize=(20,12))
    ax = fig.add_axes([0.09,0.1,0.8,0.85])

    CD = ax.contour(si,ti,densidad, linestyles='dashed', colors='k',alpha=0.5)
    ax.clabel(CD, fontsize=22, inline=1, fmt='%1.0f') # Label every second level

    # plt.xlabel('Salinity', size=30)
    plt.ylabel('Temperature (°C)', size=30)
    plt.xticks(size = 30)
    plt.yticks([8,13,18,23], size = 30)

    CSAT = ax.scatter(S[::-1],T[::-1],c = SAT[::-1],cmap = cmap_sat,s = 100,edgecolors='k',linewidths=0.1,vmin=60,vmax=140)
    cax_sat = fig.add_axes([0.92, 0.18,0.02,0.73])
    cbar_SAT = plt.colorbar(CSAT,ticks = [60,80,100,120,140],orientation = 'vertical',cax = cax_sat)
    cbar_SAT.ax.tick_params(labelsize=24)
    cbar_SAT.ax.set_ylabel('%Sat O2', size=22)

    ax.text(27.5,5,'A)',size = 30)

    plt.show()


# Plot Fluo (TS) ************************************************
for fig_fluo in range(1):

    cmap_f = plt.cm.get_cmap('Greens', 5)

    fig = plt.figure(figsize=(20,12))
    ax = fig.add_axes([0.09,0.1,0.8,0.85])

    CD = ax.contour(si,ti,densidad, linestyles='dashed', colors='k',alpha=0.5)
    ax.clabel(CD, fontsize=22, inline=1, fmt='%1.0f') # Label every second level

    plt.xlabel('Salinity', size=30)
    plt.ylabel('Temperature (°C)', size=30)
    plt.xticks(size = 30)
    plt.yticks([8,13,18,23], size = 30)

    CFLUO = ax.scatter(S[::-1],T[::-1],c = FLUO[::-1],cmap = cmap_f,s = 100,edgecolors='k',linewidths=0.1,vmin=0,vmax=5)
    cax_fluo = fig.add_axes([0.92, 0.18,0.02,0.73])
    cbar_FLUO = plt.colorbar(CFLUO,orientation = 'vertical',cax = cax_fluo)
    cbar_FLUO.ax.tick_params(labelsize=24)
    cbar_FLUO.ax.set_ylabel('Fluorescence', size=22)

    ax.text(27.5,5,'C)',size = 30)

    plt.show()




"""
Nutrientes: botella.
Por ahora estan todas las estaciones cargadas y esta la matriz t
que tiene toda la info. FALTA: sacar las estaciones mas profundas
que 200m para ver lo que pasa con la plataforma unicamente.
"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import patches



a = pd.read_excel(path_gral + 'Doctorado/Datos/STSF_data/stsf_sal_o2_cla_nut.xls')

# matriz 3d p/las est. 1-51, las otras no tienen datos
# t = array 3d. t[estacion, z(irregular), (*)]
# (*) = pres,sal,temp,satox,ox,nitrato+nitrito(NN),fosfato(fos),silicato(sil),clorofila(clo)
t = np.nan * np.ones((51,12,9))
c = 0
for st in range(1,52):
    c=0
    for i in range(0,len(a['Station'])):
        if a['Station'] [i] == st:
            if math.isnan(a['nitrato umol/kg'][i]) == False:
                t[st-1,c,:] = [a['PrDM'][i],a['Sal00'][i],a['Potemp090C'][i],a['Sbeox0ML/L'][i],a['OxsolML/L'][i],a['nitrato umol/kg'][i]+a['nitrito umol/kg'][i],a['fosfato umol/kg'][i],a['silicato umol/kg'][i],a['clorofila (mg/m3)'][i]]
                c = c+1

# matriz 3d p/las est. de plataforma, las otras tienen np.nan
# t_plat = array 3d. t[est, z(irregular), (*)]
# (*) = pres[0],sal[1],temp[2],satox[3],ox[4],NN[5],fos[6],sil[7],clo[8]
t_plat = np.copy(t)
for st in range(51):
    if np.nanmax(t_plat[st,:,0]) > 200:
        t_plat[st,:,:] = np.nan * np.ones((12,9))
t_plat[2,2,6] = np.nan # outlayer

for fig_NN in range(1):

    cmap_NN = plt.cm.get_cmap('Greens', 4)
    levels_NN = [0,5,10,15,20]

    fig = plt.figure(figsize=(15,8))
    ax = fig.add_axes([0.08,0.1,0.83,0.85])

    CD = ax.contour(si,ti,densidad, linestyles='dashed', colors='k',alpha=0.3)
    ax.clabel(CD, fontsize=20, inline=1, fmt='%1.0f') # Label every second level

    # plt.xlabel('Salinity', size=30)
    plt.ylabel('Temperature (°C)', size=30)
    plt.xticks(size = 30)
    plt.yticks([8,13,18,23], size = 30)

    CNN = ax.scatter(t_plat[:,:,1],t_plat[:,:,2],c =t_plat[:,:,5],cmap = cmap_NN,s = 200,edgecolors='k',linewidths=0.5,vmin=0,vmax=20)
    cax_NN = fig.add_axes([0.93, 0.12,0.02,0.73])
    cbar_NN = plt.colorbar(CNN,orientation = 'vertical',cax = cax_NN, ticks = levels_NN)
    cbar_NN.ax.tick_params(labelsize=24)
    cbar_NN.ax.set_xlabel(' \nN+N \n[umol/kg]', size=24,rotation = 0)


    ax.text(34.5,6.38,'4',fontsize=22); ax.arrow(34.2,6.53,0.15,0,head_width=0.1,head_length=0.1)
    ax.text(36,14.82,'32',fontsize=22); ax.arrow(35.7,15,0.15,0,head_width=0.1,head_length=0.1)
    ax.text(35.8,13.9,'23',fontsize=22); ax.arrow(35.5,14.1,0.15,0,head_width=0.1,head_length=0.1)
    ax.text(35.6,12.64,'37',fontsize=22); ax.arrow(35.3,12.77,0.15,0,head_width=0.1,head_length=0.1)
    ax.text(36.1,15.5,'42',fontsize=22); ax.arrow(35.8,15.7,0.15,0,head_width=0.1,head_length=0.1)
    ax.text(34.6,10.8,'24',fontsize=22); ax.arrow(34.7,12.3,0,-0.7,head_width=0.1,head_length=0.1)
    ax.text(27.5,5,'A)',fontsize=30)


    plt.show()

for fig_sil in range(1):

    cmap_sil = plt.cm.get_cmap('Greens', 8)
    levels_sil = [0,8,16,24,32]

    fig = plt.figure(figsize=(15,8))
    ax = fig.add_axes([0.08,0.1,0.83,0.85])

    CD = ax.contour(si,ti,densidad, linestyles='dashed', colors='k',alpha=0.3)
    ax.clabel(CD, fontsize=20, inline=1, fmt='%1.0f') # Label every second level

    plt.xlabel('Salinity', size=30)
    plt.ylabel('Temperature (°C)', size=30)
    plt.xticks(size = 30)
    plt.yticks([8,13,18,23], size = 30)

    Csil = ax.scatter(t_plat[:,:,1],t_plat[:,:,2],c =t_plat[:,:,7],cmap = cmap_sil,s = 200,edgecolors='k',linewidths=0.5,vmin=0,vmax=32)
    cax_sil = fig.add_axes([0.93, 0.12,0.02,0.73])
    cbar_sil = plt.colorbar(Csil,orientation = 'vertical',cax = cax_sil, ticks = levels_sil)
    cbar_sil.ax.tick_params(labelsize=24)
    cbar_sil.ax.set_xlabel('  \nSilicate \n[umol/kg]', size=24,rotation = 0)
    # #################################
    # for i in range(t_plat.shape[0]):
    #     for z in range(t_plat.shape[1]):
    #         ax.text(t_plat[i,z,1]+0.2,t_plat[i,z,2],(z_max[i]-t_plat[i,z,0]+5),color='b')
    #         ax.text(t_plat[i,z,1]+0.05,t_plat[i,z,2],(t_plat[i,z,0]+5),color='k')
    # # z_max = []
    # # for i in range(t_plat.shape[0]):
    # #     if i == 11:
    # #         z_max.append(np.nanmax(A012['PRES'][:]))
    # #     else:
    # #         z_max.append(eval('A0'+str(i+1)+'["PRES"][-1]'))
    #
    # ################################
    ax.text(34.5,6.38,'4',fontsize = 22); ax.arrow(34.2,6.53,0.15,0,head_width=0.1,head_length=0.1)
    ax.text(34.1,7.7,'1',fontsize = 22); ax.arrow(33.8,7.8,0.15,0,head_width=0.1,head_length=0.1)
    ax.text(28.1,14.8,'41',fontsize = 22); ax.arrow(28.6,15,-0.15,0,head_width=0.1,head_length=0.1)
    ax.text(30,14.5,'39',fontsize = 22); ax.arrow(30.5,14.7,-0.15,0,head_width=0.1,head_length=0.1)
    ax.text(35.95,17,'44',fontsize = 22); ax.arrow(35.7,17.15,0.15,0,head_width=0.1,head_length=0.1)
    ax.text(36.2,16.3,'50',fontsize = 22); ax.arrow(35.9,16.55,0.15,0,head_width=0.1,head_length=0.1)
    ax.text(27.5,5,'B)',fontsize=30)


    plt.show()

for fig_fos in range(1):

    cmap_fos = plt.cm.get_cmap('Greens', 4)
    levels_fos = [0,0.6,1.2,1.8,2.4]

    fig = plt.figure(figsize=(15,8))
    ax = fig.add_axes([0.08,0.1,0.83,0.85])

    CD = ax.contour(si,ti,densidad, linestyles='dashed', colors='k',alpha=0.3)
    ax.clabel(CD, fontsize=20, inline=1, fmt='%1.0f') # Label every second level

    # plt.xlabel('Salinity', size=30)
    plt.ylabel('Temperature (°C)', size=30)
    plt.xticks(size = 30)
    plt.yticks([8,13,18,23], size = 30)

    Cfos = ax.scatter(t_plat[:,:,1], t_plat[:,:,2], c=t_plat[:,:,6],cmap=cmap_fos,s=200, edgecolors='k',linewidths=0.5, vmin=0, vmax=2.4)
    cax_fos = fig.add_axes([0.93, 0.12,0.02,0.73])
    cbar_fos = plt.colorbar(Cfos,orientation = 'vertical',cax=cax_fos, ticks=levels_fos)
    cbar_fos.ax.tick_params(labelsize=24)
    cbar_fos.ax.set_xlabel('  \nPhosphate \n[umol/kg]', size=24,rotation=0)

    ax.text(34.5,6.38,'5',fontsize=22); ax.arrow(34.2,6.53,0.15,0,head_width=0.1,head_length=0.1)
    ax.text(34.1,7.7,'2',fontsize=22); ax.arrow(33.8,7.8,0.15,0,head_width=0.1,head_length=0.1)
    ax.text(34.1,9,'25',fontsize=22); ax.arrow(33.8,9.2,0.15,0,head_width=0.1,head_length=0.1)
    ax.text(34.6,14.1,'28',fontsize=22); ax.arrow(33.8,9.2,0.15,0,head_width=0.1,head_length=0.1)
    ax.text(28.6,14.,'40',fontsize=22); ax.arrow(29.2,14.2,-0.15,0,head_width=0.1,head_length=0.1)

    ax.text(27.5,5,'C)',fontsize=30)


    plt.show()


# Fig silicato vs Salinidad
#Regresion lineal:
x, y = t_plat[:,:,1].flatten(),t_plat[:,:,7].flatten()
variable1 = 'Salinity'
variable2 = 'Silicate'
leyenda = 'Silicate vs Salinity'
color = 'darkblue'
LR = stats.linregress(x[x <= 33.8],y[x <= 33.8])
x_reg = np.linspace(np.nanmin(x), 33.8, 200)
y_reg = x_reg*LR.slope + LR.intercept
txt = 'Linear Regression '+'\n $R^{2} = $'+ str(LR.rvalue**2)[:4]

fig = plt.figure('Silicate vs Salinity')
ax = fig.add_axes([0.1,0.1,0.8,0.8])
ax.scatter(x, y, label = leyenda, s = 10, color = 'k')
#Regresion:
ax.plot(x_reg, y_reg, color = color)
ax.text (28.2,1,txt,c = 'b',size = 30)
#relative high Si
c1 = patches.Ellipse((35.5,15),   width=1.6, height=4,edgecolor = 'k',fill = False,linewidth= 2.8,linestyle='-')   #st 9/15/19
ax.add_artist(c1)
txt_Si = 'Relative high [Si]'
ax.text (34.5,17.4,txt_Si,size = 30)
ax.text (27.5,1,'D)',size = 30)

plt.grid(ls = '--',alpha = 0.5)
plt.xlabel(str(variable1), size = 30)
plt.ylabel(str(variable2)+'$ [umol.kg^{-1}]$', size = 30)
plt.xticks([28,30,32,34,36],fontsize=30); plt.yticks([10,20,30],fontsize=30)
plt.axis([27,37,0,36])
plt.show()
