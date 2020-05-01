"""
Nutrientes
Mapa de oxigeno
"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import f_CTD_exploratorio as fCTD
import f_nutrientes as fn
import f_ADCP_exploratorio as fADCP
U_1, V_1 = fADCP.datos_ADCP_cada_1m()


DATA,lat_CTD,lon_CTD = fCTD.carga_datos_CTD()

T_CTD,S_CTD,O2_CTD,Fluo_CTD,z_CTD = fCTD.regrillado_vertical(DATA)

Sal_B, T_B, NN_B, F_B, S_B, Cl_B = fn.regrillado_vertical_nutrientes()

fn.mapa_O2_vel(lon_CTD,lat_CTD,O2_CTD,prof = 0)
fn.mapa_fluo_vel(lon_CTD,lat_CTD,Fluo_CTD,prof = 0)
fn.mapa_fosfato_vel(lon_CTD[:51],lat_CTD[:51],F_B,prof = 0)

prof = 10
fn.mapa_O2_fluo_vel(lon_CTD,lat_CTD,O2_CTD,Fluo_CTD,prof)

















###
