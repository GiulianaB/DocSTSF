"""
Descarga de datos SMOS por SFTP
Todo esto se pone en la terminal y descarga esos archivos (octubre2013) en el path de la terminal.
"""
sftp -oPort=27500 GiulianaBerden@becftp.icm.csic.es

cd data/OCEAN/SSS/SMOS/Global/v2.0/L3/2013

get BEC_SSS_L3_B_20131001T000000_20131001T235959_025_002.nc
get BEC_SSS_L3_B_20131002T000000_20131002T235959_025_002.nc
get BEC_SSS_L3_B_20131003T000000_20131003T235959_025_002.nc
get BEC_SSS_L3_B_20131004T000000_20131004T235959_025_002.nc
get BEC_SSS_L3_B_20131005T000000_20131005T235959_025_002.nc
get BEC_SSS_L3_B_20131006T000000_20131006T235959_025_002.nc
get BEC_SSS_L3_B_20131007T000000_20131007T235959_025_002.nc
get BEC_SSS_L3_B_20131008T000000_20131008T235959_025_002.nc
get BEC_SSS_L3_B_20131009T000000_20131009T235959_025_002.nc
get BEC_SSS_L3_B_20131010T000000_20131010T235959_025_002.nc
get BEC_SSS_L3_B_20131011T000000_20131011T235959_025_002.nc
get BEC_SSS_L3_B_20131012T000000_20131012T235959_025_002.nc
get BEC_SSS_L3_B_20131013T000000_20131013T235959_025_002.nc
get BEC_SSS_L3_B_20131014T000000_20131014T235959_025_002.nc
get BEC_SSS_L3_B_20131015T000000_20131015T235959_025_002.nc
get BEC_SSS_L3_B_20131016T000000_20131016T235959_025_002.nc
get BEC_SSS_L3_B_20131017T000000_20131017T235959_025_002.nc
get BEC_SSS_L3_B_20131018T000000_20131018T235959_025_002.nc
get BEC_SSS_L3_B_20131019T000000_20131019T235959_025_002.nc
get BEC_SSS_L3_B_20131020T000000_20131020T235959_025_002.nc
get BEC_SSS_L3_B_20131021T000000_20131021T235959_025_002.nc
get BEC_SSS_L3_B_20131022T000000_20131022T235959_025_002.nc
get BEC_SSS_L3_B_20131023T000000_20131023T235959_025_002.nc
get BEC_SSS_L3_B_20131024T000000_20131024T235959_025_002.nc
get BEC_SSS_L3_B_20131025T000000_20131025T235959_025_002.nc
get BEC_SSS_L3_B_20131026T000000_20131026T235959_025_002.nc
get BEC_SSS_L3_B_20131027T000000_20131027T235959_025_002.nc
get BEC_SSS_L3_B_20131028T000000_20131028T235959_025_002.nc
get BEC_SSS_L3_B_20131029T000000_20131029T235959_025_002.nc
get BEC_SSS_L3_B_20131030T000000_20131030T235959_025_002.nc
get BEC_SSS_L3_B_20131031T000000_20131031T235959_025_002.nc

"""
Descarga de datos Wind http://www.remss.com/ por FTP
"""
ftp ftp.remss.com
Name: giuliberden@gmail.com
Password: giuliberden@gmail.com

cd ccmp/v02.0/Y2013
binary
cd M01
get CCMP_Wind_Analysis_20130101_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130102_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130103_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130104_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130105_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130106_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130107_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130108_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130109_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130110_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130111_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130112_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130113_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130114_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130115_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130116_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130117_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130118_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130119_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130120_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130121_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130122_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130123_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130124_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130125_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130126_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130127_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130128_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130129_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130130_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130131_V02.0_L3.0_RSS.nc

cd ..
cd M02
get CCMP_Wind_Analysis_20130201_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130202_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130203_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130204_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130205_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130206_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130207_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130208_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130209_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130210_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130211_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130212_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130213_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130214_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130215_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130216_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130217_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130218_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130219_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130220_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130221_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130222_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130223_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130224_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130225_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130226_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130227_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130228_V02.0_L3.0_RSS.nc


cd ..
cd M03
get CCMP_Wind_Analysis_20130301_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130302_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130303_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130304_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130305_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130306_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130307_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130308_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130309_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130310_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130311_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130312_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130313_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130314_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130315_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130316_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130317_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130318_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130319_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130320_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130321_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130322_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130323_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130324_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130325_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130326_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130327_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130328_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130329_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130330_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130331_V02.0_L3.0_RSS.nc

cd ..
cd M04
get CCMP_Wind_Analysis_20130401_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130402_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130403_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130404_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130405_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130406_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130407_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130408_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130409_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130410_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130411_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130412_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130413_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130414_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130415_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130416_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130417_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130418_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130419_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130420_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130421_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130422_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130423_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130424_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130425_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130426_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130427_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130428_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130429_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130430_V02.0_L3.0_RSS.nc

cd ..
cd M05
get CCMP_Wind_Analysis_20130501_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130502_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130503_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130504_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130505_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130506_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130507_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130508_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130509_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130510_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130511_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130512_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130513_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130514_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130515_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130516_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130517_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130518_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130519_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130520_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130521_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130522_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130523_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130524_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130525_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130526_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130527_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130528_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130529_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130530_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130531_V02.0_L3.0_RSS.nc

cd ..
cd M06
get CCMP_Wind_Analysis_20130601_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130602_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130603_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130604_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130605_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130606_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130607_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130608_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130609_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130610_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130611_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130612_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130613_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130614_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130615_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130616_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130617_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130618_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130619_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130620_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130621_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130622_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130623_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130624_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130625_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130626_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130627_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130628_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130629_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130630_V02.0_L3.0_RSS.nc

cd ..
cd M07
get CCMP_Wind_Analysis_20130701_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130702_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130703_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130704_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130705_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130706_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130707_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130708_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130709_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130710_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130711_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130712_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130713_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130714_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130715_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130716_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130717_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130718_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130719_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130720_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130721_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130722_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130723_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130724_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130725_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130726_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130727_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130728_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130729_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130730_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130731_V02.0_L3.0_RSS.nc

cd ..
cd M08
get CCMP_Wind_Analysis_20130801_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130802_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130803_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130804_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130805_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130806_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130807_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130808_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130809_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130810_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130811_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130812_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130813_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130814_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130815_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130816_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130817_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130818_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130819_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130820_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130821_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130822_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130823_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130824_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130825_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130826_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130827_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130828_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130829_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130830_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130831_V02.0_L3.0_RSS.nc

cd ..
cd M09
get CCMP_Wind_Analysis_20130901_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130902_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130903_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130904_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130905_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130906_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130907_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130908_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130909_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130910_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130911_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130912_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130913_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130914_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130915_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130916_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130917_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130918_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130919_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130920_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130921_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130922_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130923_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130924_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130925_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130926_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130927_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130928_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130929_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20130930_V02.0_L3.0_RSS.nc

cd ..
cd M10
get CCMP_Wind_Analysis_20131001_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131002_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131003_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131004_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131005_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131006_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131007_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131008_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131009_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131010_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131011_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131012_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131013_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131014_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131015_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131016_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131017_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131018_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131019_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131020_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131021_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131022_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131023_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131024_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131025_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131026_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131027_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131028_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131029_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131030_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131031_V02.0_L3.0_RSS.nc

cd ..
cd M11
get CCMP_Wind_Analysis_20131101_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131102_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131103_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131104_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131105_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131106_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131107_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131108_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131109_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131110_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131111_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131112_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131113_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131114_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131115_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131116_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131117_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131118_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131119_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131120_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131121_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131122_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131123_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131124_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131125_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131126_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131127_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131128_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131129_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131130_V02.0_L3.0_RSS.nc

cd ..
cd M12
get CCMP_Wind_Analysis_20131201_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131202_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131203_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131204_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131205_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131206_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131207_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131208_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131209_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131210_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131211_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131212_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131213_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131214_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131215_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131216_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131217_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131218_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131219_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131220_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131221_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131222_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131223_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131224_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131225_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131226_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131227_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131228_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131229_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131230_V02.0_L3.0_RSS.nc
get CCMP_Wind_Analysis_20131231_V02.0_L3.0_RSS.nc






























##
