#%matplotlib inline    /bin/bash runTestCases_docker.sh
import numpy as np
import matplotlib.pyplot as plt 
from netCDF4 import Dataset,netcdftime,num2date
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import csv
#%% hru names
#hruidxID = [10033,10089,10209,10312,10327,10328,10330,10336,10338,10339,10349,10369,10371,
#            10375,10389,10410,10521,10523,10524,10526,10534,10544,10545,10590,10726]
#hruidxID = [10317,10327,10336,10369,10375,10524,10526,10726] #p12
hruidxID = [10030,10041,10042,10053,10054,10077,10078,10079,10081,10082,10083,10084,10085,10094,10095,
            10096,10097,10098,10103,10104,10105,10106,10107,10108,10109,10110,10111,10112,10113,10114,
            10115,10116,10117,10118,10119,10120,10122,10126,10156,10223,10224,10225,10234,10235,10336,
            10337,10342,10348,10365,10379,10380,10381,10392,10393,10394,10398,10399,10400,10408,10409,
            10410,10411,10412,10413,10414,10421,10423,10426,10427,10428,10429,10430,10431,10432,10433,
            10436,10437,10438,10439,10440,10441,10442,10443,10444,10453,10454,10460,10461,10462,10617,
            10625,10626,10628,10637,10650,10652,10653,10669,10670,10737,10749,10752,10756,10767,10768,
            10769,10788,10790,10794,10795,10796,10825,10826,10827,10833,10834,10835,10836,10837,10841,
            10850,10865,10866,10867,10879,10880,10881,10882,10883,10884,10885,10886,10887,10888,10904,
            10910,10911,10912,10942,10945,10948,10951,10958,10959,10960,11272,11273,11274]

hru_num = np.size(hruidxID)

#%% #Swamp Angel forcing data
#swampangel_forcing = open('swamp_angel_forcingdata2_corrected.csv', 'rb')
#sa_forcing = csv.reader(swampangel_forcing)#, delimiter=',')
with open("C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa2_bestSweSD/swamp_angel_forcingdata2_corrected_precipCalib_final_L.csv") as safd:
    reader = csv.reader(safd)
    data_forcing = [r for r in reader]
data_forcing2 = data_forcing[1:]
sa_fd_column = []
for csv_counter1 in range (len (data_forcing2)):
    for csv_counter2 in range (11):
        sa_fd_column.append(float(data_forcing2[csv_counter1][csv_counter2]))
sa_forcing=np.reshape(sa_fd_column,(len (data_forcing2),11))
#%% #Senator Beck basin forcing data-Air pressure
#fr = Dataset('testCases_data/inputData/fieldData/reynolds/forcing_above_aspen.nc')
sbFD = Dataset('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/testVegFunctionsImpact/SenatorBeck_forcing.nc')
#ap_sb = sbFD.variables['airpres'][:]
TimeSb = sbFD.variables['time'][:] # get values
t_unitS = sbFD.variables['time'].units # get unit  "days since 1950-01-01T00:00:00Z"

try :

    t_cal = sbFD.variables['time'].calendar

except AttributeError : # Attribute doesn't exist

    t_cal = u"gregorian" # or standard

tvalueSb = num2date(TimeSb, units=t_unitS, calendar=t_cal)
sbDate = [i.strftime("%Y-%m-%d %H:%M") for i in tvalueSb]

#%% # ***** How to get time value into Python DateTIme Objects *****
# time step for swamp Angel
first_day = 6087.0000
#last_day = 7943.04166667
interval = 0.04166667
time_lentgh = 18264
saTime = []
for counter1 in range (time_lentgh):
    saTime.append(first_day+interval*counter1)
#saTime = np.concatenate([TimeSb,extended_year])

tvalueSa = num2date(saTime, units=t_unitS, calendar=t_cal)
saDate = [i.strftime("%Y-%m-%d %H:%M") for i in tvalueSa] # to display dates as string #i.strftime("%Y-%m-%d %H:%M")
#%% swamp angel forcing data dataframe
sa_forcing2 = sa_forcing[:,[3,4,5,6,7,8,9,10]]   #[:,1:]  #np.delete(sa_forcing, 0, axis=1)
sa_df = pd.DataFrame (sa_forcing2, columns=['day','pptrate','SWRadAtm','LWRadAtm','airtemp','windspd','airpres','spechum'])
sa_df.set_index(pd.DatetimeIndex(saDate),inplace=True)
#%% swamp angel Temp and ppt average to select the year
temp_data=pd.Series(np.array(sa_df['airtemp']),index=pd.DatetimeIndex(saDate))
temp_meanyr=temp_data.resample("A").mean()

ppt_data=pd.Series(np.array(sa_df['pptrate']),index=pd.DatetimeIndex(saDate))
ppt_meanyr=ppt_data.resample("A").sum()
#%% # I go through the a forcing file from the test cases to see what our netcdfs need to look like
print sbFD.file_format
# read out variables, data types, and dimensions of original forcing netcdf
for varname in sbFD.variables.keys():
    var = sbFD.variables[varname]
    print (varname, var.dtype, var.dimensions, var.shape)

for dimname in sbFD.dimensions.keys():
    dim = sbFD.dimensions[dimname]
    #print(dimname, len(dim), dim.isunlimited())

atemp = sbFD.variables['airtemp'][:]
#%% make new nc file
new_fc_sa = Dataset("C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa2_bestSweSD/SwampAngel_forcingD.nc",'w',format='NETCDF3_CLASSIC')
# define dimensions 
hru = new_fc_sa.createDimension('hru', hru_num)
time = new_fc_sa.createDimension('time', None)
# define variables
hruid = new_fc_sa.createVariable('hruId', np.int32,('hru',))
lat = new_fc_sa.createVariable('latitude', np.float64,('hru',))
lon = new_fc_sa.createVariable('longitude', np.float64,('hru',))
ds = new_fc_sa.createVariable('data_step', np.float64)
times = new_fc_sa.createVariable('time', np.float64,('time',))
lwrad = new_fc_sa.createVariable('LWRadAtm', np.float64,('time','hru'), fill_value = -999.0)
swrad = new_fc_sa.createVariable('SWRadAtm', np.float64,('time','hru'), fill_value = -999.0)
airpres = new_fc_sa.createVariable('airpres', np.float64,('time','hru'), fill_value = -999.0)
airtemp = new_fc_sa.createVariable('airtemp', np.float64,('time','hru'), fill_value = -999.0)
pptrate = new_fc_sa.createVariable('pptrate', np.float64,('time','hru'), fill_value = -999.0)
spechum = new_fc_sa.createVariable('spechum', np.float64,('time','hru'), fill_value = -999.0)
windspd = new_fc_sa.createVariable('windspd', np.float64,('time','hru'), fill_value = -999.0)
# give variables units
times.units = 'days since 1990-01-01 00:00:00'
ds.units = 'seconds'
lwrad.units = 'W m-2'
swrad.units = 'W m-2'
airpres.units = 'Pa'
airtemp.units = 'K'
pptrate.units = 'kg m-2 s-1'
spechum.units = 'g g-1'
windspd.units = 'm s-1'
# give variables value type
lwrad.vtype = 'scalarv'
swrad.vtype = 'scalarv'
airpres.vtype = 'scalarv'
airtemp.vtype = 'scalarv'
pptrate.vtype = 'scalarv'
spechum.vtype = 'scalarv'
windspd.vtype = 'scalarv'
# read out to compare with original
#for varname in new_fc_sa.variables.keys():
#    var = new_fc_sa.variables[varname]
#    print (varname, var.dtype, var.dimensions, var.shape)
#%% define hru id, time step (1hr), lat and lon
step = np.array([3600])

lat_sa = np.array([37.906914133])#33333) np.array(sbFD.variables['latitude'][:])
len_lat = np.repeat(lat_sa[:,np.newaxis], hru_num, axis=1); len_lat=len_lat.reshape(hru_num,)

long_sa = np.array([360 - 107.711322011])#11111)np.array(sbFD.variables['longitude'][:])
len_lon= np.repeat(long_sa[:,np.newaxis], hru_num, axis=1); len_lon=len_lon.reshape(hru_num,)
#lat_sb = np.array(sbFD.variables['latitude'][:])
#len_lat = np.repeat(lat_sb[:,np.newaxis], hru_num, axis=1); len_lat=len_lat.reshape(hru_num,)
#long_sb =np.array(sbFD.variables['longitude'][:])
#len_lon= np.repeat(long_sb[:,np.newaxis], hru_num, axis=1); len_lon=len_lon.reshape(hru_num,)
#%%
# assign newly created variables with lists of values from NLDAS and Sagehen data
hruid[:] = hruidxID 
lat[:] = len_lat
lon[:] = len_lon
ds[:] = step

new_ix = np.array(saTime) #new_ix = sbFD.variables['time'][:]
times[:] = new_ix

lwr_sa = np.array(sa_df['LWRadAtm'])  #lwr_sb = sbFD.variables['LWRadAtm'][:]
lwr_sa_hru = np.repeat(lwr_sa[:,np.newaxis], hru_num, axis=1)
lwrad[:] = lwr_sa_hru

swr_sa = np.array(sa_df['SWRadAtm'])
swr_sa_hru = np.repeat(swr_sa[:,np.newaxis], hru_num, axis=1)
swrad[:] = swr_sa_hru

ap_sa = np.array(sa_df['airpres'])
ap_sa_hru = np.repeat(ap_sa[:,np.newaxis], hru_num, axis=1)
airpres[:] = ap_sa_hru

at_sa = np.array(sa_df['airtemp'])
at_sa_hru = np.repeat(at_sa[:,np.newaxis], hru_num, axis=1) 
airtemp[:] = at_sa_hru

ws_sa = np.array(sa_df['windspd'])
ws_sa_hru = np.repeat(ws_sa[:,np.newaxis], hru_num, axis=1) 
windspd[:] = ws_sa_hru

sh_sa = np.array(sa_df['spechum'])
sh_sa_hru = np.repeat(sh_sa[:,np.newaxis], hru_num, axis=1) 
spechum[:] = sh_sa_hru

ppt_sa = np.array(sa_df['pptrate'])
ppt_sa_hru = np.repeat(ppt_sa[:,np.newaxis], hru_num, axis=1) 
pptrate[:] = ppt_sa_hru

#%%specific humididty calculations***********************************************
#at0_sb = sbFD.variables['airtemp'][:]
#
#e_t = (ap_sb * sh_sb)/0.622
#p_da = ap_sb - e_t
#e_star_t = 611*(np.exp((17.27*(at0_sb-273.15))/(at0_sb-273.15+237.3)))
#rh = e_t/e_star_t
#
#e_star_t2 = 611*(np.exp((17.27*(at0_sb+2-273.15))/(at0_sb+2-273.15+237.3)))
#e_star_t4 = 611*(np.exp((17.27*(at0_sb+4-273.15))/(at0_sb+4-273.15+237.3)))
#
#e_t2 = rh * e_star_t2
#e_t4 = rh * e_star_t4
#
#p_t2 = p_da + e_t2
#p_t4 = p_da + e_t4
#
#sh_t2 = 0.622 * e_t2 / p_t2
#sh_t4 = 0.622 * e_t4 / p_t4
#%%******************************************************************************
test = new_fc_sa.variables['pptrate'][:]

# close the file to write it
new_fc_sa.close()
#%%
testfd = Dataset("C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa2_bestSweSD/SwampAngel_forcingD.nc")
print testfd.variables['time'][:]
# read out variables, data types, and dimensions of original forcing netcdf
for varname in testfd.variables.keys():
    var = testfd.variables[varname]
    print (varname, var.dtype, var.dimensions, var.shape)








