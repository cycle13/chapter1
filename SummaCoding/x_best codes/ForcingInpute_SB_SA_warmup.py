#%matplotlib inline    /bin/bash runTestCases_docker.sh
import numpy as np
import matplotlib.pyplot as plt 
from netCDF4 import Dataset,netcdftime,num2date
import pandas as pd
import matplotlib.pyplot as plt
import itertools
#%% hru names
p1 = [700000] #albedoDecayRate
p2 = [0.8] #albedoMaxVisible 
p3 = [0.6] #albedoMinVisible  

p4 = [0.001] #z0Snow
p5 = [0.18,0.28,0.50] #windReductionParam
#p6 = [0.35]#,0.35,0.50] #fixedThermalCond_snow

def hru_ix_ID(p1, p2, p3, p4, p5):
    ix1 = np.arange(1,len(p1)+1)
    ix2 = np.arange(1,len(p2)+1)
    ix3 = np.arange(1,len(p3)+1)
    ix4 = np.arange(1,len(p4)+1)
    ix5 = np.arange(1,len(p5)+1)
    #ix6 = np.arange(1,len(p6)+1)
    
    c = list(itertools.product(ix1,ix2,ix3,ix4,ix5))
    ix_numlist=[]
    for tup in c:
        ix_numlist.append(''.join(map(str, tup)))
    new_list = [float(i) for i in ix_numlist]

    return(new_list)  

hruidxID = hru_ix_ID(p1, p2, p3, p4, p5)
hruidx_df = pd.DataFrame (hruidxID)

out_names = ['AvASsSWcTCj','AvASsSWcTCs']
hru_num = np.size(hruidxID)
#hru = ['hru{}'.format(i) for i in hruidxID]
hru_names =[]
for i in out_names:
    hru_names.append(['{}{}'.format(i, j) for j in hruidxID])

hru_names1 = np.reshape(hru_names,(6,1))
hru_names_df = pd.DataFrame (hru_names1)
#%% #Senator Beck basin forcing data

#fr = Dataset('testCases_data/inputData/fieldData/reynolds/forcing_above_aspen.nc')
sbFD = Dataset('0SenatorBeck_forcing0.nc')

print sbFD.variables.keys() 
print sbFD.variables['spechum']
#%% # ***** How to get time value into Python DateTIme Objects *****

TimeSb = sbFD.variables['time'][:] # get values
t_unitS = sbFD.variables['time'].units # get unit  "days since 1950-01-01T00:00:00Z"

try :

    t_cal = sbFD.variables['time'].calendar

except AttributeError : # Attribute doesn't exist

    t_cal = u"gregorian" # or standard

tvalueSb = num2date(TimeSb, units=t_unitS, calendar=t_cal)
DateSb = [i.strftime("%Y-%m-%d %H:%M") for i in tvalueSb] # to display dates as string #i.strftime("%Y-%m-%d %H:%M")

#%% # I go through the a forcing file from the test cases to see what our netcdfs need to look like
print sbFD.file_format
# read out variables, data types, and dimensions of original forcing netcdf
for varname in sbFD.variables.keys():
    var = sbFD.variables[varname]
    print (varname, var.dtype, var.dimensions, var.shape)

for dimname in sbFD.dimensions.keys():
    dim = sbFD.dimensions[dimname]
    #print(dimname, len(dim), dim.isunlimited())

print sbFD.variables['hruId'][:]
atemp = sbFD.variables['airtemp'][:]

#%% # instead of overwriting, make new nc file
# write new file
#test = Dataset("testCases_data/inputData/fieldData/SH/shT1_force2.nc",'w',format='NETCDF3_CLASSIC')
new_fc_sb = Dataset("SenatorBeck_forcing_T0v.nc",'w',format='NETCDF3_CLASSIC')
# define dimensions 
# HRU NEEDS TO BE THE NUMBER OF HRUs FROM THE PARAM TRIAL
hru = new_fc_sb.createDimension('hru', hru_num)
time = new_fc_sb.createDimension('time', None)
# define variables
hruid = new_fc_sb.createVariable('hruId', np.int32,('hru',))
lat = new_fc_sb.createVariable('latitude', np.float64,('hru',))
lon = new_fc_sb.createVariable('longitude', np.float64,('hru',))
ds = new_fc_sb.createVariable('data_step', np.float64)
times = new_fc_sb.createVariable('time', np.float64,('time',))
lwrad = new_fc_sb.createVariable('LWRadAtm', np.float64,('time','hru'), fill_value = -999.0)
swrad = new_fc_sb.createVariable('SWRadAtm', np.float64,('time','hru'), fill_value = -999.0)
airpres = new_fc_sb.createVariable('airpres', np.float64,('time','hru'), fill_value = -999.0)
airtemp = new_fc_sb.createVariable('airtemp', np.float64,('time','hru'), fill_value = -999.0)
pptrate = new_fc_sb.createVariable('pptrate', np.float64,('time','hru'), fill_value = -999.0)
spechum = new_fc_sb.createVariable('spechum', np.float64,('time','hru'), fill_value = -999.0)
windspd = new_fc_sb.createVariable('windspd', np.float64,('time','hru'), fill_value = -999.0)
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
for varname in new_fc_sb.variables.keys():
    var = new_fc_sb.variables[varname]
    #print (varname, var.dtype, var.dimensions, var.shape)
#%%
# define hru id, time step (1hr), lat and lon
# call the index you made in the other notebook
#idd = idxt
step = np.array([3600])
lat_sb = np.array(sbFD.variables['latitude'][:])
len_lat = np.repeat(lat_sb[:,np.newaxis], hru_num, axis=1); len_lat=len_lat.reshape(hru_num,)
long_sb =np.array(sbFD.variables['longitude'][:])
len_lon= np.repeat(long_sb[:,np.newaxis], hru_num, axis=1); len_lon=len_lon.reshape(hru_num,)
#%%
# assign newly created variables with lists of values from NLDAS and Sagehen data
hruid[:] = hruidxID 
lat[:] = len_lat
lon[:] = len_lon
ds[:] = step

new_ix = sbFD.variables['time'][:]
times[:] = new_ix

lwr_sb = sbFD.variables['LWRadAtm'][:]
lwr6_sb = np.repeat(lwr_sb[:,np.newaxis], hru_num, axis=1)
lwrad[:] = lwr6_sb

swr_sb = sbFD.variables['SWRadAtm'][:]
swr6_sb = np.repeat(swr_sb[:,np.newaxis], hru_num, axis=1)
swrad[:] = swr6_sb

ap_sb = sbFD.variables['airpres'][:]
ap6_sb = np.repeat(ap_sb[:,np.newaxis], hru_num, axis=1)
airpres[:] = ap6_sb

at_sb = sbFD.variables['airtemp'][:]
at6_sb = np.repeat(at_sb[:,np.newaxis], hru_num, axis=1) 
airtemp[:] = at6_sb

ppt_sb = sbFD.variables['pptrate'][:]
ppt6_sb = np.repeat(ppt_sb[:,np.newaxis], hru_num, axis=1) 
pptrate[:] = ppt6_sb

#specific humididty calculations***********************************************
at0_sb = sbFD.variables['airtemp'][:]
sh_sb = sbFD.variables['spechum'][:]

e_t = (ap_sb * sh_sb)/0.622
p_da = ap_sb - e_t
e_star_t = 611*(np.exp((17.27*(at0_sb-273.15))/(at0_sb-273.15+237.3)))
rh = e_t/e_star_t

e_star_t2 = 611*(np.exp((17.27*(at0_sb+2-273.15))/(at0_sb+2-273.15+237.3)))
e_star_t4 = 611*(np.exp((17.27*(at0_sb+4-273.15))/(at0_sb+4-273.15+237.3)))

e_t2 = rh * e_star_t2
e_t4 = rh * e_star_t4

p_t2 = p_da + e_t2
p_t4 = p_da + e_t4

sh_t2 = 0.622 * e_t2 / p_t2
sh_t4 = 0.622 * e_t4 / p_t4
#******************************************************************************
sh6_sb = np.repeat(sh_sb[:,np.newaxis], hru_num, axis=1) 
spechum[:] = sh6_sb

ws_sb = sbFD.variables['windspd'][:]
ws6_sb = np.repeat(ws_sb[:,np.newaxis], hru_num, axis=1) 
windspd[:] = ws6_sb

testsb = new_fc_sb.variables['airtemp'][:]
# close the file to write it
new_fc_sb.close()
#%%
# open it as read only and double check that everything looks good
testfd = Dataset("SenatorBeck_forcing_T0v.nc")
testfd.variables.keys()
print testfd.variables['spechum'][:]








