###       /bin/bash runTestCases_docker.sh

#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt 
#from scipy.io import netcdf
from netCDF4 import Dataset,netcdftime,num2date
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import itertools
#%% # ***** How to get time value into Python DateTIme Objects *****
hru_num = 27

observation = Dataset("senatorBeck_SASP_1hr.nc")
#for i in observation.variables:
#    print i

time_obsrv = observation.variables['time'][:] # get values
t_unit_obs = observation.variables['time'].units # get unit  "days since 1950-01-01T00:00:00Z"

try :

    t_cal_obs = observation.variables['time'].calendar

except AttributeError : # Attribute doesn't exist

    t_cal_obs = u"gregorian" # or standard

tvalue_obs = num2date(time_obsrv, units=t_unit_obs, calendar=t_cal_obs)
date_obs = [i.strftime("%Y-%m-%d %H:%M") for i in tvalue_obs] # -%d %H:%M to display dates as string #i.strftime("%Y-%m-%d %H:%M")

#observ_df_int = np.column_stack([date_obs, observation['snowDepth'],observation['skinTemp'],observation['swRadUp'],observation['swRadDown'],observation['lwRadDown'],observation['hiRH']])
#observ_df = pd.DataFrame (observ_df_int, columns=['Date','snowDepth','skinTemp','swRadUp','swRadDown','lwRadDown','hiRH'])
#observ_df.set_index(pd.DatetimeIndex(date_obs),inplace=True)

observ_df_int = np.column_stack([observation['snowDepth'],observation['skinTemp'],observation['swRadUp'],observation['swRadDown'],observation['lwRadDown'],observation['hiRH']])
observ_df = pd.DataFrame (observ_df_int, columns=['snowDepth','skinTemp','swRadUp','swRadDown','lwRadDown','hiRH'])
observ_df.set_index(pd.DatetimeIndex(date_obs),inplace=True)
#%% day of snow disappearance-observation data
for colname in observ_df:
    observ_df [colname] = pd.to_numeric(observ_df[colname])
    observ_df [colname].replace(-9999.0, np.nan, inplace=True)

observ_df2010_init = observ_df['2010-10-01 00:00':'2011-09-30 00:00']
observ_df2010_init ['snowDepth'].replace(np.nan, 0, inplace=True)

counter = pd.DataFrame(np.arange(0,np.size(observ_df2010_init['snowDepth'])),columns=['counter'])
counter.set_index(observ_df2010_init.index,inplace=True)
Obsrv_df2010 = pd.concat([counter, observ_df2010_init], axis=1)

zero_snow_date = []
for c1 in range(np.size(Obsrv_df2010['snowDepth'])):
    if Obsrv_df2010['snowDepth'][c1]==0 and Obsrv_df2010['counter'][c1]>6000:
        zero_snow_date.append(Obsrv_df2010['counter'][c1])

dayofsnowdisappearance_obs = zero_snow_date [0]        
#%% Output
print '**********************************************************************' 

output_av = Dataset("albedo-albedoTest_2010-2011_senatorVariableDecayRate_1.nc")
output_ac = Dataset("albedo-albedoTest_2010-2011_senatorConstantDecayRate_1.nc")

for i in output_av.variables:
    print i
print np.size(output_ac.variables['time'][:])

#%% output time step
TimeSb = output_av.variables['time'][:] # get values
t_unitS = output_av.variables['time'].units # get unit  "days since 1950-01-01T00:00:00Z"

try :

    t_cal = output_av.variables['time'].calendar

except AttributeError : # Attribute doesn't exist

    t_cal = u"gregorian" # or standard

tvalueSb = num2date(TimeSb, units=t_unitS, calendar=t_cal)
DateSb = [i.strftime("%Y-%m-%d %H:%M") for i in tvalueSb] # -%d %H:%M to display dates as string #i.strftime("%Y-%m-%d %H:%M")
#%% day of snow disappearance-output #variable albedo decay rate
out1_df_int = np.column_stack([output_av['scalarSnowDepth'][:,0],output_av['scalarSurfaceTemp'][:,0],output_av['scalarSenHeatTotal'][:,0],output_av['scalarLatHeatTotal'][:,0]])
snowdepth_va = output_av.variables['scalarSnowDepth'][:]
hru = ['hru{}'.format(i) for i in range(1, hru_num+1)]
snowdepth_va_df = pd.DataFrame (snowdepth_va, columns=[hru])
snowdepth_va_df.set_index(pd.DatetimeIndex(DateSb),inplace=True)

counter = pd.DataFrame(np.arange(0,np.size(snowdepth_va_df['hru1'])),columns=['counter'])
counter.set_index(snowdepth_va_df.index,inplace=True)
snowdepth_va_df2 = pd.concat([counter, snowdepth_va_df], axis=1)
#
zerosnowdate_va = []
dayofsnowdisappearance_lsva =[]
for val in hru:
    for c1 in range(np.size(snowdepth_va_df2[val])):
        if snowdepth_va_df2[val][c1]==0 and snowdepth_va_df2['counter'][c1]>6000:
            zerosnowdate_va.append(snowdepth_va_df2['counter'][c1])
    dayofsnowdisappearance_lsva.append(zerosnowdate_va[0])
    zerosnowdate_va = []

num_day_dif_va = abs((dayofsnowdisappearance_lsva - dayofsnowdisappearance_obs)/24)
#%% day of snow disappearance-output #Costant albedo decay rate
out1_df_int_ca = np.column_stack([output_ac['scalarSnowDepth'][:,0],output_ac['scalarSurfaceTemp'][:,0],output_ac['scalarSenHeatTotal'][:,0],output_ac['scalarLatHeatTotal'][:,0]])
snowdepth_ca = output_ac.variables['scalarSnowDepth'][:]
hru = ['hru{}'.format(i) for i in range(1, hru_num+1)]

snowdepth_ca_df = pd.DataFrame (snowdepth_ca, columns=[hru])
snowdepth_ca_df.set_index(pd.DatetimeIndex(DateSb),inplace=True)
snowdepth_ca_df2 = pd.concat([counter, snowdepth_ca_df], axis=1)

zerosnowdate_ca = []
dayofsnowdisappearance_lsca =[]
for val in hru:
    for c1 in range(np.size(snowdepth_ca_df2[val])):
        if snowdepth_ca_df2[val][c1]==0 and snowdepth_ca_df2['counter'][c1]>6000:
            zerosnowdate_ca.append(snowdepth_ca_df2['counter'][c1])
    dayofsnowdisappearance_lsca.append(zerosnowdate_ca[0])
    zerosnowdate_ca = []

num_day_dif_ca = abs((dayofsnowdisappearance_lsca - dayofsnowdisappearance_obs)/24)

#%%
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
fig = plt.figure()
#plt.xticks(x, hru[::3], rotation=25)
plt.subplot(211)
plt.bar(x,num_day_dif_ca, color='purple')
plt.title('Date of snow disappariance difference from observation', fontsize=12)
plt.suptitle('Conctant decay rate', position=(0.88, 0.8), ha='right', fontsize=10)

plt.subplot(212)
plt.bar(x,num_day_dif_va, color='green')

plt.xlabel('HRUs')
plt.title('Variable decay rate', position=(0.06, 0.7), ha='left', fontsize=10)

#vax.yaxis.set_label_coords(0.5, -0.1) 
plt.savefig('Snowdisappariance.png')
#%% based on nc file
#sd_total_v = output_av.variables['scalarSnowDepth'][:]
#sd_total_c = output_ac.variables['scalarSnowDepth'][:]

#sd_V_1a = []
#for k in range (hru_num):
#    for j in range (8737): #np.size(time1)
#        sd_V_1a.append(sd_total_v[j][k])
#sd_psa_v = np.reshape(sd_V_1a, (27,8737)).T
#sd_p1 = sd_psa_v[:,0]    
#
#sd_C_1a = []
#for k in range (hru_num):
#    for j in range (8737): #np.size(time1)
#        sd_C_1a.append(sd_total_c[j][k])
#sd_psa_c = np.reshape(sd_C_1a, (27,8737)).T
#sd_p1_c = sd_psa[:,0]   
#%%
#param_nam_list = ['p1albedoMax', 'p1albedoMinWinter', 'p1albedoMinSpring', 'p1albedoMaxVisible', 'p1albedoMinVisible', 
#                  'p1albedoMaxNearIR', 'p1albedoMinNearIR', 'p1albedoSootLoad', 'p1albedoRefresh']
#color_list = ['darkred','maroon','mediumorchid','mediumpurple','mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','mistyrose','navy','olive','olivedrab','orange','orchid','paleturquoise','papayawhip','pink','purple','red','sandybrown','skyblue','tan', 'teal','thistle','violet', 'yellowgreen']
sd_obs2010 = Obsrv_df2010['snowDepth'][:]
#DateSb2 = [i.strftime("%Y-%m-%d") for i in DateSb]
sbx = np.arange(0,np.size(DateSb))
sb_xticks = DateSb
sbfig, sbax = plt.subplots(1,1)
plt.xticks(sbx, sb_xticks[::1000], rotation=25)
sbax.xaxis.set_major_locator(ticker.AutoLocator())

for hruname in hru:
       
    plt.subplot(211)
    plt.plot(sbx, snowdepth_va_df[hruname], sbx, sd_obs2010, 'k--')#, label='wwe', color='maroon') param_nam_list[q] color_list[q]
    plt.title('Variable decay rate', position=(0.05, 0.75), ha='left', fontsize=12)
    plt.xlabel('Time 2010-2011')
    plt.ylabel('Snow Depth (m)')
    
    plt.subplot(212)
    plt.plot(sbx, snowdepth_ca_df[hruname], sbx, sd_obs2010, 'k--')
    plt.title('Constant decay rate', position=(0.05, 0.75), ha='left', fontsize=12)
    plt.xlabel('Time 2010-2011')
    plt.ylabel('Snow Depth (m)')

#sb_xticks = DateSb
#sbfig, sbax = plt.subplots(1,1)
#plt.xticks(sbx, sb_xticks[::1000], rotation=25)
#sbax.xaxis.set_major_locator(ticker.AutoLocator())

#plt.show()
plt.savefig('SnowDepth_VA2.png')



















