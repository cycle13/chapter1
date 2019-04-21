###       /bin/bash runTestCases_docker.sh

#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt 
from netCDF4 import Dataset,netcdftime,num2date
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from sklearn.metrics import mean_squared_error

#%% # ***** How to get time value into Python DateTIme Objects *****
observation = Dataset("senatorBeck_FD.nc")
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

observ_df_int = np.column_stack([observation['snowDepth'],observation['skinTemp'],observation['swRadUp'],observation['swRadDown'],observation['lwRadDown'],observation['hiRH']])
observ_df = pd.DataFrame (observ_df_int, columns=['snowDepth','skinTemp','swRadUp','swRadDown','lwRadDown','hiRH'])
observ_df.set_index(pd.DatetimeIndex(date_obs),inplace=True)
#%% day of snow disappearance-observation data
for colname in observ_df:
    observ_df [colname] = pd.to_numeric(observ_df[colname])
    observ_df [colname].replace(-9999.0, np.nan, inplace=True)

observ_df2010_init = observ_df['2010-10-01 00:00':'2011-07-01 00:00']
observ_df2010_init ['snowDepth'].replace(np.nan, 0, inplace=True)
observ_df2010_init ['skinTemp'].replace(np.nan, 0, inplace=True)

counter = pd.DataFrame(np.arange(0,np.size(observ_df2010_init['snowDepth'])),columns=['counter'])
counter.set_index(observ_df2010_init.index,inplace=True)
Observ_df2010 = pd.concat([counter, observ_df2010_init], axis=1)

zero_snow_date = []
for c1 in range(np.size(Observ_df2010['snowDepth'])):
    if Observ_df2010['snowDepth'][c1]==0 and Observ_df2010['counter'][c1]>6000:
        zero_snow_date.append(Observ_df2010['counter'][c1])

dayofsnowdisappearance_obs = zero_snow_date [0]        
#%% Output ---- Sensititvity analysis - Step2
print '******************************************************************************************************************************' 

Ac_ASlouis_SWRclm_TCSjrdn = Dataset("Ac_ASlouis_SWRclm_TCSjrdn_2010-2011_senatorConstantDecayRate_1.nc")
Ac_ASlouis_SWRclm_TCSsmnv2 = Dataset("Ac_ASlouis_SWRclm_TCSsvp2_2010-2011_senatorConstantDecayRate_1.nc")
Ac_ASlouis_SWRueb_TCSjrdn = Dataset("Ac_ASlouis_SWRueb_TCSjrdn_2010-2011_senatorConstantDecayRate_1.nc")
Ac_ASlouis_SWRueb_TCSsmnv2 = Dataset("Ac_ASlouis_SWRueb_TCSsmnvp2_2010-2011_senatorConstantDecayRate_1.nc")
Ac_ASstd_SWRclm_TCSjrdn = Dataset("Ac_ASstd_SWRclm_TCSjrdn_2010-2011_senatorConstantDecayRate_1.nc")
Ac_ASstd_SWRclm_TCSsmnv2 = Dataset("Ac_ASstd_SWRclm_TCSsmnvp2_2010-2011_senatorConstantDecayRate_1.nc")
Ac_ASstd_SWRueb_TCSjrdn = Dataset("Ac_ASstd_SWRueb_TCSjrdn_2010-2011_senatorConstantDecayRate_1.nc")
Ac_ASstd_SWRueb_TCSsmnv2 = Dataset("Ac_ASstd_SWRueb_TCSsmnvp2_2010-2011_senatorConstantDecayRate_1.nc")
Av_ASlouis_SWRclm_TCSjrdn = Dataset("Av_ASlouis_SWRclm_TCSjrdn_2010-2011_senatorVariableDecayRate_1.nc")
Av_ASlouis_SWRclm_TCSsmnv2 = Dataset("Av_ASlouis_SWRclm_TCSsmnv_2010-2011_senatorVariableDecayRate_1.nc")
Av_ASlouis_SWRueb_TCSjrdn = Dataset("Av_ASlouis_SWRueb_TCSjrdn_2010-2011_senatorVariableDecayRate_1.nc")
Av_ASlouis_SWRueb_TCSsmnv2 = Dataset("Av_ASlouis_SWRueb_TCSsmnv_2010-2011_senatorVariableDecayRate_1.nc")
Av_ASstd_SWRclm_TCSjrdn = Dataset("Av_ASstd_SWRclm_TCSjrdn_2010-2011_senatorVariableDecayRate_1.nc")
Av_ASstd_SWRclm_TCSsmnv2 = Dataset("Av_ASstd_SWRclm_TCSsmnv_2010-2011_senatorVariableDecayRate_1.nc")

snowdepth_df = pd.concat([pd.DataFrame (Ac_ASlouis_SWRclm_TCSjrdn.variables['scalarSnowDepth'][:]),
pd.DataFrame (Ac_ASlouis_SWRclm_TCSsmnv2.variables['scalarSnowDepth'][:]),
pd.DataFrame (Ac_ASlouis_SWRueb_TCSjrdn.variables['scalarSnowDepth'][:]),
pd.DataFrame (Ac_ASlouis_SWRueb_TCSsmnv2.variables['scalarSnowDepth'][:]),
pd.DataFrame (Ac_ASstd_SWRclm_TCSjrdn.variables['scalarSnowDepth'][:]),
pd.DataFrame (Ac_ASstd_SWRclm_TCSsmnv2.variables['scalarSnowDepth'][:]),
pd.DataFrame (Ac_ASstd_SWRueb_TCSjrdn.variables['scalarSnowDepth'][:]),
pd.DataFrame (Ac_ASstd_SWRueb_TCSsmnv2.variables['scalarSnowDepth'][:]),
pd.DataFrame (Av_ASlouis_SWRclm_TCSjrdn.variables['scalarSnowDepth'][:]),
pd.DataFrame (Av_ASlouis_SWRclm_TCSsmnv2.variables['scalarSnowDepth'][:]),
pd.DataFrame (Av_ASlouis_SWRueb_TCSjrdn.variables['scalarSnowDepth'][:]),
pd.DataFrame (Av_ASlouis_SWRueb_TCSsmnv2.variables['scalarSnowDepth'][:]),
pd.DataFrame (Av_ASstd_SWRclm_TCSjrdn.variables['scalarSnowDepth'][:]),
pd.DataFrame (Av_ASstd_SWRclm_TCSsmnv2.variables['scalarSnowDepth'][:])], axis=1)
#pd.concat([counter, snowdepth_Av_ASl_SWRc_TCj_df], axis=1)

hru_num = np.size(Ac_ASlouis_SWRclm_TCSjrdn.variables['hru'][:])
hru = ['hru{}'.format(i) for i in range(1, 14*(hru_num+1))]

#%% output time step
TimeSb = Ac_ASlouis_SWRclm_TCSjrdn.variables['time'][:] # get values
t_unitSb = Ac_ASlouis_SWRclm_TCSjrdn.variables['time'].units # get unit  "days since 1950-01-01T00:00:00Z"

try :

    t_cal = Ac_ASlouis_SWRclm_TCSjrdn.variables['time'].calendar

except AttributeError : # Attribute doesn't exist

    t_cal = u"gregorian" # or standard

tvalueSb = num2date(TimeSb, units=t_unitSb, calendar=t_cal)
DateSb = [i.strftime("%Y-%m-%d %H:%M") for i in tvalueSb] # -%d %H:%M to display dates as string #i.strftime("%Y-%m-%d %H:%M")
#%% output snowdepth dataframe
#output_list = [Ac_ASlouis_SWRclm_TCSjrdn,Ac_ASlouis_SWRclm_TCSsmnv2,Ac_ASlouis_SWRueb_TCSjrdn,Ac_ASlouis_SWRueb_TCSsmnv2,
#               Ac_ASstd_SWRclm_TCSjrdn,Ac_ASstd_SWRclm_TCSsmnv2,Ac_ASstd_SWRueb_TCSjrdn,Ac_ASstd_SWRueb_TCSsmnv2,
#               Av_ASlouis_SWRclm_TCSjrdn,Av_ASlouis_SWRclm_TCSsmnv2,Av_ASlouis_SWRueb_TCSjrdn,Av_ASlouis_SWRueb_TCSsmnv2,
#               Av_ASstd_SWRclm_TCSjrdn, Av_ASstd_SWRclm_TCSsmnv2]
#snowDepth_ls =[]
#for output in output_list:
#    snowDepth_ls.append(output['scalarSnowDepth'][:])
#  
#
#snowdepth_df =[]
#for mld in range(0,14):
#    for hrus in range(0,hru_num+1):
#        snowdepth_df.append(pd.DataFrame(snowDepth_ls[mld][hrus]))#, columns=[hru])
#        
#%%
snowdepth_Av_ASl_SWRc_TCj = Ac_ASstd_SWRclm_TCSsmnv2.variables['scalarSnowDepth'][:]
hru = ['hru{}'.format(i) for i in range(1, hru_num+1)]
snowdepth_Av_ASl_SWRc_TCj_df = pd.DataFrame (snowdepth_Av_ASl_SWRc_TCj, columns=[hru])
snowdepth_Av_ASl_SWRc_TCj_df.set_index(pd.DatetimeIndex(DateSb),inplace=True)

counter = pd.DataFrame(np.arange(0,np.size(snowdepth_Av_ASl_SWRc_TCj_df['hru1'])),columns=['counter'])
counter.set_index(snowdepth_Av_ASl_SWRc_TCj_df.index,inplace=True)
snowdepth_Av_ASl_SWRc_TCj_df2 = pd.concat([counter, snowdepth_Av_ASl_SWRc_TCj_df], axis=1)

#%%   day of snow disappearance-output ---- Sensititvity analysis - Step2
snowdepth_Av_ASl_SWRc_TCj_df6000 = snowdepth_Av_ASl_SWRc_TCj_df2[:][6000:6553]

zerosnowdate = []
for val in hru:
    zerosnowdate.append(np.where(snowdepth_Av_ASl_SWRc_TCj_df6000[val]==0)[0])
for i,item in enumerate(zerosnowdate):
    if len(item) == 0:
        zerosnowdate[i] = 553
for i,item in enumerate(zerosnowdate):
    zerosnowdate[i] = zerosnowdate[i]+6000

taip_ls = type(zerosnowdate[0])
taip_int = type(zerosnowdate[37])

first_zerosnowdate =[]
for i,item in enumerate(zerosnowdate):
    if np.size(item)>1:
        #print np.size(item)
        first_zerosnowdate.append(item[0])
    if np.size(item)==1:
        first_zerosnowdate.append(item)

dif_obs_zerosnowdate = abs((first_zerosnowdate - dayofsnowdisappearance_obs)/24)

#%%
#x = list(np.arange(1,hru_num+1))
#fig = plt.figure()
##plt.xticks(x, hru[::3], rotation=25)
#plt.bar(x,dif_obs_zerosnowdate, color='navy')
#plt.title('Date of snow disappariance difference from observation', fontsize=12)
#plt.xlabel('HRUs')
#
##vax.yaxis.set_label_coords(0.5, -0.1) Av_ASlouis_SWRueb_TCSsmnv
#plt.savefig('Snowdisappariance_Av_ASl_SWRu_TCs.png')
#
##%%
#param_nam_list = ['p1albedoMax', 'p1albedoMinWinter', 'p1albedoMinSpring', 'p1albedoMaxVisible', 'p1albedoMinVisible', 
#                  'p1albedoMaxNearIR', 'p1albedoMinNearIR', 'p1albedoSootLoad', 'p1albedoRefresh']
#color_list = ['darkred','maroon','mediumorchid','mediumpurple','mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','mistyrose','navy','olive','olivedrab','orange','orchid','paleturquoise','papayawhip','pink','purple','red','sandybrown','skyblue','tan', 'teal','thistle','violet', 'yellowgreen']
#tvalueSb = num2date(TimeSb, units=t_unitSb, calendar=t_cal)
#DateSb2 = [i.strftime("%Y-%m") for i in tvalueSb]
#
#sd_obs2010 = Observ_df2010['snowDepth'][:]
#sbx = np.arange(0,np.size(DateSb2))
#sb_xticks = DateSb2
#sbfig, sbax = plt.subplots(1,1)
#plt.xticks(sbx, sb_xticks[::1000], rotation=25)
#sbax.xaxis.set_major_locator(ticker.AutoLocator())
#
#for hruname in hru:
#    plt.plot(sbx, snowdepth_Av_ASl_SWRc_TCj_df[hruname], sbx, sd_obs2010, 'k--', linewidth=0.5)#, label='wwe', color='maroon') param_nam_list[q] color_list[q]
#    plt.title('Av_ASlouis_SWRueb_TCSsmnv2', position=(0.04, 0.88), ha='left', fontsize=12)
#    plt.xlabel('Time 2010-2011')
#    plt.ylabel('Snow Depth (m)')
##plt.show()
#plt.savefig('SnowDepth_Av_ASl_SWRu_TCs.png')
##%% r-squered and RMSE for snowdepth 
#r2sd_Av_ASl_SWRc_TCj =[]
#for item in hru:
#    slope, intercept, r_value, p_value, std_err = stats.linregress(sd_obs2010,snowdepth_Av_ASl_SWRc_TCj_df2[item])
#    r2sd_Av_ASl_SWRc_TCj.append(np.power(r_value,2))
#
#rmsesd_Av_ASl_SWRc_TCj =[]
#for item in hru:
#    rmsesd_Av_ASl_SWRc_TCj.append(np.power(mean_squared_error(sd_obs2010,snowdepth_Av_ASl_SWRc_TCj_df2[item]),0.5))
#
#plt.plot(rmsesd_Av_ASl_SWRc_TCj, r2sd_Av_ASl_SWRc_TCj, 'ro')
#plt.title('snowDepth_Av_ASlouis_SWRueb_TCSsmnv2', position=(0.04, 0.03), ha='left', fontsize=12)
#plt.xlabel('RMSE')
#plt.ylabel('R2')
#plt.savefig('RMSE_R2_SnowDepth_Av_ASl_SWRu_TCs.png')
#plt.hold()
##%% skinTemp dataframe
#skinTemp_Av_ASl_SWRc_TCj = output_sb_Av_ASl_SWRc_TCj.variables['scalarSurfaceTemp'][:]
#skinTemp_Av_ASl_SWRc_TCj_df = pd.DataFrame (skinTemp_Av_ASl_SWRc_TCj, columns=[hru])
#skinTemp_Av_ASl_SWRc_TCj_df.set_index(pd.DatetimeIndex(DateSb),inplace=True)
#skinTemp_Av_ASl_SWRc_TCj_df2 = skinTemp_Av_ASl_SWRc_TCj_df-273.15
##skinTemp_Ac_ASl_SWRc_TCj_df2 = pd.concat([counter, skinTemp_Ac_ASl_SWRc_TCj_df], axis=1)-273.15
#
#skintemp2010 = Observ_df2010['skinTemp']
#                            
#r2st_Av_ASl_SWRc_TCj =[]
#for item in hru:
#    slope, intercept, r_value, p_value, std_err = stats.linregress(skintemp2010,skinTemp_Av_ASl_SWRc_TCj_df2[item])
#    r2st_Av_ASl_SWRc_TCj.append(np.power(r_value,2))
#
#rmsest_Av_ASl_SWRc_TCj =[]
#for item in hru:
#    rmsest_Av_ASl_SWRc_TCj.append(np.power(mean_squared_error(skintemp2010,skinTemp_Av_ASl_SWRc_TCj_df2[item]),0.5))
#
#plt.plot(rmsest_Av_ASl_SWRc_TCj, r2st_Av_ASl_SWRc_TCj, 'bo')
#plt.title('skinTemp_Av_ASlouis_SWRueb_TCSsmnv2', position=(0.04, 0.03), ha='left', fontsize=12)
#plt.xlabel('RMSE')
#plt.ylabel('R2')
#plt.savefig('RMSE_R2_skinTemp_Av_ASl_SWRu_TCs.png')
#plt.hold(True)










