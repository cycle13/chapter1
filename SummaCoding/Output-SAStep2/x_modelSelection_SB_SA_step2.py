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
import itertools
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

date_swe = ['2010-12-14 08:00', '2011-01-06 08:00', '2011-01-30 08:00', '2011-02-28 08:00', '2011-03-10 08:00', '2011-04-04 08:00', '2011-05-04 08:00', '2011-05-04 08:30', '2011-06-21 08:00']
swe_mm = [120, 280, 385, 444, 537, 568, 836, 828, 503]  
#%% day of snow disappearance-observation data
for colname in observ_df:
    observ_df [colname] = pd.to_numeric(observ_df[colname])
    observ_df [colname].replace(-9999.0, np.nan, inplace=True)

hiRH =observation.variables['hiRH'][:]
loRH =observation.variables['loRH'][:]

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
maxSWEobs = Observ_df2010['snowDepth'].max()      
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
#%% output snowdepth dataframe
p1 = [200000, 360000, 1000000] #albedoDecayRate  
p2 = [0.40, 0.55, 0.90] #albedoMinSpring   
p3 = [0.75, 0.84, 0.91] #albedoMax      

p4 = [0.001,0.002,0.004] #z0Snow
p5 = [0.18,0.28,0.50] #windReductionParam
p6 = [0.35]#,0.25,0.50] #fixedThermalCond_snow

def hru_ix_ID(p1, p2, p3, p4, p5, p6):
    ix1 = np.arange(1,len(p1)+1)
    ix2 = np.arange(1,len(p2)+1)
    ix3 = np.arange(1,len(p3)+1)
    ix4 = np.arange(1,len(p4)+1)
    ix5 = np.arange(1,len(p5)+1)
    ix6 = np.arange(1,len(p6)+1)
    
    c = list(itertools.product(ix1,ix2,ix3,ix4,ix5,ix6))
    ix_numlist=[]
    for tup in c:
        ix_numlist.append(''.join(map(str, tup)))
    new_list = [float(i) for i in ix_numlist]

    return(new_list)  

hruidxID = hru_ix_ID(p1, p2, p3, p4, p5, p6)
hruidx_df = pd.DataFrame (hruidxID)
hruidx_dfAll = pd.concat([hruidx_df]*14, ignore_index=True)

hru_num = np.size(Ac_ASlouis_SWRclm_TCSjrdn.variables['hru'][:])
hru = ['hru{}'.format(i) for i in range(1, 3403)]

mod_swe_mm = output_sb_Av_ASl_SWRc_TCj.variables['scalarSWE'][:]

snowdepth_df = pd.concat([pd.DataFrame (Ac_ASlouis_SWRclm_TCSjrdn.variables['scalarSnowDepth'][:], columns=['hru{}'.format(i) for i in range(1, 244)]),
pd.DataFrame (Ac_ASlouis_SWRclm_TCSsmnv2.variables['scalarSnowDepth'][:], columns=['hru{}'.format(i) for i in range(244, 487)]),
pd.DataFrame (Ac_ASlouis_SWRueb_TCSjrdn.variables['scalarSnowDepth'][:], columns=['hru{}'.format(i) for i in range(487, 730)]),
pd.DataFrame (Ac_ASlouis_SWRueb_TCSsmnv2.variables['scalarSnowDepth'][:], columns=['hru{}'.format(i) for i in range(730, 973)]),
pd.DataFrame (Ac_ASstd_SWRclm_TCSjrdn.variables['scalarSnowDepth'][:], columns=['hru{}'.format(i) for i in range(973, 1216)]),
pd.DataFrame (Ac_ASstd_SWRclm_TCSsmnv2.variables['scalarSnowDepth'][:], columns=['hru{}'.format(i) for i in range(1216, 1459)]),
pd.DataFrame (Ac_ASstd_SWRueb_TCSjrdn.variables['scalarSnowDepth'][:], columns=['hru{}'.format(i) for i in range(1459, 1702)]),
pd.DataFrame (Ac_ASstd_SWRueb_TCSsmnv2.variables['scalarSnowDepth'][:], columns=['hru{}'.format(i) for i in range(1702, 1945)]),
pd.DataFrame (Av_ASlouis_SWRclm_TCSjrdn.variables['scalarSnowDepth'][:], columns=['hru{}'.format(i) for i in range(1945, 2188)]),
pd.DataFrame (Av_ASlouis_SWRclm_TCSsmnv2.variables['scalarSnowDepth'][:], columns=['hru{}'.format(i) for i in range(2188, 2431)]),
pd.DataFrame (Av_ASlouis_SWRueb_TCSjrdn.variables['scalarSnowDepth'][:], columns=['hru{}'.format(i) for i in range(2431, 2674)]),
pd.DataFrame (Av_ASlouis_SWRueb_TCSsmnv2.variables['scalarSnowDepth'][:], columns=['hru{}'.format(i) for i in range(2674, 2917)]),
pd.DataFrame (Av_ASstd_SWRclm_TCSjrdn.variables['scalarSnowDepth'][:], columns=['hru{}'.format(i) for i in range(2917, 3160)]),
pd.DataFrame (Av_ASstd_SWRclm_TCSsmnv2.variables['scalarSnowDepth'][:], columns=['hru{}'.format(i) for i in range(3160, 3403)])], axis=1)

#snowdepth_df = pd.DataFrame(snowdepth_df0, columns=[hru])
#%% output time step
TimeSb = Ac_ASlouis_SWRclm_TCSjrdn.variables['time'][:] # get values
t_unitSb = Ac_ASlouis_SWRclm_TCSjrdn.variables['time'].units # get unit  "days since 1950-01-01T00:00:00Z"

try :

    t_cal = Ac_ASlouis_SWRclm_TCSjrdn.variables['time'].calendar

except AttributeError : # Attribute doesn't exist

    t_cal = u"gregorian" # or standard

tvalueSb = num2date(TimeSb, units=t_unitSb, calendar=t_cal)
DateSb = [i.strftime("%Y-%m-%d %H:%M") for i in tvalueSb] # -%d %H:%M to display dates as string #i.strftime("%Y-%m-%d %H:%M")        
#%%
snowdepth_df.set_index(pd.DatetimeIndex(DateSb),inplace=True)

counter = pd.DataFrame(np.arange(0,np.size(snowdepth_df['hru1'])),columns=['counter'])
counter.set_index(snowdepth_df.index,inplace=True)
snowdepth_df2 = pd.concat([counter, snowdepth_df], axis=1)

#%%   day of snow disappearance-output ---- Sensititvity analysis - Step2
snowdepth_df6000 = snowdepth_df2[:][6000:6553]

zerosnowdate = []
for val in hru:
    zerosnowdate.append(np.where(snowdepth_df6000[val]==0)[0])
for i,item in enumerate(zerosnowdate):
    if len(item) == 0:
        zerosnowdate[i] = 553
for i,item in enumerate(zerosnowdate):
    zerosnowdate[i] = zerosnowdate[i]+6000
#
first_zerosnowdate =[]
for i,item in enumerate(zerosnowdate):
    if np.size(item)>1:
        #print np.size(item)
        first_zerosnowdate.append(item[0])
    if np.size(item)==1:
        first_zerosnowdate.append(item)

dif_obs_zerosnowdate = (first_zerosnowdate - dayofsnowdisappearance_obs)/24
#%% finding max snowdepth
max_snowdepth=[]
for hrus in hru:
    max_snowdepth.append(snowdepth_df2[hrus].max())
max_residualSWE = max_snowdepth - maxSWEobs
#%%
resSnowDisDate = pd.DataFrame(dif_obs_zerosnowdate, columns=['resSnowDisDate'])
resMaxSWE = pd.DataFrame(max_residualSWE, columns=['resMaxSWE'])
residual_df = pd.concat([resSnowDisDate, resMaxSWE], axis=1)

desiredHRUnum = residual_df.index[(abs(residual_df['resSnowDisDate']) <= 2) & (abs(residual_df['resMaxSWE'])<=0.443)].tolist()
desiredHRU = [y+1 for y in desiredHRUnum]
desiredHRUname = ['hru{}'.format(x) for x in desiredHRU]

desiredparams = []
for q in range (np.size(desiredHRUnum)):
    desiredparams.append(hruidx_dfAll[0][desiredHRUnum[q]])
#%%
plt.plot(max_residualSWE, dif_obs_zerosnowdate, 'go')
#plt.title('Residual maxSWE vs. snowDis', position=(0.04, 0.03), ha='left', fontsize=12)
plt.xlabel('Residual maxSWE')
plt.ylabel('Residual snowDis')
plt.savefig('Residual.png')

#%% r-squered and RMSE for snowdepth 
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










