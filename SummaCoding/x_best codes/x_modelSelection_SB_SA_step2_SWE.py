###       /bin/bash runTestCases_docker.sh
import numpy as np
import matplotlib.pyplot as plt 
from netCDF4 import Dataset,num2date
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from sklearn.metrics import mean_squared_error
import itertools
#%% # ***** How to get time value into Python DateTIme Objects *****
forcingData = Dataset("0SenatorBeck_forcing0.nc")
validation = Dataset("validation_senatorBeck_SASP_1hr.nc")
#for i in validation.variables:
#    print i

specific_humidity =forcingData.variables['spechum'][:]

time_obsrv = validation.variables['time'][:] # get values
t_unit_obs = validation.variables['time'].units # get unit  "days since 1950-01-01T00:00:00Z"

try :

    t_cal_obs = validation.variables['time'].calendar

except AttributeError : # Attribute doesn't exist

    t_cal_obs = u"gregorian" # or standard

tvalue_obs = num2date(time_obsrv, units=t_unit_obs, calendar=t_cal_obs)
date_obs = [i.strftime("%Y-%m-%d %H:%M") for i in tvalue_obs] # -%d %H:%M to display dates as string #i.strftime("%Y-%m-%d %H:%M")

observ_df_int = np.column_stack([validation['skinTemp'],validation['snowDepth'],validation['swRadUp'],validation['swRadDown'],validation['lwRadDown'],validation['nirRadDown'], validation['nirRadUp'],validation['hiRH'], validation['loRH']])
observ_df = pd.DataFrame (observ_df_int, columns=['skinTemp','snowDepth','swRadUp','swRadDown','lwRadDown','nirRadDown','nirRadUp','hiRH', 'loRH'])
observ_df.set_index(pd.DatetimeIndex(date_obs),inplace=True)

date_swe = ['2010-12-14 08:00', '2011-01-06 08:00', '2011-01-30 08:00', '2011-02-28 08:00', '2011-03-10 08:00', '2011-04-04 08:00', '2011-05-04 08:00', '2011-05-04 08:30', '2011-06-21 08:00']
swe_mm = [120, 280, 385, 444, 537, 568, 836, 828, 503]  

#obs_swe_date = pd.DataFrame (np.column_stack([date_swe,swe_mm]), columns=['date_swe','swe_mm'])
obs_swe = pd.DataFrame (swe_mm, columns=['swe_mm'])
obs_swe.set_index(pd.DatetimeIndex(date_swe),inplace=True)
#%% day of snow disappearance-observation/validation data
for colname in observ_df:
    observ_df [colname] = pd.to_numeric(observ_df[colname])
    observ_df [colname].replace(-9999.0, np.nan, inplace=True)

observ_df2010_init = observ_df['2010-10-01 00:00':'2011-07-01 00:00']
observ_df2010_init ['snowDepth'].replace(np.nan, 0, inplace=True)
observ_df2010_init ['skinTemp'].replace(np.nan, 0, inplace=True)

counter = pd.DataFrame(np.arange(0,np.size(observ_df2010_init['snowDepth'])),columns=['counter'])
counter.set_index(observ_df2010_init.index,inplace=True)
observ_df2010 = pd.concat([counter, observ_df2010_init], axis=1)

zero_snow_date = []
for c1 in range(np.size(observ_df2010['snowDepth'])):
    if observ_df2010['snowDepth'][c1]==0 and observ_df2010['counter'][c1]>6000:
        zero_snow_date.append(observ_df2010['counter'][c1])

dayofsnowdisappearance_obs = zero_snow_date [0]  
max_swe_obs = max(obs_swe['swe_mm'])
#max_swe_date = np.where (obs_swe['swe_mm'] == max_swe_obs)
max_swe_date = obs_swe[obs_swe ['swe_mm']== max_swe_obs].index.tolist()    
#%% Output ---- Sensititvity analysis - Step2
print '******************************************************************************************************************************' 

Ac_ASlouis_SWRclm_TCSjrdn = Dataset("C:/1UNR-University Folder/Dissertation/Chapter 1-Snowmelt/SummaCoding/Output-SAStep2/Ac_ASlouis_SWRclm_TCSjrdn_2010-2011_senatorConstantDecayRate_1.nc")
Ac_ASlouis_SWRclm_TCSsmnv = Dataset("C:/1UNR-University Folder/Dissertation/Chapter 1-Snowmelt/SummaCoding/Output-SAStep2/Ac_ASlouis_SWRclm_TCSsvp2_2010-2011_senatorConstantDecayRate_1.nc")
Ac_ASlouis_SWRueb_TCSjrdn = Dataset("C:/1UNR-University Folder/Dissertation/Chapter 1-Snowmelt/SummaCoding/Output-SAStep2/Ac_ASlouis_SWRueb_TCSjrdn_2010-2011_senatorConstantDecayRate_1.nc")
Ac_ASlouis_SWRueb_TCSsmnv = Dataset("C:/1UNR-University Folder/Dissertation/Chapter 1-Snowmelt/SummaCoding/Output-SAStep2/Ac_ASlouis_SWRueb_TCSsmnvp2_2010-2011_senatorConstantDecayRate_1.nc")
Ac_ASstd_SWRclm_TCSjrdn = Dataset("C:/1UNR-University Folder/Dissertation/Chapter 1-Snowmelt/SummaCoding/Output-SAStep2/Ac_ASstd_SWRclm_TCSjrdn_2010-2011_senatorConstantDecayRate_1.nc")
Ac_ASstd_SWRclm_TCSsmnv = Dataset("C:/1UNR-University Folder/Dissertation/Chapter 1-Snowmelt/SummaCoding/Output-SAStep2/Ac_ASstd_SWRclm_TCSsmnvp2_2010-2011_senatorConstantDecayRate_1.nc")
Ac_ASstd_SWRueb_TCSjrdn = Dataset("C:/1UNR-University Folder/Dissertation/Chapter 1-Snowmelt/SummaCoding/Output-SAStep2/Ac_ASstd_SWRueb_TCSjrdn_2010-2011_senatorConstantDecayRate_1.nc")
Ac_ASstd_SWRueb_TCSsmnv = Dataset("C:/1UNR-University Folder/Dissertation/Chapter 1-Snowmelt/SummaCoding/Output-SAStep2/Ac_ASstd_SWRueb_TCSsmnvp2_2010-2011_senatorConstantDecayRate_1.nc")
Av_ASlouis_SWRclm_TCSjrdn = Dataset("C:/1UNR-University Folder/Dissertation/Chapter 1-Snowmelt/SummaCoding/Output-SAStep2/Av_ASlouis_SWRclm_TCSjrdn_2010-2011_senatorVariableDecayRate_1.nc")
Av_ASlouis_SWRclm_TCSsmnv = Dataset("C:/1UNR-University Folder/Dissertation/Chapter 1-Snowmelt/SummaCoding/Output-SAStep2/Av_ASlouis_SWRclm_TCSsmnv_2010-2011_senatorVariableDecayRate_1.nc")
Av_ASlouis_SWRueb_TCSjrdn = Dataset("C:/1UNR-University Folder/Dissertation/Chapter 1-Snowmelt/SummaCoding/Output-SAStep2/Av_ASlouis_SWRueb_TCSjrdn_2010-2011_senatorVariableDecayRate_1.nc")
Av_ASlouis_SWRueb_TCSsmnv = Dataset("C:/1UNR-University Folder/Dissertation/Chapter 1-Snowmelt/SummaCoding/Output-SAStep2/Av_ASlouis_SWRueb_TCSsmnv_2010-2011_senatorVariableDecayRate_1.nc")
Av_ASstd_SWRclm_TCSjrdn = Dataset("C:/1UNR-University Folder/Dissertation/Chapter 1-Snowmelt/SummaCoding/Output-SAStep2/Av_ASstd_SWRclm_TCSjrdn_2010-2011_senatorVariableDecayRate_1.nc")
Av_ASstd_SWRclm_TCSsmnv = Dataset("C:/1UNR-University Folder/Dissertation/Chapter 1-Snowmelt/SummaCoding/Output-SAStep2/Av_ASstd_SWRclm_TCSsmnv_2010-2011_senatorVariableDecayRate_1.nc")

all_out = [Ac_ASlouis_SWRclm_TCSjrdn,Ac_ASlouis_SWRclm_TCSsmnv,Ac_ASlouis_SWRueb_TCSjrdn,Ac_ASlouis_SWRueb_TCSsmnv,
                Ac_ASstd_SWRclm_TCSjrdn,Ac_ASstd_SWRclm_TCSsmnv,Ac_ASstd_SWRueb_TCSjrdn,Ac_ASstd_SWRueb_TCSsmnv,
                Av_ASlouis_SWRclm_TCSjrdn,Av_ASlouis_SWRclm_TCSsmnv,Av_ASlouis_SWRueb_TCSjrdn,Av_ASlouis_SWRueb_TCSsmnv,
                Av_ASstd_SWRclm_TCSjrdn,Av_ASstd_SWRclm_TCSsmnv]
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

out_names = ['AcASlSWcTCj','AcASlSWcTCs','AcASlSWuTCj','AcASlSWuTCs','AcASsSWcTCj','AcASsSWcTCs','AcASsSWuTCj','AcASsSWuTCs',
             'AvASlSWcTCj','AvASlSWcTCs','AvASlSWuTCj','AvASlSWuTCs','AvASsSWcTCj','AvASsSWcTCs']
hru_num = np.size(hruidxID)
#hru = ['hru{}'.format(i) for i in hruidxID]
hru_names =[]
for i in out_names:
    hru_names.append(['{}{}'.format(i, j) for j in hruidxID])

hru_names1 = np.reshape(hru_names,(3402,1))
hru_names_df = pd.DataFrame (hru_names1)
#%% output time step
TimeSb = Ac_ASlouis_SWRclm_TCSjrdn.variables['time'][:] # get values
t_unitSb = Ac_ASlouis_SWRclm_TCSjrdn.variables['time'].units # get unit  "days since 1950-01-01T00:00:00Z"

try :

    t_cal = Ac_ASlouis_SWRclm_TCSjrdn.variables['time'].calendar

except AttributeError : # Attribute doesn't exist

    t_cal = u"gregorian" # or standard

tvalueSb = num2date(TimeSb, units=t_unitSb, calendar=t_cal)
DateSb = [i.strftime("%Y-%m-%d %H:%M") for i in tvalueSb] # -%d %H:%M to display dates as string #i.strftime("%Y-%m-%d %H:%M")   
#%% modeled snowdepth
all_SD = []
for p in range (np.size(all_out)):
    all_SD.append(pd.DataFrame(all_out[p].variables['scalarSnowDepth'][:], columns=[hru_names[p]]))

all_sd_df = pd.concat(all_SD, axis=1)
all_sd_df.set_index(pd.DatetimeIndex(DateSb),inplace=True)

counter = pd.DataFrame(np.arange(0,np.size(DateSb)),columns=['counter'])
counter.set_index(all_sd_df.index,inplace=True)
all_sd_df2 = pd.concat([counter, all_sd_df], axis=1)  
#%% modeled swe
all_SWE = []
for p in range (np.size(all_out)):
    all_SWE.append(pd.DataFrame(all_out[p].variables['scalarSWE'][:], columns=[hru_names[p]]))

all_swe_df = pd.concat(all_SWE, axis=1)   
all_swe_df.set_index(pd.DatetimeIndex(DateSb),inplace=True)

counter = pd.DataFrame(np.arange(0,np.size(DateSb)),columns=['counter'])
counter.set_index(all_swe_df.index,inplace=True)
all_swe_df2 = pd.concat([counter, all_swe_df], axis=1)  
#%%   day of snow disappearance-output ---- Sensititvity analysis - Step2
all_sd_df6000 = all_sd_df2[:][6000:6553]

zerosnowdate = []
for val in hru_names_df [0]:
    zerosnowdate.append(np.where(all_sd_df6000[val]==0)[0])
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
#%% finding max snowdepth and swe
max_swe=[]
for hrus in hru_names_df[0]:
    max_swe.append(all_swe_df2[hrus].max())
max_residual_SWE = max_swe - max_swe_obs

swe_corsp_max2date = []
for hrus in hru_names_df[0]:
    swe_corsp_max2date.append(all_swe_df2[hrus][max_swe_date])
max_residual_swe_corsp = swe_corsp_max2date - max_swe_obs

#%%
plt.plot(max_residual_swe_corsp, dif_obs_zerosnowdate, 'go')
#plt.title('Residual maxSWE vs. snowDis', position=(0.04, 0.03), ha='left', fontsize=12)
plt.xlabel('Residual maxSWE')
plt.ylabel('Residual snowDis')
plt.savefig('Residual.png')
#%%
resSnowDisDate = pd.DataFrame(dif_obs_zerosnowdate, columns=['resSnowDisDate'])
resCorspMaxSWE = pd.DataFrame(max_residual_swe_corsp, columns=['resCorspMaxSWE'])
residual_df = pd.concat([resSnowDisDate, resCorspMaxSWE], axis=1)
residual_df.set_index(hru_names_df[0],inplace=True)
pareto_model_param = pd.DataFrame(residual_df.index[(abs(residual_df['resSnowDisDate']) <= 2) & (abs(residual_df['resCorspMaxSWE'])<=123)].tolist())

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










