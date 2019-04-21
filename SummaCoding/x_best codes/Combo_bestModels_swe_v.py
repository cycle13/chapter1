###       /bin/bash runTestCases_docker.sh
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
#%%
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

#%% output snowdepth dataframe
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

out_names = ['AvASlSWcTCj','AvASlSWcTCs','AvASlSWuTCj','AvASlSWuTCs','AvASsSWcTCj','AvASsSWcTCs']
hru_num = np.size(hruidxID)
#hru = ['hru{}'.format(i) for i in hruidxID]
hru_names =[]
for i in out_names:
    hru_names.append(['{}{}'.format(i, j) for j in hruidxID])

hru_names1 = np.reshape(hru_names,(18,1))
hru_names_df = pd.DataFrame (hru_names1)
#%%
TConst_louis_clm_jrdn = Dataset("TConst_louis_clm_jrdn_2010-2011_senatorConstantDecayRate_1.nc")
T2Const_louis_clm_jrdn = Dataset("T2Const_louis_clm_jrdn_2010-2011_senatorConstantDecayRate_1.nc")
T4Const_louis_clm_jrdn = Dataset("T4Const_louis_clm_jrdn_2010-2011_senatorConstantDecayRate_1.nc")

#for varname in TConst_louis_clm_jrdn.variables.keys():
#    var = TConst_louis_clm_jrdn.variables[varname]
#    print (varname, var.dtype, var.dimensions, var.shape)
    
TConst_louis_clm_smnv = Dataset("TConst_louis_clm_smnv_2010-2011_senatorConstantDecayRate_1.nc")
T2Const_louis_clm_smnv = Dataset("T2Const_louis_clm_smnv_2010-2011_senatorConstantDecayRate_1.nc")
T4Const_louis_clm_smnv = Dataset("T4Const_louis_clm_smnv_2010-2011_senatorConstantDecayRate_1.nc")

TConst_louis_eub_jrdn = Dataset("TConst_louis_eub_jrdn_2010-2011_senatorConstantDecayRate_1.nc")
T2Const_louis_eub_jrdn = Dataset("T2Const_louis_eub_jrdn_2010-2011_senatorConstantDecayRate_1.nc")
T4Const_louis_eub_jrdn = Dataset("T4Const_louis_eub_jrdn_2010-2011_senatorConstantDecayRate_1.nc")

TConst_louis_eub_smnv = Dataset("TConst_louis_eub_smnv_2010-2011_senatorConstantDecayRate_1.nc")
T2Const_louis_eub_smnv = Dataset("T2Const_louis_eub_smnv_2010-2011_senatorConstantDecayRate_1.nc")
T4Const_louis_eub_smnv = Dataset("T4Const_louis_eub_smnv_2010-2011_senatorConstantDecayRate_1.nc")

TConst_std_clm_jrdn = Dataset("TConst_std_clm_jrdn_2010-2011_senatorConstantDecayRate_1.nc")
T2Const_std_clm_jrdn = Dataset("T2Const_std_clm_jrdn_2010-2011_senatorConstantDecayRate_1.nc")
T4Const_std_clm_jrdn = Dataset("T4Const_std_clm_jrdn_2010-2011_senatorConstantDecayRate_1.nc")

TConst_std_clm_smnv = Dataset("TConst_std_clm_smnv_2010-2011_senatorConstantDecayRate_1.nc")
T2Const_std_clm_smnv = Dataset("T2Const_std_clm_smnv_2010-2011_senatorConstantDecayRate_1.nc")
T4Const_std_clm_smnv = Dataset("T4Const_std_clm_smnv_2010-2011_senatorConstantDecayRate_1.nc")

TConst_std_eub_jrdn = Dataset("TConst_std_eub_jrdn_2010-2011_senatorConstantDecayRate_1.nc")
T2Const_std_eub_jrdn = Dataset("T2Const_std_eub_jrdn_2010-2011_senatorConstantDecayRate_1.nc")
T4Const_std_eub_jrdn = Dataset("T4Const_std_eub_jrdn_2010-2011_senatorConstantDecayRate_1.nc")

TConst_std_eub_smnv = Dataset("TConst_std_eub_smnv_2010-2011_senatorConstantDecayRate_1.nc")
T2Const_std_eub_smnv = Dataset("T2Const_std_eub_smnv_2010-2011_senatorConstantDecayRate_1.nc")
T4Const_std_eub_smnv = Dataset("T4Const_std_eub_smnv_2010-2011_senatorConstantDecayRate_1.nc")

hru_num = np.size(T2Const_louis_clm_jrdn.variables['hru'][:])

hruT0lcj = ['T0lcjHRU{}'.format(i) for i in hruidxID]
hruT0lcs = ['T0lcsHRU{}'.format(i) for i in hruidxID]
hruT0lej = ['T0lejHRU{}'.format(i) for i in hruidxID]
hruT0les = ['T0lesHRU{}'.format(i) for i in hruidxID]
hruT0scj = ['T0scjHRU{}'.format(i) for i in hruidxID]
hruT0scs = ['T0scsHRU{}'.format(i) for i in hruidxID]
hruT0sej = ['T0sejHRU{}'.format(i) for i in hruidxID]
hruT0ses = ['T0sesHRU{}'.format(i) for i in hruidxID]

hruT2lcj = ['T2lcjHRU{}'.format(i) for i in hruidxID]
hruT2lcs = ['T2lcsHRU{}'.format(i) for i in hruidxID]
hruT2lej = ['T2lejHRU{}'.format(i) for i in hruidxID]
hruT2les = ['T2lesHRU{}'.format(i) for i in hruidxID]
hruT2scj = ['T2scjHRU{}'.format(i) for i in hruidxID]
hruT2scs = ['T2scsHRU{}'.format(i) for i in hruidxID]
hruT2sej = ['T2sejHRU{}'.format(i) for i in hruidxID]
hruT2ses = ['T2sesHRU{}'.format(i) for i in hruidxID]

hruT4lcj = ['T4lcjHRU{}'.format(i) for i in hruidxID]
hruT4lcs = ['T4lcsHRU{}'.format(i) for i in hruidxID]
hruT4lej = ['T4lejHRU{}'.format(i) for i in hruidxID]
hruT4les = ['T4lesHRU{}'.format(i) for i in hruidxID]
hruT4scj = ['T4scjHRU{}'.format(i) for i in hruidxID]
hruT4scs = ['T4scsHRU{}'.format(i) for i in hruidxID]
hruT4sej = ['T4sejHRU{}'.format(i) for i in hruidxID]
hruT4ses = ['T4sesHRU{}'.format(i) for i in hruidxID]

hru_mod = pd.concat([pd.DataFrame(hruT0lcj),pd.DataFrame(hruT0lcs),pd.DataFrame (hruT0lej),pd.DataFrame (hruT0les),pd.DataFrame (hruT0scj),
pd.DataFrame (hruT0scs),pd.DataFrame (hruT0sej),pd.DataFrame (hruT0ses),pd.DataFrame (hruT2lcj),pd.DataFrame (hruT2lcs),pd.DataFrame (hruT2lej),
pd.DataFrame (hruT2les),pd.DataFrame (hruT2scj),pd.DataFrame (hruT2scs),pd.DataFrame (hruT2sej),pd.DataFrame (hruT2ses),pd.DataFrame (hruT4lcj),
pd.DataFrame (hruT4lcs),pd.DataFrame (hruT4lej),pd.DataFrame (hruT4les),pd.DataFrame (hruT4scj),pd.DataFrame (hruT4scs),pd.DataFrame (hruT4sej),
pd.DataFrame (hruT4ses)])#, axis=1)
#%%
snowdepth_df = pd.concat([pd.DataFrame (TConst_louis_clm_jrdn.variables['scalarSnowDepth'][:], columns=[hruT0lcj]),
pd.DataFrame (TConst_louis_clm_smnv.variables['scalarSnowDepth'][:], columns=[hruT0lcs]),
pd.DataFrame (TConst_louis_eub_jrdn.variables['scalarSnowDepth'][:], columns=[hruT0lej]),
pd.DataFrame (TConst_louis_eub_smnv.variables['scalarSnowDepth'][:], columns=[hruT0les]),
pd.DataFrame (TConst_std_clm_jrdn.variables['scalarSnowDepth'][:], columns=[hruT0scj]),
pd.DataFrame (TConst_std_clm_smnv.variables['scalarSnowDepth'][:], columns=[hruT0scs]),
pd.DataFrame (TConst_std_eub_jrdn.variables['scalarSnowDepth'][:], columns=[hruT0sej]),
pd.DataFrame (TConst_std_eub_smnv.variables['scalarSnowDepth'][:], columns=[hruT0ses]),

pd.DataFrame (T2Const_louis_clm_jrdn.variables['scalarSnowDepth'][:], columns=[hruT2lcj]),
pd.DataFrame (T2Const_louis_clm_smnv.variables['scalarSnowDepth'][:], columns=[hruT2lcs]),
pd.DataFrame (T2Const_louis_eub_jrdn.variables['scalarSnowDepth'][:], columns=[hruT2lej]),
pd.DataFrame (T2Const_louis_eub_smnv.variables['scalarSnowDepth'][:], columns=[hruT2les]),
pd.DataFrame (T2Const_std_clm_jrdn.variables['scalarSnowDepth'][:], columns=[hruT2scj]),
pd.DataFrame (T2Const_std_clm_smnv.variables['scalarSnowDepth'][:], columns=[hruT2scs]),
pd.DataFrame (T2Const_std_eub_jrdn.variables['scalarSnowDepth'][:], columns=[hruT2sej]),
pd.DataFrame (T2Const_std_eub_smnv.variables['scalarSnowDepth'][:], columns=[hruT2ses]),

pd.DataFrame (T4Const_louis_clm_jrdn.variables['scalarSnowDepth'][:], columns=[hruT4lcj]),
pd.DataFrame (T4Const_louis_clm_smnv.variables['scalarSnowDepth'][:], columns=[hruT4lcs]),
pd.DataFrame (T4Const_louis_eub_jrdn.variables['scalarSnowDepth'][:], columns=[hruT4lej]),
pd.DataFrame (T4Const_louis_eub_smnv.variables['scalarSnowDepth'][:], columns=[hruT4les]),
pd.DataFrame (T4Const_std_clm_jrdn.variables['scalarSnowDepth'][:], columns=[hruT4scj]),
pd.DataFrame (T4Const_std_clm_smnv.variables['scalarSnowDepth'][:], columns=[hruT4scs]),
pd.DataFrame (T4Const_std_eub_jrdn.variables['scalarSnowDepth'][:], columns=[hruT4sej]),
pd.DataFrame (T4Const_std_eub_smnv.variables['scalarSnowDepth'][:], columns=[hruT4ses])], axis=1)

#snowdepth_df = pd.DataFrame(snowdepth_df0, columns=[hru])
#%% output time step
TimeSb = TConst_louis_clm_jrdn.variables['time'][:] # get values
t_unitSb = TConst_louis_clm_jrdn.variables['time'].units # get unit  "days since 1950-01-01T00:00:00Z"

try :

    t_cal = TConst_louis_clm_jrdn.variables['time'].calendar

except AttributeError : # Attribute doesn't exist

    t_cal = u"gregorian" # or standard

tvalueSb = num2date(TimeSb, units=t_unitSb, calendar=t_cal)
DateSb = [i.strftime("%Y-%m-%d %H:%M") for i in tvalueSb] # -%d %H:%M to display dates as string #i.strftime("%Y-%m-%d %H:%M")        
#%%
snowdepth_df.set_index(pd.DatetimeIndex(DateSb),inplace=True)

counter = pd.DataFrame(np.arange(0,np.size(snowdepth_df['T0lcjHRU111111.0'])),columns=['counter'])
counter.set_index(snowdepth_df.index,inplace=True)
snowdepth_df2 = pd.concat([counter, snowdepth_df], axis=1)

#%%   day of snow disappearance-output ---- Sensititvity analysis - Step2
snowdepth_df6000 = snowdepth_df2[:][6000:6553]

zerosnowdate = []
for val in hru_mod[0]:
    zerosnowdate.append(np.where(snowdepth_df6000[val]==0))
zerosnowdate_omg = [item[0] for item in zerosnowdate] #if item[0] == 1]  
for i,item in enumerate(zerosnowdate_omg):
    if len(item) == 0:
        zerosnowdate_omg[i] = 553
for i,item in enumerate(zerosnowdate_omg):
    zerosnowdate_omg[i] = zerosnowdate_omg[i]+6000
        
first_zerosnowdate =[]
for i,item in enumerate(zerosnowdate_omg):
    if np.size(item)>1:
        #print np.size(item)
        first_zerosnowdate.append(item[0])
    if np.size(item)==1:
        first_zerosnowdate.append(item)

dif_obs_zerosnowdate = (first_zerosnowdate - dayofsnowdisappearance_obs)/24
#%% finding max snowdepth
max_snowdepth=[]
for hrus in hru_mod[0]:
    max_snowdepth.append(snowdepth_df2[hrus].max())
max_residualSWE = max_snowdepth - maxSWEobs
#%%
resSnowDisDate = pd.DataFrame(dif_obs_zerosnowdate, columns=['resSnowDisDate'])
resMaxSWE = pd.DataFrame(max_residualSWE, columns=['resMaxSWE'])
residual_df = pd.concat([resSnowDisDate, resMaxSWE], axis=1)
residual_df.set_index(hru_mod[0],inplace=True)

#desiredHRUnum = residual_df.index[(abs(residual_df['resSnowDisDate']) <= 2) & (abs(residual_df['resMaxSWE'])<=0.443)].tolist()
#desiredHRU = [y+1 for y in desiredHRUnum]
#desiredHRUname = ['hru{}'.format(x) for x in desiredHRU]
#
#desiredparams = []
#for q in range (np.size(desiredHRUnum)):
#    desiredparams.append(hruidx_dfAll[0][desiredHRUnum[q]])
#%%
#plt.plot(max_residualSWE, dif_obs_zerosnowdate, 'go')
##plt.title('Residual maxSWE vs. snowDis', position=(0.04, 0.03), ha='left', fontsize=12)
#plt.xlabel('Residual maxSWE')
#plt.ylabel('Residual snowDis')
#plt.savefig('Residual.png')
#%%
desiredMaxSweT0 = [residual_df ['resMaxSWE']['T0lcjHRU112111.0'],residual_df ['resMaxSWE']['T0lcjHRU112121.0'],residual_df ['resMaxSWE']['T0lcjHRU112131.0'],
                   
                   residual_df ['resMaxSWE']['T0lcsHRU112111.0'],residual_df ['resMaxSWE']['T0lcsHRU112121.0'],residual_df ['resMaxSWE']['T0lcsHRU112131.0'],
                   residual_df ['resMaxSWE']['T0lcsHRU112211.0'],residual_df ['resMaxSWE']['T0lcsHRU112221.0'],residual_df ['resMaxSWE']['T0lcsHRU112231.0'],
                   
                   residual_df ['resMaxSWE']['T0lejHRU112111.0'],residual_df ['resMaxSWE']['T0lejHRU112121.0'],residual_df ['resMaxSWE']['T0lejHRU112131.0'],
                   
                   residual_df ['resMaxSWE']['T0lesHRU112111.0'],residual_df ['resMaxSWE']['T0lesHRU112121.0'],residual_df ['resMaxSWE']['T0lesHRU112131.0'],
                   residual_df ['resMaxSWE']['T0lesHRU112211.0'],residual_df ['resMaxSWE']['T0lesHRU112221.0'],residual_df ['resMaxSWE']['T0lesHRU112231.0'],
                   
                   residual_df ['resMaxSWE']['T0scjHRU112111.0'],residual_df ['resMaxSWE']['T0scjHRU112211.0'],residual_df ['resMaxSWE']['T0scjHRU112221.0'],residual_df ['resMaxSWE']['T0scjHRU112231.0'],
                   
                   residual_df ['resMaxSWE']['T0scsHRU112111.0'],residual_df ['resMaxSWE']['T0scsHRU111111.0'],residual_df ['resMaxSWE']['T0scsHRU111121.0'],residual_df ['resMaxSWE']['T0scsHRU111131.0'],
                   residual_df ['resMaxSWE']['T0scsHRU112211.0'],residual_df ['resMaxSWE']['T0scsHRU112221.0'],residual_df ['resMaxSWE']['T0scsHRU112231.0'],residual_df ['resMaxSWE']['T0scsHRU212211.0'],
                   residual_df ['resMaxSWE']['T0scsHRU212221.0'],residual_df ['resMaxSWE']['T0scsHRU212231.0'],
                   
                   residual_df ['resMaxSWE']['T0sejHRU112111.0'],residual_df ['resMaxSWE']['T0sejHRU112231.0'],
                   
                   residual_df ['resMaxSWE']['T0sesHRU112111.0'],residual_df ['resMaxSWE']['T0sesHRU112211.0'],residual_df ['resMaxSWE']['T0sesHRU112221.0'],
                   residual_df ['resMaxSWE']['T0sesHRU112231.0'],residual_df ['resMaxSWE']['T0sesHRU111131.0'],residual_df ['resMaxSWE']['T0sesHRU212131.0']]

desiredMaxSweT2 = [residual_df ['resMaxSWE']['T2lcjHRU112111.0'],residual_df ['resMaxSWE']['T2lcjHRU112121.0'],residual_df ['resMaxSWE']['T2lcjHRU112131.0'],
                   
                   residual_df ['resMaxSWE']['T2lcsHRU112111.0'],residual_df ['resMaxSWE']['T2lcsHRU112121.0'],residual_df ['resMaxSWE']['T2lcsHRU112131.0'],
                   residual_df ['resMaxSWE']['T2lcsHRU112211.0'],residual_df ['resMaxSWE']['T2lcsHRU112221.0'],residual_df ['resMaxSWE']['T2lcsHRU112231.0'],
                   
                   residual_df ['resMaxSWE']['T2lejHRU112111.0'],residual_df ['resMaxSWE']['T2lejHRU112121.0'],residual_df ['resMaxSWE']['T2lejHRU112131.0'],
                   
                   residual_df ['resMaxSWE']['T2lesHRU112111.0'],residual_df ['resMaxSWE']['T2lesHRU112121.0'],residual_df ['resMaxSWE']['T2lesHRU112131.0'],
                   residual_df ['resMaxSWE']['T2lesHRU112211.0'],residual_df ['resMaxSWE']['T2lesHRU112221.0'],residual_df ['resMaxSWE']['T2lesHRU112231.0'],
                   
                   residual_df ['resMaxSWE']['T2scjHRU112111.0'],residual_df ['resMaxSWE']['T2scjHRU112211.0'],residual_df ['resMaxSWE']['T2scjHRU112221.0'],residual_df ['resMaxSWE']['T2scjHRU112231.0'],
                   
                   residual_df ['resMaxSWE']['T2scsHRU112111.0'],residual_df ['resMaxSWE']['T2scsHRU111111.0'],residual_df ['resMaxSWE']['T2scsHRU111121.0'],residual_df ['resMaxSWE']['T2scsHRU111131.0'],
                   residual_df ['resMaxSWE']['T2scsHRU112211.0'],residual_df ['resMaxSWE']['T2scsHRU112221.0'],residual_df ['resMaxSWE']['T2scsHRU112231.0'],residual_df ['resMaxSWE']['T2scsHRU212211.0'],
                   residual_df ['resMaxSWE']['T2scsHRU212221.0'],residual_df ['resMaxSWE']['T2scsHRU212231.0'],
                   
                   residual_df ['resMaxSWE']['T2sejHRU112111.0'],residual_df ['resMaxSWE']['T2sejHRU112231.0'],
                   
                   residual_df ['resMaxSWE']['T2sesHRU112111.0'],residual_df ['resMaxSWE']['T2sesHRU112211.0'],residual_df ['resMaxSWE']['T2sesHRU112221.0'],
                   residual_df ['resMaxSWE']['T2sesHRU112231.0'],residual_df ['resMaxSWE']['T2sesHRU111131.0'],residual_df ['resMaxSWE']['T2sesHRU212131.0']]

desiredMaxSweT4 = [residual_df ['resMaxSWE']['T4lcjHRU112111.0'],residual_df ['resMaxSWE']['T4lcjHRU112121.0'],residual_df ['resMaxSWE']['T4lcjHRU112131.0'],
                   
                   residual_df ['resMaxSWE']['T4lcsHRU112111.0'],residual_df ['resMaxSWE']['T4lcsHRU112121.0'],residual_df ['resMaxSWE']['T4lcsHRU112131.0'],
                   residual_df ['resMaxSWE']['T4lcsHRU112211.0'],residual_df ['resMaxSWE']['T4lcsHRU112221.0'],residual_df ['resMaxSWE']['T4lcsHRU112231.0'],
                   
                   residual_df ['resMaxSWE']['T4lejHRU112111.0'],residual_df ['resMaxSWE']['T4lejHRU112121.0'],residual_df ['resMaxSWE']['T4lejHRU112131.0'],
                   
                   residual_df ['resMaxSWE']['T4lesHRU112111.0'],residual_df ['resMaxSWE']['T4lesHRU112121.0'],residual_df ['resMaxSWE']['T4lesHRU112131.0'],
                   residual_df ['resMaxSWE']['T4lesHRU112211.0'],residual_df ['resMaxSWE']['T4lesHRU112221.0'],residual_df ['resMaxSWE']['T4lesHRU112231.0'],
                   
                   residual_df ['resMaxSWE']['T4scjHRU112111.0'],residual_df ['resMaxSWE']['T4scjHRU112211.0'],residual_df ['resMaxSWE']['T4scjHRU112221.0'],residual_df ['resMaxSWE']['T4scjHRU112231.0'],
                   
                   residual_df ['resMaxSWE']['T4scsHRU112111.0'],residual_df ['resMaxSWE']['T4scsHRU111111.0'],residual_df ['resMaxSWE']['T4scsHRU111121.0'],residual_df ['resMaxSWE']['T4scsHRU111131.0'],
                   residual_df ['resMaxSWE']['T4scsHRU112211.0'],residual_df ['resMaxSWE']['T4scsHRU112221.0'],residual_df ['resMaxSWE']['T4scsHRU112231.0'],residual_df ['resMaxSWE']['T4scsHRU212211.0'],
                   residual_df ['resMaxSWE']['T4scsHRU212221.0'],residual_df ['resMaxSWE']['T4scsHRU212231.0'],
                   
                   residual_df ['resMaxSWE']['T4sejHRU112111.0'],residual_df ['resMaxSWE']['T4sejHRU112231.0'],
                   
                   residual_df ['resMaxSWE']['T4sesHRU112111.0'],residual_df ['resMaxSWE']['T4sesHRU112211.0'],residual_df ['resMaxSWE']['T4sesHRU112221.0'],
                   residual_df ['resMaxSWE']['T4sesHRU112231.0'],residual_df ['resMaxSWE']['T4sesHRU111131.0'],residual_df ['resMaxSWE']['T4sesHRU212131.0']]

#fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(6, 6), sharey=True)
#axes[0].boxplot(desiredMaxSweT0, patch_artist=True)
#axes[0].set_title('Default_Temp')
#    
#axes[1].boxplot(desiredMaxSweT2, patch_artist=True)
#axes[1].set_title('S1_Temp2')
#
#axes[2].boxplot(desiredMaxSweT4, patch_artist=True)
#axes[2].set_title('S2_Temp4')
##plt.ylabel('Residual maxSWE')
#
##plt.title('Residual maxSWE vs. snowDis', position=(0.04, 0.03), ha='left', fontsize=12)
##plt.xlabel('Residual maxSWE')
#plt.savefig('S2.png')

#%%
#d0 = [desiredMaxSweT0,desiredMaxSweT2,desiredMaxSweT4]
#bp0 = plt.boxplot(d0, patch_artist=True)
#bp0['boxes'][0].set(color='red', linewidth=2, facecolor = 'skyblue', hatch = '/')
#bp0['boxes'][1].set(color='orange', linewidth=2, facecolor = 'olive', hatch = '/')
#bp0['boxes'][2].set(color='tan', linewidth=2, facecolor = 'pink', hatch = '/')
#plt.xticks([1, 2, 3], ['T0', 'T+2', 'T+4'])
#plt.savefig('resSD2.png')
#plt.hold()

#for box in bp0['boxes']:
#    # change outline color
#    box.set(color='red', linewidth=2)
#    # change fill color
#    box.set(facecolor = 'green' )
#    # change hatch
#    box.set(hatch = '/')

#for box in bp1['boxes']:
#    box.set(color='blue', linewidth=5)
#    box.set(facecolor = 'red' )

#%%
desiredresDateT0 = [residual_df ['resSnowDisDate']['T0lcjHRU112111.0'],residual_df ['resSnowDisDate']['T0lcjHRU112121.0'],residual_df ['resSnowDisDate']['T0lcjHRU112131.0'],
                   
                   residual_df ['resSnowDisDate']['T0lcsHRU112111.0'],residual_df ['resSnowDisDate']['T0lcsHRU112121.0'],residual_df ['resSnowDisDate']['T0lcsHRU112131.0'],
                   residual_df ['resSnowDisDate']['T0lcsHRU112211.0'],residual_df ['resSnowDisDate']['T0lcsHRU112221.0'],residual_df ['resSnowDisDate']['T0lcsHRU112231.0'],
                   
                   residual_df ['resSnowDisDate']['T0lejHRU112111.0'],residual_df ['resSnowDisDate']['T0lejHRU112121.0'],residual_df ['resSnowDisDate']['T0lejHRU112131.0'],
                   
                   residual_df ['resSnowDisDate']['T0lesHRU112111.0'],residual_df ['resSnowDisDate']['T0lesHRU112121.0'],residual_df ['resSnowDisDate']['T0lesHRU112131.0'],
                   residual_df ['resSnowDisDate']['T0lesHRU112211.0'],residual_df ['resSnowDisDate']['T0lesHRU112221.0'],residual_df ['resSnowDisDate']['T0lesHRU112231.0'],
                   
                   residual_df ['resSnowDisDate']['T0scjHRU112111.0'],residual_df ['resSnowDisDate']['T0scjHRU112211.0'],residual_df ['resSnowDisDate']['T0scjHRU112221.0'],residual_df ['resSnowDisDate']['T0scjHRU112231.0'],
                   
                   residual_df ['resSnowDisDate']['T0scsHRU112111.0'],residual_df ['resSnowDisDate']['T0scsHRU111111.0'],residual_df ['resSnowDisDate']['T0scsHRU111121.0'],residual_df ['resSnowDisDate']['T0scsHRU111131.0'],
                   residual_df ['resSnowDisDate']['T0scsHRU112211.0'],residual_df ['resSnowDisDate']['T0scsHRU112221.0'],residual_df ['resSnowDisDate']['T0scsHRU112231.0'],residual_df ['resSnowDisDate']['T0scsHRU212211.0'],
                   residual_df ['resSnowDisDate']['T0scsHRU212221.0'],residual_df ['resSnowDisDate']['T0scsHRU212231.0'],
                   
                   residual_df ['resSnowDisDate']['T0sejHRU112111.0'],residual_df ['resSnowDisDate']['T0sejHRU112231.0'],
                   
                   residual_df ['resSnowDisDate']['T0sesHRU112111.0'],residual_df ['resSnowDisDate']['T0sesHRU112211.0'],residual_df ['resSnowDisDate']['T0sesHRU112221.0'],
                   residual_df ['resSnowDisDate']['T0sesHRU112231.0'],residual_df ['resSnowDisDate']['T0sesHRU111131.0'],residual_df ['resSnowDisDate']['T0sesHRU212131.0']]

desiredresDateT2 = [residual_df ['resSnowDisDate']['T2lcjHRU112111.0'],residual_df ['resSnowDisDate']['T2lcjHRU112121.0'],residual_df ['resSnowDisDate']['T2lcjHRU112131.0'],
                   
                   residual_df ['resSnowDisDate']['T2lcsHRU112111.0'],residual_df ['resSnowDisDate']['T2lcsHRU112121.0'],residual_df ['resSnowDisDate']['T2lcsHRU112131.0'],
                   residual_df ['resSnowDisDate']['T2lcsHRU112211.0'],residual_df ['resSnowDisDate']['T2lcsHRU112221.0'],residual_df ['resSnowDisDate']['T2lcsHRU112231.0'],
                   
                   residual_df ['resSnowDisDate']['T2lejHRU112111.0'],residual_df ['resSnowDisDate']['T2lejHRU112121.0'],residual_df ['resSnowDisDate']['T2lejHRU112131.0'],
                   
                   residual_df ['resSnowDisDate']['T2lesHRU112111.0'],residual_df ['resSnowDisDate']['T2lesHRU112121.0'],residual_df ['resSnowDisDate']['T2lesHRU112131.0'],
                   residual_df ['resSnowDisDate']['T2lesHRU112211.0'],residual_df ['resSnowDisDate']['T2lesHRU112221.0'],residual_df ['resSnowDisDate']['T2lesHRU112231.0'],
                   
                   residual_df ['resSnowDisDate']['T2scjHRU112111.0'],residual_df ['resSnowDisDate']['T2scjHRU112211.0'],residual_df ['resSnowDisDate']['T2scjHRU112221.0'],residual_df ['resSnowDisDate']['T2scjHRU112231.0'],
                   
                   residual_df ['resSnowDisDate']['T2scsHRU112111.0'],residual_df ['resSnowDisDate']['T2scsHRU111111.0'],residual_df ['resSnowDisDate']['T2scsHRU111121.0'],residual_df ['resSnowDisDate']['T2scsHRU111131.0'],
                   residual_df ['resSnowDisDate']['T2scsHRU112211.0'],residual_df ['resSnowDisDate']['T2scsHRU112221.0'],residual_df ['resSnowDisDate']['T2scsHRU112231.0'],residual_df ['resSnowDisDate']['T2scsHRU212211.0'],
                   residual_df ['resSnowDisDate']['T2scsHRU212221.0'],residual_df ['resSnowDisDate']['T2scsHRU212231.0'],
                   
                   residual_df ['resSnowDisDate']['T2sejHRU112111.0'],residual_df ['resSnowDisDate']['T2sejHRU112231.0'],
                   
                   residual_df ['resSnowDisDate']['T2sesHRU112111.0'],residual_df ['resSnowDisDate']['T2sesHRU112211.0'],residual_df ['resSnowDisDate']['T2sesHRU112221.0'],
                   residual_df ['resSnowDisDate']['T2sesHRU112231.0'],residual_df ['resSnowDisDate']['T2sesHRU111131.0'],residual_df ['resSnowDisDate']['T2sesHRU212131.0']]

desiredresDateT4 = [residual_df ['resSnowDisDate']['T4lcjHRU112111.0'],residual_df ['resSnowDisDate']['T4lcjHRU112121.0'],residual_df ['resSnowDisDate']['T4lcjHRU112131.0'],
                   
                   residual_df ['resSnowDisDate']['T4lcsHRU112111.0'],residual_df ['resSnowDisDate']['T4lcsHRU112121.0'],residual_df ['resSnowDisDate']['T4lcsHRU112131.0'],
                   residual_df ['resSnowDisDate']['T4lcsHRU112211.0'],residual_df ['resSnowDisDate']['T4lcsHRU112221.0'],residual_df ['resSnowDisDate']['T4lcsHRU112231.0'],
                   
                   residual_df ['resSnowDisDate']['T4lejHRU112111.0'],residual_df ['resSnowDisDate']['T4lejHRU112121.0'],residual_df ['resSnowDisDate']['T4lejHRU112131.0'],
                   
                   residual_df ['resSnowDisDate']['T4lesHRU112111.0'],residual_df ['resSnowDisDate']['T4lesHRU112121.0'],residual_df ['resSnowDisDate']['T4lesHRU112131.0'],
                   residual_df ['resSnowDisDate']['T4lesHRU112211.0'],residual_df ['resSnowDisDate']['T4lesHRU112221.0'],residual_df ['resSnowDisDate']['T4lesHRU112231.0'],
                   
                   residual_df ['resSnowDisDate']['T4scjHRU112111.0'],residual_df ['resSnowDisDate']['T4scjHRU112211.0'],residual_df ['resSnowDisDate']['T4scjHRU112221.0'],residual_df ['resSnowDisDate']['T4scjHRU112231.0'],
                   
                   residual_df ['resSnowDisDate']['T4scsHRU112111.0'],residual_df ['resSnowDisDate']['T4scsHRU111111.0'],residual_df ['resSnowDisDate']['T4scsHRU111121.0'],residual_df ['resSnowDisDate']['T4scsHRU111131.0'],
                   residual_df ['resSnowDisDate']['T4scsHRU112211.0'],residual_df ['resSnowDisDate']['T4scsHRU112221.0'],residual_df ['resSnowDisDate']['T4scsHRU112231.0'],residual_df ['resSnowDisDate']['T4scsHRU212211.0'],
                   residual_df ['resSnowDisDate']['T4scsHRU212221.0'],residual_df ['resSnowDisDate']['T4scsHRU212231.0'],
                   
                   residual_df ['resSnowDisDate']['T4sejHRU112111.0'],residual_df ['resSnowDisDate']['T4sejHRU112231.0'],
                   
                   residual_df ['resSnowDisDate']['T4sesHRU112111.0'],residual_df ['resSnowDisDate']['T4sesHRU112211.0'],residual_df ['resSnowDisDate']['T4sesHRU112221.0'],
                   residual_df ['resSnowDisDate']['T4sesHRU112231.0'],residual_df ['resSnowDisDate']['T4sesHRU111131.0'],residual_df ['resSnowDisDate']['T4sesHRU212131.0']]
#%%
#d1 = [desiredresDateT0,desiredresDateT2,desiredresDateT4]
#bp1 = plt.boxplot(d1, patch_artist=True)
#bp1['boxes'][0].set(color='navy', linewidth=2, facecolor = 'skyblue', hatch = '/')
#bp1['boxes'][1].set(color='blue', linewidth=2, facecolor = 'olive', hatch = '/')
#bp1['boxes'][2].set(color='skyblue', linewidth=2, facecolor = 'pink', hatch = '/')
#
#plt.xticks([1, 2, 3], ['T0', 'T+2', 'T+4'])
#plt.savefig('resDate2.png')

d0 = [desiredMaxSweT0,desiredMaxSweT2,desiredMaxSweT4]
d1 = [desiredresDateT0,desiredresDateT2,desiredresDateT4]

bp0 = plt.boxplot(d0, patch_artist=True)
bp1 = plt.boxplot(d1, patch_artist=True)

bp0['boxes'][0].set(color='red', linewidth=2, facecolor = 'skyblue', hatch = '/')
bp0['boxes'][1].set(color='orange', linewidth=2, facecolor = 'olive', hatch = '/')
bp0['boxes'][2].set(color='tan', linewidth=2, facecolor = 'pink', hatch = '/')

#plt.hold()

bp1['boxes'][0].set(color='navy', linewidth=2, facecolor = 'skyblue', hatch = '/')
bp1['boxes'][1].set(color='blue', linewidth=2, facecolor = 'olive', hatch = '/')
bp1['boxes'][2].set(color='skyblue', linewidth=2, facecolor = 'pink', hatch = '/')

plt.xticks([1, 2, 3], ['T0', 'T+2', 'T+4'])
plt.savefig('resSwe2.png')

































#%%
#TConst_loius_clm_jrdn.txt  112111  112131  112131  ***********
#T2Const_loius_clm_jrdn.txt
#T4Const_loius_clm_jrdn.txt

#TConst_loius_clm_smnv.txt  112111  112131  112131  112211  112231  112231 ************************
#T2Const_loius_clm_smnv.txt
#T4Const_loius_clm_smnv.txt

#TConst_loius_eub_jrdn.txt  112111  112131  112131 *******************************
#T2Const_loius_eub_jrdn.txt
#T4Const_loius_eub_jrdn.txt

#TConst_loius_eub_smnv.txt  112111  112131  112131  112211  112231  112231 *********************************
#T2Const_loius_eub_smnv.txt
#T4Const_loius_eub_smnv.txt

#TConst_std_clm_jrdn.txt  112111  112211  112221  112231 **************************************
#T2Const_std_clm_jrdn.txt
#T4Const_std_clm_jrdn.txt

#TConst_std_clm_smnv.txt  112111  111111  111121  111131  112211  112221  112231  212211  212221  212231 ***************8
#T2Const_std_clm_smnv.txt
#T4Const_std_clm_smnv.txt

#TConst_std_eub_jrdn.txt  112111  112231 *************************
#T2Const_std_eub_jrdn.txt
#T4Const_std_eub_jrdn.txt

#TConst_std_eub_smnv.txt  112111  112211  112221  112231 111131 212131
#T2Const_std_eub_smnv.txt
#T4Const_std_eub_smnv.txt

