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
#forcingData = Dataset("0SenatorBeck_forcing0.nc")
#specific_humidity =forcingData.variables['spechum'][:]

validation = Dataset("validation_senatorBeck_SASP_1hr.nc")
#for i in validation.variables:
#    print i

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

max_swe_obs = max(obs_swe['swe_mm'])
max_swe_date_obs = obs_swe[obs_swe ['swe_mm']== max_swe_obs].index.tolist()    
#%% day of snow disappearance-observation/validation data
for colname in observ_df:
    observ_df [colname] = pd.to_numeric(observ_df[colname])
    observ_df [colname].replace(-9999.0, np.nan, inplace=True)

observ_df2010_init = observ_df['2010-10-01 00:00':'2011-09-30 00:00']
observ_df2010_init ['snowDepth'].replace(np.nan, 0, inplace=True)

counter = pd.DataFrame(np.arange(0,np.size(observ_df2010_init['snowDepth'])),columns=['counter'])
counter.set_index(observ_df2010_init.index,inplace=True)
observ_df2010 = pd.concat([counter, observ_df2010_init], axis=1)

zero_snow_date = []
for c1 in range(np.size(observ_df2010['snowDepth'])):
    if observ_df2010['snowDepth'][c1]==0 and observ_df2010['counter'][c1]>6000:
        zero_snow_date.append(observ_df2010['counter'][c1])

dayofsnowdisappearance_obs = zero_snow_date [0]  
#%% output snowdepth dataframe
p1 = [200000, 360000] #albedoDecayRate  
p2 = [0.40] #albedoMinSpring   
p3 = [0.75, 0.84, 0.91] #albedoMax      

p4 = [0.001,0.002] #z0Snow
p5 = [0.18,0.28,0.50] #windReductionParam
#p6 = [0.35]#,0.25,0.50] #fixedThermalCond_snow

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

paramID = hru_ix_ID(p1, p2, p3, p4, p5)
model_names = ['AcASlSWcTCj','AcASlSWcTCs','AcASlSWuTCj','AcASlSWuTCs','AcASsSWcTCj','AcASsSWcTCs','AcASsSWuTCj','AcASsSWuTCs']
senario_names = ['T0','T2','T4','H2','H4']
hru_names =[]
for i in senario_names:
    for j in model_names:
        hru_names.append(['{}{}{}'.format(i, j, k) for k in paramID])

hru_names1 = np.reshape(hru_names,(1440,1))
hru_names_df = pd.DataFrame (hru_names1)
hru_names_finale = ['T0AcASlSWcTCj11311.0','T0AcASlSWcTCj11312.0','T0AcASlSWcTCj11313.0',
                    'T0AcASlSWcTCs11311.0','T0AcASlSWcTCs11312.0','T0AcASlSWcTCs11313.0','T0AcASlSWcTCs21113.0',
                    'T0AcASlSWuTCj11311.0','T0AcASlSWuTCj11312.0','T0AcASlSWuTCj11313.0',
                    'T0AcASlSWuTCs11311.0','T0AcASlSWuTCs11312.0','T0AcASlSWuTCs11313.0',
                    'T0AcASsSWcTCj11311.0','T0AcASsSWcTCj21111.0','T0AcASsSWcTCj21112.0','T0AcASsSWcTCj21113.0',
                    'T0AcASsSWcTCs11211.0','T0AcASsSWcTCs11212.0','T0AcASsSWcTCs11213.0','T0AcASsSWcTCs11311.0','T0AcASsSWcTCs11321.0','T0AcASsSWcTCs11322.0','T0AcASsSWcTCs11323.0','T0AcASsSWcTCs21111.0','T0AcASsSWcTCs21112.0','T0AcASsSWcTCs21113.0',
                    'T0AcASsSWuTCj11311.0','T0AcASsSWuTCj21111.0','T0AcASsSWuTCj21112.0',
                    'T0AcASsSWuTCs11311.0','T0AcASsSWuTCs11212.0','T0AcASsSWuTCs11213.0','T0AcASsSWuTCs21111.0','T0AcASsSWuTCs21112.0','T0AcASsSWuTCs21113.0',
                    
                    'T2AcASlSWcTCj11311.0','T2AcASlSWcTCj11312.0','T2AcASlSWcTCj11313.0',
                    'T2AcASlSWcTCs11311.0','T2AcASlSWcTCs11312.0','T2AcASlSWcTCs11313.0','T2AcASlSWcTCs21113.0',
                    'T2AcASlSWuTCj11311.0','T2AcASlSWuTCj11312.0','T2AcASlSWuTCj11313.0',
                    'T2AcASlSWuTCs11311.0','T2AcASlSWuTCs11312.0','T2AcASlSWuTCs11313.0',
                    'T2AcASsSWcTCj11311.0','T2AcASsSWcTCj21111.0','T2AcASsSWcTCj21112.0','T2AcASsSWcTCj21113.0',
                    'T2AcASsSWcTCs11211.0','T2AcASsSWcTCs11212.0','T2AcASsSWcTCs11213.0','T2AcASsSWcTCs11311.0','T2AcASsSWcTCs11321.0','T2AcASsSWcTCs11322.0','T2AcASsSWcTCs11323.0','T2AcASsSWcTCs21111.0','T2AcASsSWcTCs21112.0','T2AcASsSWcTCs21113.0',
                    'T2AcASsSWuTCj11311.0','T2AcASsSWuTCj21111.0','T2AcASsSWuTCj21112.0',
                    'T2AcASsSWuTCs11311.0','T2AcASsSWuTCs11212.0','T2AcASsSWuTCs11213.0','T2AcASsSWuTCs21111.0','T2AcASsSWuTCs21112.0','T2AcASsSWuTCs21113.0',
                    
                    'T4AcASlSWcTCj11311.0','T4AcASlSWcTCj11312.0','T4AcASlSWcTCj11313.0',
                    'T4AcASlSWcTCs11311.0','T4AcASlSWcTCs11312.0','T4AcASlSWcTCs11313.0','T4AcASlSWcTCs21113.0',
                    'T4AcASlSWuTCj11311.0','T4AcASlSWuTCj11312.0','T4AcASlSWuTCj11313.0',
                    'T4AcASlSWuTCs11311.0','T4AcASlSWuTCs11312.0','T4AcASlSWuTCs11313.0',
                    'T4AcASsSWcTCj11311.0','T4AcASsSWcTCj21111.0','T4AcASsSWcTCj21112.0','T4AcASsSWcTCj21113.0',
                    'T4AcASsSWcTCs11211.0','T4AcASsSWcTCs11212.0','T4AcASsSWcTCs11213.0','T4AcASsSWcTCs11311.0','T4AcASsSWcTCs11321.0','T4AcASsSWcTCs11322.0','T4AcASsSWcTCs11323.0','T4AcASsSWcTCs21111.0','T4AcASsSWcTCs21112.0','T4AcASsSWcTCs21113.0',
                    'T4AcASsSWuTCj11311.0','T4AcASsSWuTCj21111.0','T4AcASsSWuTCj21112.0',
                    'T4AcASsSWuTCs11311.0','T4AcASsSWuTCs11212.0','T4AcASsSWuTCs11213.0','T4AcASsSWuTCs21111.0','T4AcASsSWuTCs21112.0','T4AcASsSWuTCs21113.0',
                    
                    'H2AcASlSWcTCj11311.0','H2AcASlSWcTCj11312.0','H2AcASlSWcTCj11313.0',
                    'H2AcASlSWcTCs11311.0','H2AcASlSWcTCs11312.0','H2AcASlSWcTCs11313.0','H2AcASlSWcTCs21113.0',
                    'H2AcASlSWuTCj11311.0','H2AcASlSWuTCj11312.0','H2AcASlSWuTCj11313.0',
                    'H2AcASlSWuTCs11311.0','H2AcASlSWuTCs11312.0','H2AcASlSWuTCs11313.0',
                    'H2AcASsSWcTCj11311.0','H2AcASsSWcTCj21111.0','H2AcASsSWcTCj21112.0','H2AcASsSWcTCj21113.0',
                    'H2AcASsSWcTCs11211.0','H2AcASsSWcTCs11212.0','H2AcASsSWcTCs11213.0','H2AcASsSWcTCs11311.0','H2AcASsSWcTCs11321.0','H2AcASsSWcTCs11322.0','H2AcASsSWcTCs11323.0','H2AcASsSWcTCs21111.0','H2AcASsSWcTCs21112.0','H2AcASsSWcTCs21113.0',
                    'H2AcASsSWuTCj11311.0','H2AcASsSWuTCj21111.0','H2AcASsSWuTCj21112.0',
                    'H2AcASsSWuTCs11311.0','H2AcASsSWuTCs11212.0','H2AcASsSWuTCs11213.0','H2AcASsSWuTCs21111.0','H2AcASsSWuTCs21112.0','H2AcASsSWuTCs21113.0',
                    
                    'H4AcASlSWcTCj11311.0','H4AcASlSWcTCj11312.0','H4AcASlSWcTCj11313.0',
                    'H4AcASlSWcTCs11311.0','H4AcASlSWcTCs11312.0','H4AcASlSWcTCs11313.0','H4AcASlSWcTCs21113.0',
                    'H4AcASlSWuTCj11311.0','H4AcASlSWuTCj11312.0','H4AcASlSWuTCj11313.0',
                    'H4AcASlSWuTCs11311.0','H4AcASlSWuTCs11312.0','H4AcASlSWuTCs11313.0',
                    'H4AcASsSWcTCj11311.0','H4AcASsSWcTCj21111.0','H4AcASsSWcTCj21112.0','H4AcASsSWcTCj21113.0',
                    'H4AcASsSWcTCs11211.0','H4AcASsSWcTCs11212.0','H4AcASsSWcTCs11213.0','H4AcASsSWcTCs11311.0','H4AcASsSWcTCs11321.0','H4AcASsSWcTCs11322.0','H4AcASsSWcTCs11323.0','H4AcASsSWcTCs21111.0','H4AcASsSWcTCs21112.0','H4AcASsSWcTCs21113.0',
                    'H4AcASsSWuTCj11311.0','H4AcASsSWuTCj21111.0','H4AcASsSWuTCj21112.0',
                    'H4AcASsSWuTCs11311.0','H4AcASsSWuTCs11212.0','H4AcASsSWuTCs11213.0','H4AcASsSWuTCs21111.0','H4AcASsSWuTCs21112.0','H4AcASsSWuTCs21113.0']
#%% reading output_swe files
ac_ncfiles = ["T0AcASlSWcTCj_2010-2011_senatorConstantDecayRate_1.nc","T0AcASlSWcTCs_2010-2011_senatorConstantDecayRate_1.nc",
              "T0AcASlSWuTCj_2010-2011_senatorConstantDecayRate_1.nc","T0AcASlSWuTCs_2010-2011_senatorConstantDecayRate_1.nc",
              "T0AcASsSWcTCj_2010-2011_senatorConstantDecayRate_1.nc","T0AcASsSWcTCs_2010-2011_senatorConstantDecayRate_1.nc",
              "T0AcASsSWuTCj_2010-2011_senatorConstantDecayRate_1.nc","T0AcASsSWuTCs_2010-2011_senatorConstantDecayRate_1.nc",
              
              "T2AcASlSWcTCj_2010-2011_senatorConstantDecayRate_1.nc","T2AcASlSWcTCs_2010-2011_senatorConstantDecayRate_1.nc",
              "T2AcASlSWuTCj_2010-2011_senatorConstantDecayRate_1.nc","T2AcASlSWuTCs_2010-2011_senatorConstantDecayRate_1.nc",
              "T2AcASsSWcTCj_2010-2011_senatorConstantDecayRate_1.nc","T2AcASsSWcTCs_2010-2011_senatorConstantDecayRate_1.nc",
              "T2AcASsSWuTCj_2010-2011_senatorConstantDecayRate_1.nc","T2AcASsSWuTCs_2010-2011_senatorConstantDecayRate_1.nc",
              
              "T4AcASlSWcTCj_2010-2011_senatorConstantDecayRate_1.nc","T4AcASlSWcTCs_2010-2011_senatorConstantDecayRate_1.nc",
              "T4AcASlSWuTCj_2010-2011_senatorConstantDecayRate_1.nc","T4AcASlSWuTCs_2010-2011_senatorConstantDecayRate_1.nc",
              "T4AcASsSWcTCj_2010-2011_senatorConstantDecayRate_1.nc","T4AcASsSWcTCs_2010-2011_senatorConstantDecayRate_1.nc",
              "T4AcASsSWuTCj_2010-2011_senatorConstantDecayRate_1.nc","T4AcASsSWuTCs_2010-2011_senatorConstantDecayRate_1.nc",
              
              "H2AcASlSWcTCj_2010-2011_senatorConstantDecayRate_1.nc","H2AcASlSWcTCs_2010-2011_senatorConstantDecayRate_1.nc",
              "H2AcASlSWuTCj_2010-2011_senatorConstantDecayRate_1.nc","H2AcASlSWuTCs_2010-2011_senatorConstantDecayRate_1.nc",
              "H2AcASsSWcTCj_2010-2011_senatorConstantDecayRate_1.nc","H2AcASsSWcTCs_2010-2011_senatorConstantDecayRate_1.nc",
              "H2AcASsSWuTCj_2010-2011_senatorConstantDecayRate_1.nc","H2AcASsSWuTCs_2010-2011_senatorConstantDecayRate_1.nc",
              
              "H4AcASlSWcTCj_2010-2011_senatorConstantDecayRate_1.nc","H4AcASlSWcTCs_2010-2011_senatorConstantDecayRate_1.nc",
              "H4AcASlSWuTCj_2010-2011_senatorConstantDecayRate_1.nc","H4AcASlSWuTCs_2010-2011_senatorConstantDecayRate_1.nc",
              "H4AcASsSWcTCj_2010-2011_senatorConstantDecayRate_1.nc","H4AcASsSWcTCs_2010-2011_senatorConstantDecayRate_1.nc",
              "H4AcASsSWuTCj_2010-2011_senatorConstantDecayRate_1.nc","H4AcASsSWuTCs_2010-2011_senatorConstantDecayRate_1.nc"]
#%%
ac_all = []
for ncfiles in ac_ncfiles:
    ac_all.append(Dataset(ncfiles))

for varname in ac_all[0].variables.keys():
    var = ac_all[0].variables[varname]
    print (varname, var.dtype, var.dimensions, var.shape)

ac_sd = []
for dfs in ac_all:
    ac_sd.append(pd.DataFrame(dfs['scalarSnowDepth'][:]))
ac_sd_df = pd.concat (ac_sd, axis=1)
ac_sd_df.columns = hru_names_df[0]

ac_swe = []
for dfs in ac_all:
    ac_swe.append(pd.DataFrame(dfs['scalarSWE'][:]))
ac_swe_df = pd.concat (ac_swe, axis=1)
ac_swe_df.columns = hru_names_df[0]
#%% output time step
TimeSb = ac_all[0].variables['time'][:] # get values
t_unitSb = ac_all[0].variables['time'].units # get unit  "days since 1950-01-01T00:00:00Z"

try :

    t_cal = ac_all[0].variables['time'].calendar

except AttributeError : # Attribute doesn't exist

    t_cal = u"gregorian" # or standard

tvalueSb = num2date(TimeSb, units=t_unitSb, calendar=t_cal)
DateSb = [i.strftime("%Y-%m-%d %H:%M") for i in tvalueSb] # -%d %H:%M to display dates as string #i.strftime("%Y-%m-%d %H:%M")        
#%% day of snow disappearance-final output
ac_sd_df.set_index(pd.DatetimeIndex(DateSb),inplace=True)
counter = pd.DataFrame(np.arange(0,np.size(ac_sd_df['T0AcASlSWcTCj11111.0'])),columns=['counter'])
counter.set_index(ac_sd_df.index,inplace=True)
ac_sd_df2 = pd.concat([counter, ac_sd_df], axis=1)

ac_swe_df.set_index(pd.DatetimeIndex(DateSb),inplace=True)
counter = pd.DataFrame(np.arange(0,np.size(ac_swe_df['T0AcASlSWcTCj11111.0'])),columns=['counter'])
counter.set_index(ac_swe_df.index,inplace=True)
ac_swe_df2 = pd.concat([counter, ac_swe_df], axis=1)

ac_sd_df_finale = ac_sd_df2[hru_names_finale]
ac_swe_df_finale = ac_swe_df2[hru_names_finale]
#%%   
av_sd_df5000 = ac_sd_df_finale[:][5000:8737]

zerosnowdate = []
for val in hru_names_finale:
    zerosnowdate.append(np.where(av_sd_df5000[val]==0))
zerosnowdate_omg = [item[0] for item in zerosnowdate] #if item[0] == 1]  
for i,item in enumerate(zerosnowdate_omg):
    if len(item) == 0:
        zerosnowdate_omg[i] = 3737
for i,item in enumerate(zerosnowdate_omg):
    zerosnowdate_omg[i] = zerosnowdate_omg[i]+5000
        
first_zerosnowdate =[]
for i,item in enumerate(zerosnowdate_omg):
    if np.size(item)>1:
        #print np.size(item)
        first_zerosnowdate.append(item[0])
    if np.size(item)==1:
        first_zerosnowdate.append(item)

zerosnowdate_residual = pd.DataFrame((first_zerosnowdate - dayofsnowdisappearance_obs)/24, columns=['resSnowDisDate'])
zerosnowdate_residual.set_index([hru_names_finale], inplace=True)
#%% finding max snowdepth and swe
max_swe=[]
for hrus in hru_names_finale:
    max_swe.append(ac_swe_df_finale[hrus].max())
max_swe_finale = []
for items in max_swe:
    if np.size(items)==1:
        max_swe_finale.append(items)
    else:
        max_swe_finale.append(items[0])
max_residual_SWE = max_swe_finale - max_swe_obs

swe_corsp_max2date = []
for hrus in hru_names_finale:
    swe_corsp_max2date.append(ac_swe_df_finale[hrus]['2011-05-04 08:00:00'])
max_residual_swe_corsp = pd.DataFrame((swe_corsp_max2date - max_swe_obs), columns=['resCorspMaxSWE'])
max_residual_swe_corsp.set_index(pd.DataFrame(hru_names_finale)[0],inplace=True)
  
#%%
tvalueSb = num2date(TimeSb, units=t_unitSb, calendar=t_cal)
DateSb2 = [i.strftime("%Y-%m") for i in tvalueSb]
#
sbx = np.arange(0,np.size(DateSb2))
sb_xticks = DateSb2
sbfig, sbax = plt.subplots(1,1)
plt.xticks(sbx, sb_xticks[::1000], rotation=25)
sbax.xaxis.set_major_locator(ticker.AutoLocator())
#%%
count = 0
for jj in range (5):
    if (jj+count)<180 :
        #av_swe_plot = av_swe_df_finale[av_swe_df_finale.columns[count]]
        plt.plot(ac_swe_df_finale[ac_swe_df_finale.columns[jj+count]])
        count = count + 36
        print count
    
plt.plot(obs_swe['swe_mm'], 'ko', linewidth=1) 
plt.title('AcASlSWcTCj11311')
plt.legend(['T0','T2','T4','H2','H4','obs'])  
plt.xlabel('Time 2010-2011')
plt.ylabel('SWE')
#plt.show()
plt.savefig('AcASlSWcTCj11311.png')
plt.close()
#%%
count = 30
for jj in range (5):
    if (jj+count)<180 :
        #av_swe_plot = av_swe_df_finale[av_swe_df_finale.columns[count]]
        plt.plot(ac_swe_df_finale[ac_swe_df_finale.columns[jj+count]])
        count = count + 36
        print count
    
plt.plot(obs_swe['swe_mm'], 'ko', linewidth=1) 
plt.title('AcASsSWuTCs11212')
plt.legend(['T0','T2','T4','H2','H4','obs'])  
plt.xlabel('Time 2010-2011')
plt.ylabel('SWE')
#plt.show()
plt.savefig('AcASsSWuTCs11212.png')
plt.close()
#%%
zerosnowdate_residual_Scnos = pd.DataFrame(np.resize(zerosnowdate_residual, (5,36)).T, columns = ['T0','T2','T4','H2','H4'])
d0 = [zerosnowdate_residual_Scnos['T0'],zerosnowdate_residual_Scnos['T2'],zerosnowdate_residual_Scnos['T4'],zerosnowdate_residual_Scnos['H2'],zerosnowdate_residual_Scnos['H4']]
bp1 = plt.boxplot(d0, patch_artist=True)
bp1['boxes'][0].set(color='forestgreen', linewidth=2, facecolor = 'skyblue', hatch = '/')
bp1['boxes'][1].set(color='palevioletred', linewidth=2, facecolor = 'olive', hatch = '/')
bp1['boxes'][2].set(color='darkred', linewidth=2, facecolor = 'pink', hatch = '/')
bp1['boxes'][3].set(color='royalblue', linewidth=2, facecolor = 'pink', hatch = '/')
bp1['boxes'][4].set(color='darkslateblue', linewidth=2, facecolor = 'pink', hatch = '/')
plt.xticks([1, 2, 3, 4, 5], ['T0', 'T+2', 'T+4','H+2', 'H+4'])
plt.savefig('resDate.png')
plt.close()
#%%
max_residual_swe_corsp_Scnos = pd.DataFrame(np.resize(max_residual_swe_corsp, (5,36)).T, columns = ['T0','T2','T4','H2','H4'])
d1 = [max_residual_swe_corsp_Scnos['T0'],max_residual_swe_corsp_Scnos['T2'],max_residual_swe_corsp_Scnos['T4'],max_residual_swe_corsp_Scnos['H2'],max_residual_swe_corsp_Scnos['H4']]
bp2 = plt.boxplot(d1, patch_artist=True)
bp2['boxes'][0].set(color='green', linewidth=2, facecolor = 'skyblue', hatch = '/')
bp2['boxes'][1].set(color='pink', linewidth=2, facecolor = 'olive', hatch = '/')
bp2['boxes'][2].set(color='orangered', linewidth=2, facecolor = 'pink', hatch = '/')
bp2['boxes'][3].set(color='deepskyblue', linewidth=2, facecolor = 'pink', hatch = '/')
bp2['boxes'][4].set(color='steelblue', linewidth=2, facecolor = 'pink', hatch = '/')
plt.xticks([1, 2, 3, 4, 5], ['T0', 'T+2', 'T+4','H+2', 'H+4'])
plt.savefig('resSWE.png')
plt.close()
#%%
import Combo_bestModels_swe_v
#%%
v_residual = Combo_bestModels_swe_v.residual_df_finale
zerosnowdate_residual_VScnos = pd.DataFrame(np.resize(v_residual['resSnowDisDate'], (5,4)).T, columns = ['T0','T2','T4','H2','H4'])
max_residual_swe_corsp_VScnos = pd.DataFrame(np.resize(v_residual['resCorspMaxSWE'], (5,4)).T, columns = ['T0','T2','T4','H2','H4'])

zerosnowdate_residual_Sc_finale = pd.concat([zerosnowdate_residual_Scnos, zerosnowdate_residual_VScnos])
max_residual_swe_corsp_Sc_finale = pd.concat([max_residual_swe_corsp_Scnos, max_residual_swe_corsp_VScnos])
#%%
d0 = [zerosnowdate_residual_Sc_finale['T0'],zerosnowdate_residual_Sc_finale['T2'],zerosnowdate_residual_Sc_finale['T4'],zerosnowdate_residual_Sc_finale['H2'],zerosnowdate_residual_Sc_finale['H4']]
bp1 = plt.boxplot(d0, patch_artist=True)
bp1['boxes'][0].set(color='forestgreen', linewidth=2, facecolor = 'skyblue', hatch = '/')
bp1['boxes'][1].set(color='palevioletred', linewidth=2, facecolor = 'olive', hatch = '/')
bp1['boxes'][2].set(color='darkred', linewidth=2, facecolor = 'pink', hatch = '/')
bp1['boxes'][3].set(color='royalblue', linewidth=2, facecolor = 'pink', hatch = '/')
bp1['boxes'][4].set(color='darkslateblue', linewidth=2, facecolor = 'pink', hatch = '/')
plt.xticks([1, 2, 3, 4, 5], ['T0', 'T+2', 'T+4','H+2', 'H+4'])
plt.ylabel('residul snow disappearance date (day)')
plt.savefig('resDate_finale.png')
plt.close()
#%%
d1 = [max_residual_swe_corsp_Sc_finale['T0'],max_residual_swe_corsp_Sc_finale['T2'],max_residual_swe_corsp_Sc_finale['T4'],max_residual_swe_corsp_Sc_finale['H2'],max_residual_swe_corsp_Sc_finale['H4']]
bp2 = plt.boxplot(d1, patch_artist=True)
bp2['boxes'][0].set(color='green', linewidth=2, facecolor = 'skyblue', hatch = '/')
bp2['boxes'][1].set(color='pink', linewidth=2, facecolor = 'olive', hatch = '/')
bp2['boxes'][2].set(color='orangered', linewidth=2, facecolor = 'pink', hatch = '/')
bp2['boxes'][3].set(color='deepskyblue', linewidth=2, facecolor = 'pink', hatch = '/')
bp2['boxes'][4].set(color='steelblue', linewidth=2, facecolor = 'pink', hatch = '/')
plt.xticks([1, 2, 3, 4, 5], ['T0', 'T+2', 'T+4','H+2', 'H+4'])
plt.ylabel('residul swe (mm)')
plt.savefig('resSWE_finale.png')
plt.close()
#%%annual mean sensible heat
ac_tsh = []
for dfs in ac_all:
    ac_tsh.append(pd.DataFrame(dfs['scalarSenHeatTotal'][:]))
ac_tsh_df = pd.concat (ac_tsh, axis=1)
ac_tsh_df.columns = hru_names_df[0]
ac_tsh_df_finale = ac_tsh_df[hru_names_finale]
ac_tsh_mean = []
for hrus in hru_names_finale:
    ac_tsh_mean.append(ac_tsh_df_finale[hrus].mean())
ac_tsh_mean_df = pd.DataFrame(ac_tsh_mean, columns =['mean_yr_sensibleheat'])
ac_tsh_mean_df.set_index(pd.DataFrame(hru_names_finale)[0],inplace=True)
ac_tsh_mean_df_sc = pd.DataFrame(np.resize(ac_tsh_mean_df['mean_yr_sensibleheat'], (5,36)).T, columns = ['T0','T2','T4','H2','H4'])

av_all2 = Combo_bestModels_swe_v.av_all
av_tsh = []
for dfs in av_all2:
    av_tsh.append(pd.DataFrame(dfs['scalarSenHeatTotal'][:]))
av_tsh_df = pd.concat (av_tsh, axis=1)
av_tsh_df.columns = Combo_bestModels_swe_v.hru_names_df[0]
hru_names_finaleV = Combo_bestModels_swe_v.hru_names_finale[0]
av_tsh_df_finale = av_tsh_df[Combo_bestModels_swe_v.hru_names_finale[0]]
av_tsh_mean = []
for hrus in hru_names_finaleV:
    av_tsh_mean.append(av_tsh_df_finale[hrus].mean())
av_tsh_mean_df = pd.DataFrame(av_tsh_mean, columns =['mean_yr_sensibleheat'])
av_tsh_mean_df.set_index(pd.DataFrame(hru_names_finaleV)[0],inplace=True)
av_tsh_mean_df_sc = pd.DataFrame(np.resize(av_tsh_mean_df['mean_yr_sensibleheat'], (5,4)).T, columns = ['T0','T2','T4','H2','H4'])

tsh_mean_df = pd.concat([ac_tsh_mean_df_sc, av_tsh_mean_df_sc])

d2 = [tsh_mean_df['T0'],tsh_mean_df['T2'],tsh_mean_df['T4'],tsh_mean_df['H2'],tsh_mean_df['H4']]
bp3 = plt.boxplot(d2, patch_artist=True)
bp3['boxes'][0].set(color='seagreen', linewidth=2, facecolor = 'skyblue', hatch = '/')
bp3['boxes'][1].set(color='orange', linewidth=2, facecolor = 'olive', hatch = '/')
bp3['boxes'][2].set(color='red', linewidth=2, facecolor = 'pink', hatch = '/')
bp3['boxes'][3].set(color='skyblue', linewidth=2, facecolor = 'pink', hatch = '/')
bp3['boxes'][4].set(color='navy', linewidth=2, facecolor = 'pink', hatch = '/')
plt.xticks([1, 2, 3, 4, 5], ['T0', 'T+2', 'T+4','H+2', 'H+4'])
plt.ylabel('mean annual sensible heat')
plt.savefig('sensibleHeat.png')
plt.close()
#%%annual mean latent heat
ac_tlh = []
for dfs in ac_all:
    ac_tlh.append(pd.DataFrame(dfs['scalarLatHeatTotal'][:]))
ac_tlh_df = pd.concat (ac_tlh, axis=1)
ac_tlh_df.columns = hru_names_df[0]
ac_tlh_df_finale = ac_tlh_df[hru_names_finale]
ac_tlh_mean = []
for hrus in hru_names_finale:
    ac_tlh_mean.append(ac_tlh_df_finale[hrus].mean())
ac_tlh_mean_df = pd.DataFrame(ac_tlh_mean, columns =['mean_yr_latentheat'])
ac_tlh_mean_df.set_index(pd.DataFrame(hru_names_finale)[0],inplace=True)
ac_tlh_mean_df_sc = pd.DataFrame(np.resize(ac_tlh_mean_df['mean_yr_latentheat'], (5,36)).T, columns = ['T0','T2','T4','H2','H4'])

av_tlh = []
for dfs in av_all2:
    av_tlh.append(pd.DataFrame(dfs['scalarLatHeatTotal'][:]))
av_tlh_df = pd.concat (av_tlh, axis=1)
av_tlh_df.columns = Combo_bestModels_swe_v.hru_names_df[0]
av_tlh_df_finale = av_tlh_df[Combo_bestModels_swe_v.hru_names_finale[0]]
av_tlh_mean = []
for hrus in hru_names_finaleV:
    av_tlh_mean.append(av_tlh_df_finale[hrus].mean())
av_tlh_mean_df = pd.DataFrame(av_tlh_mean, columns =['mean_yr_latentheat'])
av_tlh_mean_df.set_index(pd.DataFrame(hru_names_finaleV)[0],inplace=True)
av_tlh_mean_df_sc = pd.DataFrame(np.resize(av_tlh_mean_df['mean_yr_latentheat'], (5,4)).T, columns = ['T0','T2','T4','H2','H4'])

tlh_mean_df = pd.concat([ac_tlh_mean_df_sc, av_tlh_mean_df_sc])

d3 = [tlh_mean_df['T0'],tlh_mean_df['T2'],tlh_mean_df['T4'],tlh_mean_df['H2'],tlh_mean_df['H4']]
bp4 = plt.boxplot(d3, patch_artist=True)
bp4['boxes'][0].set(color='olivedrab', linewidth=2, facecolor = 'skyblue', hatch = '/')
bp4['boxes'][1].set(color='tomato', linewidth=2, facecolor = 'olive', hatch = '/')
bp4['boxes'][2].set(color='crimson', linewidth=2, facecolor = 'pink', hatch = '/')
bp4['boxes'][3].set(color='dodgerblue', linewidth=2, facecolor = 'pink', hatch = '/')
bp4['boxes'][4].set(color='steelblue', linewidth=2, facecolor = 'pink', hatch = '/')
plt.xticks([1, 2, 3, 4, 5], ['T0', 'T+2', 'T+4','H+2', 'H+4'])
plt.ylabel('mean annual latent heat')
plt.savefig('latentHeat.png')
plt.close()


  
#scalarSnowAlbedo
#spectralIncomingDirect
#spectralIncomingDiffuse
#scalarLWNetUbound
#spectralIncomingDirect(1) = SWRadAtm*scalarFractionDirect*Frad_vis                         ! (direct vis)
# spectralIncomingDirect(2) = SWRadAtm*scalarFractionDirect*(1._dp - Frad_vis)               ! (direct nir)
# ! compute diffuse shortwave radiation, in the visible and near-infra-red part of the spectrum
# spectralIncomingDiffuse(1) = SWRadAtm*(1._dp - scalarFractionDirect)*Frad_vis              ! (diffuse vis)
# spectralIncomingDiffuse(2) = SWRadAtm*(1._dp - scalarFractionDirect)*(1._dp - Frad_vis)    ! (diffuse nir)

#scalarLWNetCanopy               => flux_data%var(iLookFLUX%scalarLWNetCanopy)%dat(1),              & ! intent(out): [dp] net longwave radiation at the canopy (W m-2)
# scalarLWNetGround               => flux_data%var(iLookFLUX%scalarLWNetGround)%dat(1),              & ! intent(out): [dp] net longwave radiation at the ground surface (W m-2)
# scalarLWNetUbound               => flux_data%var(iLookFLUX%scalarLWNetUbound)%dat(1),              & ! intent(out): [dp] net longwave radiation at the upper boundary (W m-2)






#d0 = [desiredMaxSweT0,desiredMaxSweT2,desiredMaxSweT4]cadetblue
#d1 = [desiredresDateT0,desiredresDateT2,desiredresDateT4]
#
#bp0 = plt.boxplot(d0, patch_artist=True)
#bp1 = plt.boxplot(d1, patch_artist=True)
#
#bp0['boxes'][0].set(color='red', linewidth=2, facecolor = 'skyblue', hatch = '/')
#bp0['boxes'][1].set(color='orange', linewidth=2, facecolor = 'olive', hatch = '/')
#bp0['boxes'][2].set(color='tan', linewidth=2, facecolor = 'pink', hatch = '/')
#
##plt.hold()
#
#bp1['boxes'][0].set(color='navy', linewidth=2, facecolor = 'skyblue', hatch = '/')
#bp1['boxes'][1].set(color='blue', linewidth=2, facecolor = 'olive', hatch = '/')
#bp1['boxes'][2].set(color='skyblue', linewidth=2, facecolor = 'pink', hatch = '/')
#
#plt.xticks([1, 2, 3], ['T0', 'T+2', 'T+4'])
#plt.savefig('resSwe2.png')



