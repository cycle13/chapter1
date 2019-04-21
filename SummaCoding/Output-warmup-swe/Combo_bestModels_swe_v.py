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

paramID = hru_ix_ID(p1, p2, p3, p4, p5)
model_names = ['AvASsSWcTCj','AvASsSWcTCs']
senario_names = ['T0','T2','T4','H2','H4']
hru_names =[]
for i in senario_names:
    for j in model_names:
        hru_names.append(['{}{}{}'.format(i, j, k) for k in paramID])

hru_names1 = np.reshape(hru_names,(30,1))
hru_names_df = pd.DataFrame (hru_names1)
hru_names_finale = hru_names_df.drop(hru_names_df.index[[1,2,7,8,13,14,19,20,25,26]])
hru_names_finale.reset_index(drop=True, inplace=True)
#hru_num = np.size(paramID)
#paramID_df = pd.DataFrame (paramID)
#%% reading output_swe files
av_ncfiles = ["T0AvASsSWcTCj_2010-2011_senatorVariableDecayRate_1.nc",
              "T0AvASsSWcTCs_2010-2011_senatorVariableDecayRate_1.nc",
              "T2AvASsSWcTCj_2010-2011_senatorVariableDecayRate_1.nc",
              "T2AvASsSWcTCs_2010-2011_senatorVariableDecayRate_1.nc",
              "T4AvASsSWcTCj_2010-2011_senatorVariableDecayRate_1.nc",
              "T4AvASsSWcTCs_2010-2011_senatorVariableDecayRate_1.nc",
              "H2AvASsSWcTCj_2010-2011_senatorVariableDecayRate_1.nc",
              "H2AvASsSWcTCs_2010-2011_senatorVariableDecayRate_1.nc",
              "H4AvASsSWcTCj_2010-2011_senatorVariableDecayRate_1.nc",
              "H4AvASsSWcTCs_2010-2011_senatorVariableDecayRate_1.nc"]
av_all = []
for ncfiles in av_ncfiles:
    av_all.append(Dataset(ncfiles))

for varname in av_all[0].variables.keys():
    var = av_all[0].variables[varname]
    print (varname, var.dtype, var.dimensions, var.shape)

av_sd = []
for dfs in av_all:
    av_sd.append(pd.DataFrame(dfs['scalarSnowDepth'][:]))
av_sd_df = pd.concat (av_sd, axis=1)
av_sd_df.columns = hru_names_df[0]
#for colnames in range (30):
#    av_sd_df.rename(columns={av_sd_df.columns[colnames]: hru_names_df[0][colnames]}, inplace=True)
av_swe = []
for dfs in av_all:
    av_swe.append(pd.DataFrame(dfs['scalarSWE'][:]))
av_swe_df = pd.concat (av_swe, axis=1)
av_swe_df.columns = hru_names_df[0]
#%% output time step
TimeSb = av_all[0].variables['time'][:] # get values
t_unitSb = av_all[0].variables['time'].units # get unit  "days since 1950-01-01T00:00:00Z"

try :

    t_cal = av_all[0].variables['time'].calendar

except AttributeError : # Attribute doesn't exist

    t_cal = u"gregorian" # or standard

tvalueSb = num2date(TimeSb, units=t_unitSb, calendar=t_cal)
DateSb = [i.strftime("%Y-%m-%d %H:%M") for i in tvalueSb] # -%d %H:%M to display dates as string #i.strftime("%Y-%m-%d %H:%M")        
#%% day of snow disappearance-final output
av_sd_df.set_index(pd.DatetimeIndex(DateSb),inplace=True)
counter = pd.DataFrame(np.arange(0,np.size(av_sd_df['T0AvASsSWcTCj11111.0'])),columns=['counter'])
counter.set_index(av_sd_df.index,inplace=True)
av_sd_df2 = pd.concat([counter, av_sd_df], axis=1)

av_swe_df.set_index(pd.DatetimeIndex(DateSb),inplace=True)
counter = pd.DataFrame(np.arange(0,np.size(av_swe_df['T0AvASsSWcTCj11111.0'])),columns=['counter'])
counter.set_index(av_swe_df.index,inplace=True)
av_swe_df2 = pd.concat([counter, av_swe_df], axis=1)
#%%   
av_sd_df5000 = av_sd_df2[:][5000:8737]

zerosnowdate = []
for val in hru_names_df[0]:
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

zerosnowdate_dif_obs = pd.DataFrame((first_zerosnowdate - dayofsnowdisappearance_obs)/24, columns=['resSnowDisDate'])
zerosnowdate_dif_obs.set_index(hru_names_df[0],inplace=True)
#%% finding max snowdepth and swe
max_swe=[]
for hrus in hru_names_df[0]:
    max_swe.append(av_swe_df2[hrus].max())
max_residual_SWE = max_swe - max_swe_obs

swe_corsp_max2date = []
for hrus in hru_names_df[0]:
    swe_corsp_max2date.append(av_swe_df2[hrus][max_swe_date_obs])
max_residual_swe_corsp = pd.DataFrame((swe_corsp_max2date - max_swe_obs), columns=['resCorspMaxSWE'])
max_residual_swe_corsp.set_index(hru_names_df[0],inplace=True)
residual_df = pd.concat([zerosnowdate_dif_obs,max_residual_swe_corsp], axis=1)
residual_df_finale = residual_df.drop(['T0AvASsSWcTCj11112.0', 'T0AvASsSWcTCj11113.0','T2AvASsSWcTCj11112.0', 'T2AvASsSWcTCj11113.0','T4AvASsSWcTCj11112.0', 'T4AvASsSWcTCj11113.0',
                                       'H2AvASsSWcTCj11112.0', 'H2AvASsSWcTCj11113.0','H4AvASsSWcTCj11112.0', 'H4AvASsSWcTCj11113.0'])
    
av_sd_df_finale = av_sd_df2.drop(['T0AvASsSWcTCj11112.0', 'T0AvASsSWcTCj11113.0','T2AvASsSWcTCj11112.0', 'T2AvASsSWcTCj11113.0','T4AvASsSWcTCj11112.0', 'T4AvASsSWcTCj11113.0',
                                  'H2AvASsSWcTCj11112.0', 'H2AvASsSWcTCj11113.0','H4AvASsSWcTCj11112.0', 'H4AvASsSWcTCj11113.0'], axis=1)
av_swe_df_finale = av_swe_df2.drop(['T0AvASsSWcTCj11112.0', 'T0AvASsSWcTCj11113.0','T2AvASsSWcTCj11112.0', 'T2AvASsSWcTCj11113.0','T4AvASsSWcTCj11112.0', 'T4AvASsSWcTCj11113.0',
                                    'H2AvASsSWcTCj11112.0', 'H2AvASsSWcTCj11113.0','H4AvASsSWcTCj11112.0', 'H4AvASsSWcTCj11113.0'], axis=1)

#%%
tvalueSb = num2date(TimeSb, units=t_unitSb, calendar=t_cal)
DateSb2 = [i.strftime("%Y-%m") for i in tvalueSb]

sbx = np.arange(0,np.size(DateSb2))
sb_xticks = DateSb2
sbfig, sbax = plt.subplots(1,1)
plt.xticks(sbx, sb_xticks[::1000], rotation=25)
sbax.xaxis.set_major_locator(ticker.AutoLocator())
#%%
count = 1
for jj in range (5):
    if (jj+count)<=20 :
        #av_swe_plot = av_swe_df_finale[av_swe_df_finale.columns[count]]
        plt.plot(av_swe_df_finale[av_swe_df_finale.columns[jj+count]])
        count = count + 3
        print count
    
plt.plot(obs_swe['swe_mm'], 'ko', linewidth=1) 
plt.title('SWE_AvASsSWcTCj11111')
plt.legend(['T0','T2','T4','H2','H4','obs'])  
plt.xlabel('Time 2010-2011')
plt.ylabel('SWE')
#plt.show()
plt.savefig('SWE_AvASsSWcTCj11111.png')
plt.close()
#%%
count = 2
for jj in range (5):
    if (jj+count)<=20 :
        #av_swe_plot = av_swe_df_finale[av_swe_df_finale.columns[count]]
        plt.plot(av_swe_df_finale[av_swe_df_finale.columns[jj+count]])
        count = count + 3
           
plt.plot(obs_swe['swe_mm'], 'ko', linewidth=1) 
plt.title('SWE_AvASsSWcTCs11111')
plt.legend(['T0','T2','T4','H2','H4','obs'])  
plt.xlabel('Time 2010-2011')
plt.ylabel('SWE')
#plt.show()
plt.savefig('SWE_AvASsSWcTCs11111.png')
plt.close()
#%%
count = 3
for jj in range (5):
    if (jj+count)<=20 :
        #av_swe_plot = av_swe_df_finale[av_swe_df_finale.columns[count]]
        plt.plot(av_swe_df_finale[av_swe_df_finale.columns[jj+count]])
        count = count + 3
            
plt.plot(obs_swe['swe_mm'], 'ko', linewidth=1) 
plt.title('SWE_AvASsSWcTCs11112')
plt.legend(['T0','T2','T4','H2','H4','obs'])  
plt.xlabel('Time 2010-2011')
plt.ylabel('SWE')
#plt.show()
plt.savefig('SWE_AvASsSWcTCs11112.png')
plt.close()
#%%
count = 4
for jj in range (5):
    if (jj+count)<=20 :
        #av_swe_plot = av_swe_df_finale[av_swe_df_finale.columns[count]]
        plt.plot(av_swe_df_finale[av_swe_df_finale.columns[jj+count]])
        count = count + 3
            
plt.plot(obs_swe['swe_mm'], 'ko', linewidth=1) 
plt.title('SWE_AvASsSWcTCs11113')
plt.legend(['T0','T2','T4','H2','H4','obs'])  
plt.xlabel('Time 2010-2011')
plt.ylabel('SWE')
#plt.show()
plt.savefig('SWE_AvASsSWcTCs11113.png')
plt.close()

#d1 = [desiredresDateT0,desiredresDateT2,desiredresDateT4]
#bp1 = plt.boxplot(d1, patch_artist=True)
#bp1['boxes'][0].set(color='navy', linewidth=2, facecolor = 'skyblue', hatch = '/')
#bp1['boxes'][1].set(color='blue', linewidth=2, facecolor = 'olive', hatch = '/')
#bp1['boxes'][2].set(color='skyblue', linewidth=2, facecolor = 'pink', hatch = '/')
#
#plt.xticks([1, 2, 3], ['T0', 'T+2', 'T+4'])
#plt.savefig('resDate2.png')
#
#d0 = [desiredMaxSweT0,desiredMaxSweT2,desiredMaxSweT4]
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

