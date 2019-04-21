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
#%%
date_swe = ['2006-11-01 08:00', '2006-11-30 08:00', '2007-01-01 08:00', '2007-01-30 08:00', '2007-03-05 08:00', '2007-03-12 08:00', 
            '2007-03-19 08:00', '2007-03-26 08:00', '2007-04-02 08:00', '2007-04-18 08:00', '2007-04-23 08:00', '2007-05-02 08:00', 
            '2007-05-09 08:00', '2007-05-16 08:00', '2007-05-23 08:00', '2007-05-30 08:00', '2007-06-06 08:00', 
            
            '2007-12-03 08:00', '2008-01-01 08:00', '2008-01-31 08:00', '2008-03-03 08:00', '2008-03-24 08:00', '2008-04-01 08:00', 
            '2008-04-08 08:00', '2008-04-14 08:00', '2008-04-22 08:00', '2008-04-28 08:00', '2008-05-06 08:00', '2008-05-12 08:00',
            '2008-05-19 08:00', '2008-05-26 08:00', '2008-06-02 08:00', '2008-06-08 08:00', 
            
            '2008-12-02 08:00', '2009-01-01 08:00', '2009-02-01 08:00', '2009-02-28 08:00', '2009-03-09 08:00', '2009-03-16 08:00',
            '2009-03-24 08:00', '2009-03-30 08:00', '2009-04-07 08:00', '2009-04-15 08:00', '2009-04-22 08:00', '2009-04-29 08:00', 
            '2009-05-06 08:00', '2009-05-13 08:00', 
            
            '2009-11-27 08:00', '2009-12-31 08:00', '2010-01-31 08:00', '2010-03-02 08:00', '2010-03-21 08:00', '2010-04-05 08:00',
            '2010-04-12 08:00', '2010-04-20 08:00', '2010-04-26 08:00', '2010-05-03 08:00', '2010-05-11 08:00', '2010-05-17 08:00',
            '2010-05-24 08:00', 
            
            '2010-11-02 08:00', '2010-12-04 08:00', '2011-01-02 08:00', '2011-02-03 08:00', '2011-03-01 08:00', '2011-03-29 08:00',
            '2011-04-06 08:00', '2011-04-11 08:00', '2011-04-22 08:00', '2011-05-03 08:00', '2011-05-12 08:00', '2011-05-23 08:00', 
            '2011-06-01 08:00', '2011-06-08 08:00', '2011-06-14 08:00', 
            
            '2011-11-29 08:00', '2012-01-02 08:00', '2012-02-01 08:00', '2012-03-05 08:00', '2012-03-26 08:00', '2012-04-02 08:00', 
            '2012-04-08 08:00', '2012-04-16 08:00', '2012-04-23 08:00', '2012-04-30 08:00', '2012-05-07 08:00']# 
            
            #'2012-12-01 08:00', '2013-01-02 08:00', '2013-02-04 08:00', '2013-03-01 08:00', '2013-03-22 08:00', '2013-04-01 08:00',
            #'2013-04-10 08:00', '2013-04-16 08:00', '2013-04-22 08:00', '2013-04-30 08:00', '2013-05-06 08:00', '2013-05-13 08:00']

swe_mm = [58,  169, 267, 315, 499, 523, 503, 549, 611, 678, 654, 660, 711, 550, 443, 309, 84, 
          141, 300, 501, 737, 781, 837, 614, 977, 950, 873, 894, 872, 851, 739, 538, 381, 
          133, 380, 456, 564, 512, 568, 626, 627, 715, 772, 764, 699, 698, 389, 
          89,  255, 347, 481, 608, 646, 682, 585, 553, 608, 520, 440, 302,  
          50,  165, 361, 454, 611, 704, 717, 774, 867, 951, 984, 999, 915, 699, 450, 
          130, 188, 290, 494, 542, 425, 433, 453, 413, 283, 150] 
          #55,  182, 305, 419, 481, 489, 508, 569, 624, 528, 405, 325]  

#obs_swe_date = pd.DataFrame (np.column_stack([date_swe,swe_mm]), columns=['date_swe','swe_mm'])
obs_swe = pd.DataFrame (swe_mm, columns=['swe_mm'])
obs_swe.set_index(pd.DatetimeIndex(date_swe),inplace=True)

max_swe_obs = max(obs_swe['swe_mm'])
max_swe_date_obs = obs_swe[obs_swe ['swe_mm']== max_swe_obs].index.tolist()    

#%%
hruidxID = list(np.arange(101,170))
hru_num = np.size(hruidxID)
out_names = ['AvInitial','Avwp2e','Avas2l','Avas3m','Avce2s','Avtc2s','Avtc3t','Avtc4m','Avns2a','Avns3p','Avns4c']
hru_names =[]
for i in out_names:
    hru_names.append(['{}{}'.format(i, j) for j in hruidxID])
hru_names1 = np.reshape(hru_names,(759,1))
hru_names_df = pd.DataFrame (hru_names1)
#%% reading output_swe files
av_ncfiles = ["AvInitial_swampAngel_2007-2008_senatorVariableDecayRate_1.nc", 
              "Avwp2e_swampAngel_2007-2008_senatorVariableDecayRate_1.nc", 
              "Avas2l_swampAngel_2007-2008_senatorVariableDecayRate_1.nc", 
              "Avas3m_swampAngel_2007-2008_senatorVariableDecayRate_1.nc", 
              "Avce2s_swampAngel_2007-2008_senatorVariableDecayRate_1.nc", 
              "Avtc2s_swampAngel_2007-2008_senatorVariableDecayRate_1.nc", 
              "Avtc3t_swampAngel_2007-2008_senatorVariableDecayRate_1.nc", 
              "Avtc4m_swampAngel_2007-2008_senatorVariableDecayRate_1.nc", 
              "Avns2a_swampAngel_2007-2008_senatorVariableDecayRate_1.nc",
              "Avns3p_swampAngel_2007-2008_senatorVariableDecayRate_1.nc", 
              "Avns4c_swampAngel_2007-2008_senatorVariableDecayRate_1.nc"]
av_all = []
for ncfiles in av_ncfiles:
    av_all.append(Dataset(ncfiles))

#for varname in av_all[0].variables.keys():
#    var = av_all[0].variables[varname]
#    print (varname, var.dtype, var.dimensions, var.shape)

av_sd = []
for dfs in av_all:
    av_sd.append(pd.DataFrame(dfs['scalarSnowDepth'][:]))
av_sd_df = pd.concat (av_sd, axis=1)
av_sd_df.columns =  hru_names_df[0]

av_swe = []
for dfs in av_all:
    av_swe.append(pd.DataFrame(dfs['scalarSWE'][:]))
av_swe_df = pd.concat (av_swe, axis=1)
av_swe_df.columns = hru_names_df[0]

#test_ds = Dataset("T0AvTest_swampAngel_2010-2011_senatorVariableDecayRate_1.nc")
#saTime = test_ds.variables['time']
#plt.plot(saTime,av_swe_df['tcs1']) 
#plt.plot(saTime,av_swe_df['tcs2']) 
#plt.plot(saTime,av_swe_df['tcs3']) 
#plt.savefig('test.png')
#%% output time step
TimeSa = av_all[0].variables['time'][:] # get values
t_unitSa = av_all[0].variables['time'].units # get unit  "days since 1950-01-01T00:00:00Z"

try :

    t_cal = av_all[0].variables['time'].calendar

except AttributeError : # Attribute doesn't exist

    t_cal = u"gregorian" # or standard

tvalueSa = num2date(TimeSa, units=t_unitSa, calendar=t_cal)
DateSa = [i.strftime("%Y-%m-%d %H:%M") for i in tvalueSa] # -%d %H:%M to display dates as string #i.strftime("%Y-%m-%d %H:%M")        
#%% day of snow disappearance-final output
#av_sd_df.set_index(pd.DatetimeIndex(DateSb),inplace=True)
#counter = pd.DataFrame(np.arange(0,np.size(av_sd_df['scalarSnowDepth'])),columns=['counter'])
#counter.set_index(av_sd_df.index,inplace=True)
#av_sd_df2 = pd.concat([counter, av_sd_df], axis=1)
#
#av_swe_df.set_index(pd.DatetimeIndex(DateSb),inplace=True)
#counter = pd.DataFrame(np.arange(0,np.size(av_swe_df['scalarSWE'])),columns=['counter'])
#counter.set_index(av_swe_df.index,inplace=True)
#av_swe_df2 = pd.concat([counter, av_swe_df], axis=1)
#%%   
#av_sd_df5000 = av_sd_df2[:][5000:8737]
#
#zerosnowdate = []
#for val in hru_names_df[0]:
#    zerosnowdate.append(np.where(av_sd_df5000[val]==0))
#zerosnowdate_omg = [item[0] for item in zerosnowdate] #if item[0] == 1]  
#for i,item in enumerate(zerosnowdate_omg):
#    if len(item) == 0:
#        zerosnowdate_omg[i] = 3737
#for i,item in enumerate(zerosnowdate_omg):
#    zerosnowdate_omg[i] = zerosnowdate_omg[i]+5000
#        
#first_zerosnowdate =[]
#for i,item in enumerate(zerosnowdate_omg):
#    if np.size(item)>1:
#        #print np.size(item)
#        first_zerosnowdate.append(item[0])
#    if np.size(item)==1:
#        first_zerosnowdate.append(item)

##zerosnowdate_dif_obs = pd.DataFrame((first_zerosnowdate - dayofsnowdisappearance_obs)/24, columns=['resSnowDisDate'])
##zerosnowdate_dif_obs.set_index(hru_names_df[0],inplace=True)
##%% finding max snowdepth and swe
##max_swe=[]
##for hrus in hru_names_df[0]:
##    max_swe.append(av_swe_df2[hrus].max())
##max_residual_SWE = max_swe - max_swe_obs
##
##swe_corsp_max2date = []
##for hrus in hru_names_df[0]:
##    swe_corsp_max2date.append(av_swe_df2[hrus][max_swe_date_obs])
##max_residual_swe_corsp = pd.DataFrame((swe_corsp_max2date - max_swe_obs), columns=['resCorspMaxSWE'])
##max_residual_swe_corsp.set_index(hru_names_df[0],inplace=True)
##residual_df = pd.concat([zerosnowdate_dif_obs,max_residual_swe_corsp], axis=1)
##residual_df_finale = residual_df.drop(['T0AvASsSWcTCj11112.0', 'T0AvASsSWcTCj11113.0','T2AvASsSWcTCj11112.0', 'T2AvASsSWcTCj11113.0','T4AvASsSWcTCj11112.0', 'T4AvASsSWcTCj11113.0',
##                                       'H2AvASsSWcTCj11112.0', 'H2AvASsSWcTCj11113.0','H4AvASsSWcTCj11112.0', 'H4AvASsSWcTCj11113.0'])
##    
##av_sd_df_finale = av_sd_df2.drop(['T0AvASsSWcTCj11112.0', 'T0AvASsSWcTCj11113.0','T2AvASsSWcTCj11112.0', 'T2AvASsSWcTCj11113.0','T4AvASsSWcTCj11112.0', 'T4AvASsSWcTCj11113.0',
##                                  'H2AvASsSWcTCj11112.0', 'H2AvASsSWcTCj11113.0','H4AvASsSWcTCj11112.0', 'H4AvASsSWcTCj11113.0'], axis=1)
##av_swe_df_finale = av_swe_df2.drop(['T0AvASsSWcTCj11112.0', 'T0AvASsSWcTCj11113.0','T2AvASsSWcTCj11112.0', 'T2AvASsSWcTCj11113.0','T4AvASsSWcTCj11112.0', 'T4AvASsSWcTCj11113.0',
##                                    'H2AvASsSWcTCj11112.0', 'H2AvASsSWcTCj11113.0','H4AvASsSWcTCj11112.0', 'H4AvASsSWcTCj11113.0'], axis=1)
#
##%%
#tvalueSb = num2date(TimeSb, units=t_unitSb, calendar=t_cal)
#DateSb2 = [i.strftime("%Y-%m") for i in tvalueSb]
#
#sbx = np.arange(0,np.size(DateSb2))
#sb_xticks = DateSb2
#sbfig, sbax = plt.subplots(1,1)
#plt.xticks(sbx, sb_xticks[::1000], rotation=25)
#sbax.xaxis.set_major_locator(ticker.AutoLocator())
##%%
#count = 1
#for jj in range (5):
#    if (jj+count)<=20 :
#        #av_swe_plot = av_swe_df_finale[av_swe_df_finale.columns[count]]
#        plt.plot(av_swe_df_finale[av_swe_df_finale.columns[jj+count]])
#        count = count + 3
#        print count
#    
#plt.plot(obs_swe['swe_mm'], 'ko', linewidth=1) 
#plt.title('SWE_AvASsSWcTCj11111')
#plt.legend(['T0','T2','T4','H2','H4','obs'])  
#plt.xlabel('Time 2010-2011')
#plt.ylabel('SWE')
##plt.show()
#plt.savefig('SWE_AvASsSWcTCj11111.png')
#plt.close()
##%%
#count = 2
#for jj in range (5):
#    if (jj+count)<=20 :
#        #av_swe_plot = av_swe_df_finale[av_swe_df_finale.columns[count]]
#        plt.plot(av_swe_df_finale[av_swe_df_finale.columns[jj+count]])
#        count = count + 3
#           
#plt.plot(obs_swe['swe_mm'], 'ko', linewidth=1) 
#plt.title('SWE_AvASsSWcTCs11111')
#plt.legend(['T0','T2','T4','H2','H4','obs'])  
#plt.xlabel('Time 2010-2011')
#plt.ylabel('SWE')
##plt.show()
#plt.savefig('SWE_AvASsSWcTCs11111.png')
#plt.close()
##%%
#count = 3
#for jj in range (5):
#    if (jj+count)<=20 :
#        #av_swe_plot = av_swe_df_finale[av_swe_df_finale.columns[count]]
#        plt.plot(av_swe_df_finale[av_swe_df_finale.columns[jj+count]])
#        count = count + 3
#            
#plt.plot(obs_swe['swe_mm'], 'ko', linewidth=1) 
#plt.title('SWE_AvASsSWcTCs11112')
#plt.legend(['T0','T2','T4','H2','H4','obs'])  
#plt.xlabel('Time 2010-2011')
#plt.ylabel('SWE')
##plt.show()
#plt.savefig('SWE_AvASsSWcTCs11112.png')
#plt.close()
##%%
#count = 4
#for jj in range (5):
#    if (jj+count)<=20 :
#        #av_swe_plot = av_swe_df_finale[av_swe_df_finale.columns[count]]
#        plt.plot(av_swe_df_finale[av_swe_df_finale.columns[jj+count]])
#        count = count + 3
#            
#plt.plot(obs_swe['swe_mm'], 'ko', linewidth=1) 
#plt.title('SWE_AvASsSWcTCs11113')
#plt.legend(['T0','T2','T4','H2','H4','obs'])  
#plt.xlabel('Time 2010-2011')
#plt.ylabel('SWE')
##plt.show()
#plt.savefig('SWE_AvASsSWcTCs11113.png')
#plt.close()

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

