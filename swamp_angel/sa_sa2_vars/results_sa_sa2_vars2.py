###       /bin/bash runTestCases_docker.sh
import numpy as np
import matplotlib.pyplot as plt 
from netCDF4 import Dataset,netcdftime,num2date
from datetime import datetime
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from sklearn.metrics import mean_squared_error
import itertools
import csv

# defining functions  

def mySubtract(myList,num):
    return list(np.subtract(myList,num))
def myMultiply(myList,num):
    return list(np.multiply(myList,num))
def sum2lists (list1,list2):
    return list(np.add(list1,list2))

def readAllNcfilesAsDataset(allNcfiles):
    allNcfilesDataset = []
    for ncfiles in allNcfiles:
        allNcfilesDataset.append(Dataset(ncfiles))
    return allNcfilesDataset

def readVariablefromMultipleNcfilesDatasetasDF(av_all,variable,hruname_df,hruidxID,out_names):
    
    variableList = []
    for ds in range(len(hruname_df)/len(hruidxID)): # 22/2
        variableNameList = []
        variableNameList_year1 = av_all[2*ds][variable][:]
        variableNameList_year2 = av_all[2*ds+1][variable][:]
        variableNameList = np.append(variableNameList_year1, variableNameList_year2, axis =0)
        variableList.append(variableNameList)
    
    list_col = []
    for i in range (len(out_names)):
        for j in range (len(hruidxID)):
            column = variableList[i][:,j]
            list_col.append(column)
    
    variable_arr = np.array(list_col).T
    
    variable_df = pd.DataFrame(variable_arr, columns = hruname_df[0])
    counter = pd.DataFrame(np.arange(0,np.size(variable_df[hruname_df[0][0]])),columns=['counter'])    
    counter.set_index(variable_df.index,inplace=True)
    variable_df = pd.concat([counter, variable_df], axis=1)
    
    return variable_df

def readVariablefromMultipleNcfilesDatasetasDF40neYear(av_all,variable,hruname_df,hruidxID,out_names):
    
    variableList = []
    for ds in range(len(hruname_df)/len(hruidxID)): # 22/2
        variableNameList = []
        variableNameList_year1 = av_all[ds][variable][:]
        variableList.append(variableNameList_year1)
    
    list_col = []
    for i in range (len(out_names)):
        for j in range (len(hruidxID)):
            column = variableList[i][:,j]
            list_col.append(column)
    
    variable_arr = np.array(list_col).T
    
    variable_df = pd.DataFrame(variable_arr, columns = hruname_df[0])
    counter = pd.DataFrame(np.arange(0,np.size(variable_df[hruname_df[0][0]])),columns=['counter'])    
    counter.set_index(variable_df.index,inplace=True)
    variable_df = pd.concat([counter, variable_df], axis=1)
    
    return variable_df


def readVariablefromMultipleNcfilesDatasetasDF4AYear(av_all,variable,hruname_df,hruidxID,out_names):

    variableList = []
    for ds in range(len(hruname_df)/len(hruidxID)): # 22/2
        variableNameList = []
        variableNameList_year1 = av_all[2*ds][variable][:]
        variableNameList_year2 = av_all[2*ds+1][variable][:]
        variableNameList = np.append(variableNameList_year1, variableNameList_year2, axis =0)
        variableList.append(variableNameList)
    
    list_col = []
    for i in range (len(out_names)):
        for j in range (len(hruidxID)):
            column = variableList[i][:,j]
            list_col.append(column)
    
    variable_arr = np.array(list_col).T
    
    variable_df = pd.DataFrame(variable_arr, columns = hruname_df[0])
    counter = pd.DataFrame(np.arange(0,np.size(variable_df[hruname_df[0][0]])),columns=['counter'])    
    counter.set_index(variable_df.index,inplace=True)
    variable_df = pd.concat([counter, variable_df], axis=1)
    
    return variable_df

def readSomePartofVariableDatafromNcfilesDatasetasDF(NcfilesDataset,variable,hruname,paretoNameDF):
    variableNameList = []
    for datasets in NcfilesDataset:
        variableNameList.append(pd.DataFrame(datasets[variable][:]))
    variableDF = pd.concat (variableNameList, axis=1)
    variableDF.columns = hruname
    desiredDataframe = []
    for paretos in paretoNameDF:
        desiredDataframe.append(variableDF[paretos])
    return desiredDataframe

def date(allNcfilesDataset,formatDate):
    Time = allNcfilesDataset.variables['time'][:] 
    #TimeSa = np.concatenate((TimeSa2006, TimeSa2007), axis=0)
    t_unit = allNcfilesDataset.variables['time'].units 
    
    try :
        t_cal = allNcfilesDataset.variables['time'].calendar
    
    except AttributeError : # Attribute doesn't exist error
        t_cal = u"gregorian" # or standard
    #tvalueSa = num2date(TimeSa, units=t_unitSa, calendar=t_cal)
    tvalue = num2date(Time, units=t_unit, calendar=t_cal)
    Date = [i.strftime(formatDate) for i in tvalue] # "%Y-%m-%d %H:%M"
    return Date

def readSpecificDatafromAllHRUs(variablename,hruname,day):
    dayData = []
    for names in hruname:
        dayData.append(variablename[names][day])
    return dayData

def sumBeforeSpecificDatafromAllHRUs(variablename,hruname,day):
    sumData = []
    for names in hruname:
        sumData.append(sum(variablename[names][0:day]))
    return sumData   

def snowLayerAttributeforSpecificDate(layerattributefile,hruname,sumlayer,snowlayer): #like snowlayertemp, volFracosIce, ....
    snowlayerattribute = []
    for names in hruname:
        snowlayerattribute.append(list(layerattributefile[names][sumlayer[names][0]:sumlayer[names][0]+snowlayer[names][0]]))
    return snowlayerattribute

def depthOfLayers(heightfile):
    finalHeight = []
    for lsts in heightfile:
        sumSofar=0
        lstscopy= lsts[:]
        lstscopy.reverse()
        height_ls = []
        for values in lstscopy:
            height=2*(abs(values)-sumSofar)
            height_ls.append(height)
            sumSofar+=height
        #print ("original:", height_ls) 
        height_ls.reverse()
        #print ("after reverse:", height_ls)
        finalHeight.append(height_ls)
    return finalHeight

def coldContentFunc(hruname,volfracLiq,volfracIce,snowlayertemp,layerHeight):
    densityofWater = 997. #kg/m³
    densityofIce = 917. #kg/m3
    heatCapacityIce2 = -2102./1000000. #Mj kg-1 m3-1
    coldcontent = []
    for nlst in range (np.size(hruname)):
        swe = np.array(sum2lists(myMultiply(volfracLiq[nlst],densityofWater/1000.),myMultiply(volfracIce[nlst],densityofIce/1000.))) #[swe] = m
        temp = np.array(mySubtract(snowlayertemp[nlst],273.2))
        HCItHS = np.array(myMultiply(heatCapacityIce2,layerHeight[nlst]))
        cct = sum(list(swe*temp*HCItHS))
        coldcontent.append(cct)
    return coldcontent

def SWEandSWEDateforSpecificDate(hruname,hour,swe_df,dosd):
    SWE = []
    SWEdate = []
    for names in hruname:
        if swe_df[names][hour]>0:
            SWE.append(swe_df[names][hour])
            SWEdate.append(hour)
        else: 
            SWE.append(float(swe_df[names][dosd[names]-1]))
            SWEdate.append(float(dosd[names]-1))
    return SWE,SWEdate

def meltingRateBetween2days(swe1,swe2,sweDate1,sweDate2):
    mdeltaday = []; mdeltaSWE = []; meltingrate = []; #cm/day
    for counterhd in range (np.size(swe1)):
        mdeltaday.append(float(sweDate2[counterhd]-sweDate1[counterhd]))
        mdeltaSWE.append(float(swe1[counterhd]-swe2[counterhd]))
        if mdeltaday[counterhd]==0:
            meltingrate.append(float(0))
        else: meltingrate.append(float(0.1*24*mdeltaSWE[counterhd]/mdeltaday[counterhd]))
    return meltingrate

def calculatingSDD(av_sd_df,hru_names_df,sdd_obs,year,resNum1,resNum2):
    av_sd_df5000 = av_sd_df

    zerosnowdate = []
    for val in hru_names_df[0]:
        zerosnowdate.append(np.where(av_sd_df5000[val]==0))
    zerosnowdate_omg = [item[0] for item in zerosnowdate] #change tuple to array
    for i,item in enumerate(zerosnowdate_omg):
        if len(item) == 0:
            zerosnowdate_omg[i] = resNum1
    for i,item in enumerate(zerosnowdate_omg):
        zerosnowdate_omg[i] = zerosnowdate_omg[i]+resNum2
            
    first_zerosnowdate =[]
    for i,item in enumerate(zerosnowdate_omg):
        if np.size(item)>1:
            #print np.size(item)
            first_zerosnowdate.append(item[0])
        if np.size(item)==1:
            first_zerosnowdate.append(item)
        
    #first_zerosnowdate_df = pd.DataFrame(np.reshape(first_zerosnowdate, ((np.size(hru_names1)),0)).T, columns=out_names)
    dosd_df = pd.DataFrame(np.array(first_zerosnowdate)).T
    dosd_df.columns = hru_names_df[0]
    #first_zerosnowdate_df_obs = pd.DataFrame(np.array([[5985],[6200]]).T,columns=out_names)
    dosd_obs = pd.DataFrame(np.array([sdd_obs]),columns=[year])
    
    dosd_residual=[]
    for hru in dosd_df.columns:
        dosd_residual.append((dosd_df[hru][0]-dosd_obs[year])/24)
    
    dosd_residual_df = pd.DataFrame(np.reshape(np.array(dosd_residual),(np.size(out_names),hru_num)).T, columns=out_names)

    return dosd_df, dosd_residual_df
#%% SWE observation data 
date_swe = ['2006-11-01 11:10','2006-11-30 12:30','2007-01-01 11:10','2007-01-30 10:35','2007-03-05 14:30','2007-03-12 14:00', 
            '2007-03-19 12:30','2007-03-26 12:30','2007-04-02 12:30','2007-04-18 08:35','2007-04-23 10:30','2007-05-02 08:40', 
            '2007-05-09 08:50','2007-05-16 09:00','2007-05-23 08:30','2007-05-30 09:00','2007-06-06 08:15', 
            
            '2007-12-03 10:45','2008-01-01 11:30','2008-01-31 12:00','2008-03-03 14:30','2008-03-24 09:10','2008-04-01 09:55', 
            '2008-04-14 14:45','2008-04-22 12:30','2008-04-28 12:30','2008-05-06 09:15','2008-05-12 12:45','2008-05-19 10:40',
            '2008-05-26 08:45','2008-06-02 12:45','2008-06-08 08:45'] 
            
swe_mm = [58,  169, 267, 315, 499, 523, 503, 549, 611, 678, 654, 660, 711, 550, 443, 309, 84, 
          141, 300, 501, 737, 781, 837, 977, 950, 873, 894, 872, 851, 739, 538, 381]  

#obs_swe_date = pd.DataFrame (np.column_stack([date_swe,swe_mm]), columns=['date_swe','swe_mm'])
obs_swe = pd.DataFrame (swe_mm, columns=['swe_mm'])
obs_swe.set_index(pd.DatetimeIndex(date_swe),inplace=True)

max_swe_obs = max(obs_swe['swe_mm'])
max_swe_date_obs = obs_swe[obs_swe ['swe_mm']== max_swe_obs].index.tolist()  

swe_obs2007 = pd.DataFrame (obs_swe['swe_mm']['2006-11-01':'2007-06-06'], columns=['swe_mm'])
swe_obs2008 = pd.DataFrame (obs_swe['swe_mm']['2007-12-03':'2008-06-08'], columns=['swe_mm'])
#date_swe_day = ['2006-11-01','2006-11-30','2007-01-01','2007-01-30','2007-03-05', 
#                '2007-03-12','2007-03-19','2007-03-26','2007-04-02','2007-04-18', 
#                '2007-04-23','2007-05-02','2007-05-09','2007-05-16','2007-05-23',
#                '2007-05-30','2007-06-06',
#                '2007-12-03','2008-01-01','2008-01-31','2008-03-03','2008-03-24','2008-04-01', 
#                '2008-04-14','2008-04-22','2008-04-28','2008-05-06','2008-05-12','2008-05-19',
#                '2008-05-26','2008-06-02','2008-06-08'] 
#date_swe_day_dt = pd.DatetimeIndex(date_swe_day)
#date_swe_day_dt.strftime('%Y-%m-%d')
#obs_swe.set_index(pd.DatetimeIndex(date_swe_day_dt),inplace=True)  
#obs_swe_ind = np.array([749,1441,2209,2905,3721,3889,4057,4225,4393,4777,4897,5113,5281,5449,
#                        5617,5785,5953,10273,10969,11689,12457,12967,13153,13465,13657,13801,13993,14137,
#                        14306,14473,14641,14785])
obs_swe_ind = np.array([760,1455,2220,2920,3735,3900,4070,4240,4405,4790,4910,5125,5295,5460,
                        5630,5799,5965,10290,10980,11700,12470,12980,13165,13488,13670,13815,
                        14010,14150,14320,14485,14655,14800])
obs_swe.set_index(obs_swe_ind,inplace=True)
#%% Snow depth observation data
with open("C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/snowDepth_2006_2008_2.csv") as safd1:
    reader1 = csv.reader(safd1)
    raw_snowdepth1 = [r for r in reader1]
sa_snowdepth_column1 = []
for csv_counter1 in range (len (raw_snowdepth1)):
    for csv_counter2 in range (3):
        sa_snowdepth_column1.append(raw_snowdepth1[csv_counter1][csv_counter2])
sa_snowdepth=np.reshape(sa_snowdepth_column1,(len (raw_snowdepth1),3))
sa_sd_obs=[float(val) for val in sa_snowdepth[1:len(raw_snowdepth1)-1,2:]]

#sa_sd_obs = [float(value) for value in sa_snowdepth1]
sa_sd_obs_date = pd.DatetimeIndex(sa_snowdepth[1:len(raw_snowdepth1)-1,0])

snowdepth_obs_df = pd.DataFrame(sa_sd_obs, columns = ['observed_snowdepth']) 
snowdepth_obs_df.set_index(sa_sd_obs_date,inplace=True)

snowdepth_obs_df2 = pd.DataFrame(sa_sd_obs, columns = ['observed_snowdepth']) 
indx_sd = np.arange(0,16056,24)
snowdepth_obs_df2.set_index(indx_sd,inplace=True)
#%% defining hrus_name
hruidxID = list(np.arange(10000,10954))    
hru_num = np.size(hruidxID)
years = ['2007','2008']#
out_names = ["ljh","ljp","ljc","ltc"]#"lth","ltp","sth","stc","sjc","sjp"
                       
paramModel = (np.size(out_names))*(hru_num)
hru_names =[]
for i in out_names:
    hru_names.append(['{}{}'.format(j, i) for j in hruidxID])
hru_names1 = np.reshape(hru_names,(paramModel,1))
hru_names_df = pd.DataFrame (hru_names1)

#%%  reading output files
from allNcFiles_sa2 import av_ncfiles

av_all = readAllNcfilesAsDataset(av_ncfiles)
#DateSa2007 = date(av_all[0],"%Y-%m-%d %H:%M")
#DateSa2008 = date(av_all[1],"%Y-%m-%d %H:%M")

#av_swe_df = readVariablefromMultipleNcfilesDatasetasDF40neYear(av_all,'scalarSWE',hru_names_df,hruidxID,out_names)

av_sd_df = readVariablefromMultipleNcfilesDatasetasDF(av_all,'scalarSnowDepth',hru_names_df,hruidxID,out_names)
av_swe_df = readVariablefromMultipleNcfilesDatasetasDF(av_all,'scalarSWE',hru_names_df,hruidxID,out_names)


#%% ploting annual swe curves 

DateSa21 = date(av_all[0],"%Y-%m-%d")
DateSa22 = date(av_all[1],"%Y-%m-%d")
date_sa = np.append(DateSa21,DateSa22)
#av_swe_df.index = date_sa
safig, saax = plt.subplots(1,1, figsize=(30,20))#

for hru in range (len(hru_names_df[0])):#len(hru_names_df[0]),
    #print hru_names_df[0][hru]
    plt.plot(av_swe_df[hru_names_df[0][hru]], linewidth=4)

saax.plot(obs_swe.index, obs_swe , 'ok', markersize=15)#[0:16]

plt.yticks(fontsize=40)
plt.xlabel('Time 2006-2008', fontsize=40)
saax.set_ylabel('SWE(mm)', fontsize=40)
#saax.legend('SWE(mm)',fontsize = 30)#, loc = 'upper left'

sax = np.arange(0,np.size(date_sa))
sa_xticks = date_sa
plt.xticks(sax[::3000], sa_xticks[::3000], rotation=25, fontsize=40)# 

safig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa_sa2_VARs_p12FPM_fTCs/sa_sa2_swe_4c.png')

#%% ploting snow depth 
safig2, saax2 = plt.subplots(1,1, figsize=(30,20))#

for hru2 in range (len(hru_names_df[0])):#
    #print hru_names_df[0][hru]
    plt.plot(av_sd_df[hru_names_df[0][hru2]], linewidth=4)

plt.plot(snowdepth_obs_df2.index, snowdepth_obs_df2['observed_snowdepth'], 'k', linewidth=6)#[0:16], markersize=15

plt.yticks(fontsize=40)
plt.xlabel('Time 2006-2008', fontsize=40)
saax2.set_ylabel('snow depth (mm)', fontsize=40)
plt.xticks(sax[::3000], sa_xticks[::3000], rotation=25, fontsize=40)# 

safig2.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa_sa2_VARs_p12FPM_fTCs/sa_sa2_sd_4c.png')

#%% day of snow disappearance (based on snowdepth)-final output
dosd_df2007, dosd_residual_df2007 = calculatingSDD(av_sd_df[:][5000:8737],hru_names_df,5976,'2007',3737,5000)
dosd_df2008, dosd_residual_df2008 = calculatingSDD(av_sd_df[:][14000:],hru_names_df,15000,'2008',3521,14000)

dosd_residual_df2007min = np.min(abs(dosd_residual_df2007))

dosd_residual_df2007min20 = []
dosd_residual_df2007min20_indx = []
for combo in out_names:
    dosdr_20min = abs(dosd_residual_df2007).nsmallest(20,combo)[combo]
    dosd_residual_df2007min20.append(pd.DataFrame(dosdr_20min))
    dosdr_20min_indx = dosdr_20min.index
    dosd_residual_df2007min20_indx.append(dosdr_20min_indx.values)

#plt.xticks(x, hru[::3], rotation=25)
for models in range(len(out_names)):
    x = list(hruidxID)
    fig = plt.figure(figsize=(20,15))
    plt.bar(x,dosd_residual_df2007[out_names[models]])
    plt.title(out_names[models], fontsize=42)
    plt.xlabel('hrus',fontsize=30)
    plt.ylabel('residual SDD (day)', fontsize=30)
    plt.yticks(fontsize=30)
    plt.xticks(fontsize=30)
    plt.savefig('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/figs/SDD_'+out_names[models])
#%%**************************************************************************************************
# *********************** finding max corespondance swe for '2007-05-09 08:50'***********************
#'2007-04-18' 4776: 4800, '2007-04-23' 4896:4920, '2007-05-02' 5112:5136
#Group1: '2007-03-12 14:00' (3902),'2007-03-19 12:30 (4068)','2007-03-26 12:30 (4236)','2007-04-02 12:30'(4404),
#Group2: '2007-04-18 08:35' (4784),'2007-04-23 10:30 (4907)','2007-05-02 08:40'(5121), 

maxSWE2007 = readSpecificDatafromAllHRUs(av_swe_df,hru_names_df[0],5289)
maxSWE_obs2007 = [711]  
maxSWE_residual2007 = (pd.DataFrame(mySubtract(maxSWE2007,maxSWE_obs2007))).T
maxSWE_residual2007.columns = hru_names_df[0]
maxSWE_residual2007_rsh = pd.DataFrame(np.reshape(np.array(maxSWE_residual2007),(1,4720)).T, columns = out_names)
maxSWE_residual2007_rsh.index = hruidxID


for models in range(len(out_names)):
    x = list(hruidxID)
    fig = plt.figure(figsize=(20,15))
    plt.bar(x,maxSWE_residual2007_rsh[out_names[models]])
    plt.title(out_names[models], fontsize=42)
    plt.xlabel('hrus',fontsize=40)
    plt.ylabel('residual maxSWE (day)', fontsize=30)
    plt.yticks(fontsize=30)
    plt.xticks(fontsize=30)
    plt.savefig('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/figs/maxSWE_'+out_names[models])



maxSWE2008 = readSpecificDatafromAllHRUs(av_swe_df,hru_names_df[0],13670)
maxSWE_obs2008 = [977]
#
#
#av_swe_df2 = av_swe_df.copy(); av_swe_df2.set_index(av_swe_df['counter'],inplace=True)
#realMaxSWE = av_swe_df2.max()
#realMaxSWE_date = av_swe_df2.idxmax() 
#%%**************************************************************************************************
## ********************** calculating snowmelt rate based on SWE *************************************
#sweM1,SWE1date = SWEandSWEDateforSpecificDate(hru_names_df[0],5289,av_swe_df,dosd_df)
#sweM2,SWE2date = SWEandSWEDateforSpecificDate(hru_names_df[0],5457,av_swe_df,dosd_df)
#sweM3,SWE3date = SWEandSWEDateforSpecificDate(hru_names_df[0],5793,av_swe_df,dosd_df)
#sweM4,SWE4date = SWEandSWEDateforSpecificDate(hru_names_df[0],5960,av_swe_df,dosd_df)
##%%
#meltingrate1 = meltingRateBetween2days(sweM1,sweM2,SWE1date,SWE2date)
#meltingrate2 = meltingRateBetween2days(sweM2,sweM3,SWE2date,SWE3date)
#meltingrate3 = meltingRateBetween2days(sweM3,sweM4,SWE3date,SWE4date)
#
#meltingrateAvg_mod = []
#for countermr in range (np.size(meltingrate1)):
#    meltingrateAvg_mod.append((meltingrate1[countermr]+meltingrate2[countermr]+meltingrate3[countermr])/3)
##%%
#sweMR = [711, 550, 309, 84]
#mrDate = ['2007-05-09 08:50 5289','2007-05-16 09:00 5457','2007-05-30 09:00 5793','2007-06-06 08:15 5960']  
#meltingrate1_obs = np.array([0.1*24*(711-550.)/(5457.-5289)])
#meltingrate2_obs = np.array([0.1*24*(550.-309)/(5793.-5457)])
#meltingrate3_obs = np.array([0.1*24*(309-84.)/(5960-5793.)])
#meltingrateAvg_obs = (meltingrate1_obs+meltingrate2_obs+meltingrate3_obs)/3.
##'2007-05-09 08:50':5289, to '2007-06-06 08:15': 5960, 
##swe_mm = [711, 84]
#meltingRate_obs = [0.1*24*(711-84.)/(5960-5289.)] #cm/day
##%% new criteria-swe before max 
#swe2bfrmax = readSpecificDatafromAllHRUs(av_swe_df,hru_names_df[0],4906)
#swe3bfrmax = readSpecificDatafromAllHRUs(av_swe_df,hru_names_df[0],4785)
#
#swe2bmax_obs = [654]
#swe3bmax_obs = [678]
##%% defining criteria
##coldcontentcrit = [abs(values) for values in mySubtract(coldcontent0305,cc0305)]
#meltingRateCrit = [abs(values) for values in mySubtract(meltingrateAvg_mod,meltingrateAvg_obs)]
#maxSWEcrit = [abs(values) for values in mySubtract(maxSWE,maxSWE_obs)]
#swe2bmaxCrit = [abs(values) for values in mySubtract(swe2bfrmax,swe2bmax_obs)]
#swe3bmaxCrit = [abs(values) for values in mySubtract(swe3bfrmax,swe3bmax_obs)]
##fig = plt.figure(figsize=(20,15))
##xs = meltingRateCrit
##ys = maxSWEcrit
##plt.scatter(xs, ys)
##plt.title('criteria for best combos')
##plt.xlabel('delta_maxSWE (mm)',fontsize=40)
##plt.ylabel('delta_meltingRate (cm/day)',fontsize=40)
##plt.savefig('SA2/'+'maxswe_meltinRateAvg')
##%%
##coldcontentcrit_df = pd.DataFrame(coldcontentcrit, columns=['coldContent'])
#meltingRateCrit_df = pd.DataFrame(meltingRateCrit, columns=['meltingRate'])
#maxSWECrit_df = pd.DataFrame(maxSWEcrit, columns=['maxSWE'])
#swe2bmaxCrit_df = pd.DataFrame(swe2bmaxCrit, columns=['swe2bmaxCrit'])
#swe3bmaxCrit_df = pd.DataFrame(swe3bmaxCrit, columns=['swe3bmaxCrit'])
#
#criteria_df = pd.concat([meltingRateCrit_df, maxSWECrit_df, swe2bmaxCrit_df, swe3bmaxCrit_df], axis=1) #coldcontentcrit_df, 
#criteria_df.set_index(hru_names_df[0],inplace=True)
#Apareto_model_param0 = pd.DataFrame(criteria_df.index[((criteria_df['maxSWE']) <= 5) & ((criteria_df['meltingRate'])<=0.03)].tolist()) # & ((criteria_df['coldContent'])<=7)
#
#Apareto_model_param1 = pd.DataFrame(criteria_df.index[((criteria_df['maxSWE']) <= 38.99) & ((criteria_df['swe2bmaxCrit']) <= 72.1) & ((criteria_df['swe3bmaxCrit']) <= 72.1) & ((criteria_df['meltingRate'])<=0.1)].tolist()) # & ((criteria_df['coldContent'])<=7)
#Area = 291 * 10000 #m2
#residualMax = 0.45
#sweVol = Area * residualMax ; print sweVol
##%%
#DateSa2 = date(av_all,"%Y-%m-%d")
#sax = np.arange(0,np.size(DateSa2))
#sa_xticks = DateSa2
#safig, saax = plt.subplots(1,1, figsize=(20,15))
#plt.xticks(sax, sa_xticks[::1000], rotation=25, fontsize=20)
#saax.xaxis.set_major_locator(ticker.AutoLocator())
#plt.yticks(fontsize=20)
#for hru in Apareto_model_param1[0]:
#    plt.plot(av_swe_df[hru])
#plt.plot(swe_obs2006, 'ok', markersize=15)
##plt.legend()
#plt.savefig('ccs/'+'best_combo1')
##%% **************************************************************************************************
### ************************** calculating cold content ************************************************
###observed cold content in each day
#
#
#ax2 = saax.twinx()  # instantiate a second axes that shares the same x-axis

ax2.plot(av_swe_df[hru_names_df[0][0]].index,av_sd_df['10000stp'], linewidth=4, color = 'red')
ax2.plot(av_swe_df[hru_names_df[0][0]].index,av_sd_df['10000stp102'], linewidth=4, color = 'orange')

ax2.plot(snowdepth_obs_df2.index, snowdepth_obs_df2['observed_snowdepth'], linewidth=3, color = 'black')

ax2.tick_params(axis='y')
ax2.legend('snow depth', fontsize = 30)#, loc = 'upper left'
ax2.set_ylabel('Snow depth', fontsize=40)
ax2.set_yticklabels([0,0.5,1,1.5,2,2.5,3], fontsize=40)

#ax2.xaxis.set_major_locator(ticker.AutoLocator())

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#    
#    
#    
#    
#  
  