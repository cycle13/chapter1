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
        if swe_df[names][hour]>0.:
            SWE.append(swe_df[names][hour])
            SWEdate.append(hour)
        else: 
            SWE.append(0.0)
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

def readData4multibleLayers4specificDateofAvailableSWE(av_mlt_df,obs_swe_ind,hru_names_df,sum0flayers0,sum0flayers1):
    layerTemp4dates = []
    for dates in range (len(obs_swe_ind)):
        hruLayerTemp = []
        for hrus in range (len(hru_names_df[0])):
            strt = int(sum0flayers0[dates][hrus])
            end = int(sum0flayers1[dates][hrus])
            date1LayerTemp = av_mlt_df[hru_names_df[0][hrus]][strt:end]
            hruLayerTemp.append(date1LayerTemp)
        layerTemp4dates.append(hruLayerTemp)
    return layerTemp4dates

def readData4multibleSnowLayers4specificDateofAvailableSWE(av_mlt_df,obs_swe_ind,hru_names_df,sum0flayers0,sum0flayers1):
    layerTemp4dates = []
    for dates in range (len(obs_swe_ind)):
        hruLayerTemp = []
        for hrus in range (len(hru_names_df[0])):
            strt = int(sum0flayers0[dates][hrus])
            end = int(sum0flayers1[dates][hrus])
            date1LayerTemp = av_mlt_df[hru_names_df[0][hrus]][strt:end]
            hruLayerTemp.append(date1LayerTemp)
        layerTemp4dates.append(hruLayerTemp)
    
    snowLayerTemp4dates = []
    for dates2 in range (len(obs_swe_ind)):
        hrusnowLayerTemp = []
        for hrus2 in range (len(hru_names_df[0])):
            date1snowLayerTemp = layerTemp4dates[dates2][hrus2][:-8]
            hrusnowLayerTemp.append(date1snowLayerTemp)
        snowLayerTemp4dates.append(hrusnowLayerTemp)
        
    return snowLayerTemp4dates
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
obs_swe_ind = np.array([760,1455,2220,2920,3735,3900,4070,4240,4405,4766,4886,5125,5289,5460,
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
#%% Snow surface temperature observation data
with open("C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa2_bestSweSD/sst_1hr_2006-2008.csv") as sast:
    reader2 = csv.reader(sast)
    raw_sst = [st for st in reader2]
sa_sst_column1 = []
for csv_counter3 in range (len (raw_sst)):
    for csv_counter4 in range (6):
        sa_sst_column1.append(raw_sst[csv_counter3][csv_counter4])
sa_sst=np.reshape(sa_sst_column1,(len (raw_sst),6))
sa_sst_obs=[float(val2) for val2 in sa_sst[1:len(raw_sst),5]]
sa_sst_obs[:][0:4567:12]
#sa_sd_obs = [float(value) for value in sa_snowdepth1]
sa_sst_obs_date = pd.DatetimeIndex(sa_sst[1:len(raw_sst),0])

sst_obs_df = pd.DataFrame(sa_sst_obs, columns = ['observed_snowSurfaceTemp']) 
sst_obs_df.set_index(sa_sst_obs_date,inplace=True)
sst_obs_df.set_index(sa_sst_obs_date,inplace=True)
sst_obs_df2= pd.DataFrame(sst_obs_df.values, columns = ['observed_snowSurfaceTemp']) 

sst_obs_morn_df = pd.DataFrame((sst_obs_df['observed_snowSurfaceTemp']['2006-10-01 03:00':'2008-06-13 03:00':24].values+
                                sst_obs_df['observed_snowSurfaceTemp']['2006-10-01 04:00':'2008-06-13 04:00':24].values+
                                sst_obs_df['observed_snowSurfaceTemp']['2006-10-01 05:00':'2008-06-13 05:00':24].values+
                                sst_obs_df['observed_snowSurfaceTemp']['2006-10-01 06:00':'2008-06-13 06:00':24].values)/4.)
sst_obs_morn_df.index = snowdepth_obs_df.index[0:len(sst_obs_morn_df)]

sst_obs_morn_df2 = pd.DataFrame(sst_obs_morn_df.values, columns = ['observed_snowSurfaceTemp_morning']) 
sst_obs_morn_df2.set_index(indx_sd[0:622],inplace=True)
#%% defining hrus_name
hruidxID = [10033,10089,10209,10312,10327,10328,10330,10336,10338,10339,10349,10369,10371,
            10375,10389,10410,10521,10523,10524,10526,10534,10544,10545,10590,10726]
hru_num = np.size(hruidxID)
years = ['2007','2008']#
out_names = ["ljh","ljp","ljc","ltc"]#,"sth","stc","sjc","sjp","lth","ltp"
                       
paramModel = (np.size(out_names))*(hru_num)
hru_names =[]
for i in out_names:
    hru_names.append(['{}{}'.format(j, i) for j in hruidxID])
hru_names1 = np.reshape(hru_names,(paramModel,1))
hru_names_df = pd.DataFrame (hru_names1)

#%%  reading output files
from allNcFiles_sa2_stcc import av_ncfiles

av_all = readAllNcfilesAsDataset(av_ncfiles)
#DateSa2007 = date(av_all[0],"%Y-%m-%d %H:%M")
#DateSa2008 = date(av_all[1],"%Y-%m-%d %H:%M")

#av_swe_df = readVariablefromMultipleNcfilesDatasetasDF40neYear(av_all,'scalarSWE',hru_names_df,hruidxID,out_names)

av_sd_df = readVariablefromMultipleNcfilesDatasetasDF(av_all,'scalarSnowDepth',hru_names_df,hruidxID,out_names)
av_swe_df = readVariablefromMultipleNcfilesDatasetasDF(av_all,'scalarSWE',hru_names_df,hruidxID,out_names)

av_st_df = readVariablefromMultipleNcfilesDatasetasDF(av_all,'scalarSurfaceTemp',hru_names_df,hruidxID,out_names)
av_sst_date = np.append(date(av_all[0],"%Y-%m-%d %H:%M"),date(av_all[1],"%Y-%m-%d %H:%M"))
av_st_df.index = av_sst_date
av_st_df2 =pd.DataFrame(av_st_df.values, columns=av_st_df.columns)
av_st_morn_df = pd.DataFrame((av_st_df[:]['2006-10-01 03:00':'2008-06-13 04:00':24].values+
                              av_st_df[:]['2006-10-01 04:00':'2008-06-13 05:00':24].values+
                              av_st_df[:]['2006-10-01 05:00':'2008-06-13 06:00':24].values+
                              av_st_df[:]['2006-10-01 06:00':'2008-06-13 07:00':24].values)/4.)
av_st_morn_df.index = snowdepth_obs_df.index[0:len(sst_obs_morn_df)]
av_st_morn_df2 = pd.DataFrame(av_st_morn_df.values, columns=av_st_df2.columns)
av_st_morn_df2.set_index(sst_obs_morn_df2.index,inplace=True)
#modeled cold content
#number of snow layer
av_nsl_df = readVariablefromMultipleNcfilesDatasetasDF(av_all,'nSnow',hru_names_df,hruidxID,out_names)
nsnow = [av_nsl_df.values[ns,1:] for ns in obs_swe_ind]
# snow layers temperature
av_mlt_df = readVariablefromMultipleNcfilesDatasetasDF(av_all,'mLayerTemp',hru_names_df,hruidxID,out_names)
sum0flayers0 = []
for ns in obs_swe_ind:
    if ns<8760:
        sum0flayer = np.sum(av_nsl_df.values[0:ns,1:]+8., axis=0)
        sum0flayers0.append(sum0flayer)
    else: 
        sum0flayer = 235214 + np.sum(av_nsl_df.values[8760:ns,1:]+8., axis=0)
        sum0flayers0.append(sum0flayer)
#sum0flayers0 = [np.sum(av_nsl_df.values[0:ns,1:]+8., axis=0) for ns in obs_swe_ind[0:17]]
sum0flayers1 = []
for ns in obs_swe_ind:
    if ns<8760:
        sum0flayers = np.sum(av_nsl_df.values[0:ns+1,1:]+8., axis=0)
        sum0flayers1.append(sum0flayers)
    else: 
        sum0flayers = 235214 + np.sum(av_nsl_df.values[8760:ns+1,1:]+8., axis=0)
        sum0flayers1.append(sum0flayers)
#sum0flayers1 = [np.sum(av_nsl_df.values[0:ns+1,1:]+8., axis=0) for ns in obs_swe_ind[0:17]]
layerTemp4dates = readData4multibleLayers4specificDateofAvailableSWE(av_mlt_df,obs_swe_ind,hru_names_df,sum0flayers0,sum0flayers1)
snowLayerTemp4dates = readData4multibleSnowLayers4specificDateofAvailableSWE(av_mlt_df,obs_swe_ind,hru_names_df,sum0flayers0,sum0flayers1)

#height of each layer
av_mld_df = readVariablefromMultipleNcfilesDatasetasDF(av_all,'mLayerDepth',hru_names_df,hruidxID,out_names)
layerDepth4dates = readData4multibleLayers4specificDateofAvailableSWE(av_mld_df,obs_swe_ind,hru_names_df,sum0flayers0,sum0flayers1)

#mLayerVolFracIce          | 1    
#mLayerVolFracLiq          | 1    
#mLayerVolFracWat          | 1    
#scalarVolHtCap_ice        | 1
#mLayerVolHtCapBulk        | 1
#scalarGroundSnowFraction  | 1
#mLayerFracLiqSnow         | 1
#mLayerThetaResid          | 1
#mLayerPoreSpace           | 1

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
plt.savefig('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa2_bestSweSD/sa_sa2_swe_4c_lastP_stcc1.png')

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
plt.savefig('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa2_bestSweSD/sa_sa2_sd_4c_lastP_stcc1.png')
#%% snow surface temperature 
safig3, saax3 = plt.subplots(1,1, figsize=(30,20))#
for hru3 in range (len(hru_names_df[0])):#
    plt.plot(av_st_df2[hru_names_df[0][hru3]]-273.15, linewidth=4)

plt.plot(sst_obs_df2['observed_snowSurfaceTemp'], 'k')#, linewidth=6[0:16], markersize=15
plt.yticks(fontsize=40)
plt.xlabel('Time 2006-2008', fontsize=40)
saax3.set_ylabel('snow surface temperature (C)', fontsize=40)
plt.xticks(sax[::3000], sa_xticks[::3000], rotation=25, fontsize=40)# 
safig3.tight_layout()  
plt.savefig('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa2_bestSweSD/sa_sa2_sst_4c_lastP_stcc1.png')


safig4, saax4 = plt.subplots(1,1, figsize=(30,20))#
for hru4 in range (len(hru_names_df[0])):#
    plt.plot(av_st_morn_df2[hru_names_df[0][hru4]]-273.15, linewidth=4)

plt.plot(sst_obs_morn_df2.index, sst_obs_morn_df2['observed_snowSurfaceTemp_morning'], 'k', linewidth=4)#[0:16], markersize=15
plt.yticks(fontsize=40)
plt.xlabel('Time 2006-2008', fontsize=40)
saax4.set_ylabel('snow surface temperature at before sunrise (C)', fontsize=40)
plt.xticks(sax[::2000], sa_xticks[::2000], rotation=25, fontsize=40)# 
safig4.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa2_bestSweSD/sa_sa2_sst5am_4c_lastP_stcc1.png')

#%% day of snow disappearance (based on snowdepth)-final output

dosd_df2007, dosd_residual_df2007 = calculatingSDD(av_sd_df[:][5000:8737],hru_names_df,5976,'2007',3737,5000)
dosd_df2008, dosd_residual_df2008 = calculatingSDD(av_sd_df[:][14000:],hru_names_df,15000,'2008',3521,14000)

dosd_residual_df_sum = abs(dosd_residual_df2007)+abs(dosd_residual_df2008)
dosd_residual_df_sum_r = pd.concat([dosd_residual_df_sum[out_names[0]],
                                    dosd_residual_df_sum[out_names[1]],
                                    dosd_residual_df_sum[out_names[2]],
                                    dosd_residual_df_sum[out_names[3]],],axis=0)#
#                                    dosd_residual_df_sum[out_names[4]],
#                                    dosd_residual_df_sum[out_names[5]], columns = ['sum2years'])
dosd_residual_df_sum_r.index = hru_names_df[0]
dosd_residual_df2007min = np.min(abs(dosd_residual_df2007))

dosd_obs2007 = 5976
dosd_obs2008 = 14976-8760.
dosd_normal = (dosd_obs2008 + dosd_obs2007)/24.
#%%**************************************************************************************************
# *********************** finding max corespondance swe for '2007-05-09 08:50'***********************
#'2007-04-18' 4776: 4800, '2007-04-23' 4896:4920, '2007-05-02' 5112:5136
#Group1: '2007-03-12 14:00' (3902),'2007-03-19 12:30 (4068)','2007-03-26 12:30 (4236)','2007-04-02 12:30'(4404),
#Group2: '2007-04-18 08:35' (4784),'2007-04-23 10:30 (4907)','2007-05-02 08:40'(5121), 

max1SWE2007 = readSpecificDatafromAllHRUs(av_swe_df,hru_names_df[0],5289)
max1SWE_obs2007 = [711]  
max2SWE2007 = readSpecificDatafromAllHRUs(av_swe_df,hru_names_df[0],5125)
max2SWE_obs2007 = [660]  
max3SWE2007 = readSpecificDatafromAllHRUs(av_swe_df,hru_names_df[0],4886)
max3SWE_obs2007 = [654]  
max4SWE2007 = readSpecificDatafromAllHRUs(av_swe_df,hru_names_df[0],4766)
max4SWE_obs2007 = [678] 

maxSWE1_residual2007 = (pd.DataFrame(mySubtract(max1SWE2007,max1SWE_obs2007))).T
maxSWE2_residual2007 = (pd.DataFrame(mySubtract(max2SWE2007,max2SWE_obs2007))).T
maxSWE3_residual2007 = (pd.DataFrame(mySubtract(max3SWE2007,max3SWE_obs2007))).T
maxSWE4_residual2007 = (pd.DataFrame(mySubtract(max4SWE2007,max4SWE_obs2007))).T

maxSWE_residual2007_perc = (abs(maxSWE1_residual2007)/max1SWE_obs2007[0]+
                            abs(maxSWE2_residual2007)/max2SWE_obs2007[0]+
                            abs(maxSWE3_residual2007)/max3SWE_obs2007[0]+
                            abs(maxSWE4_residual2007)/max4SWE_obs2007[0])/4.#+

maxSWE_residual2007_perc.columns = hru_names_df[0]
maxSWE_residual2007_rsh = np.reshape(np.array(maxSWE_residual2007_perc),(1,len(hru_names_df))).T
maxSWE_residual2007_perc_df = pd.DataFrame(np.reshape(maxSWE_residual2007_rsh,(len(out_names),len(hruidxID))).T,columns = out_names)#
maxSWE_residual2007_perc_df.index = hruidxID

#obs_swe_ind = np.array([760,1455,2220,2920,3735,3900,4070,4240,4405,4766,4886,5125,5289,5460,
#                        5630,5799,5965,10290,10980,11700,12470,12980,13165,13488,13670,13815,
#                        14010,14150,14320,14485,14655,14800])            
#swe_mm = [58,  169, 267, 315, 499, 523, 503, 549, 611, 678, 654, 660, 711, 550, 443, 309, 84, 
#          141, 300, 501, 737, 781, 837, 977, 950, 873, 894, 872, 851, 739, 538, 381]  

max1SWE2008 = readSpecificDatafromAllHRUs(av_swe_df,hru_names_df[0],13488)
max1SWE_obs2008 = [977]
max2SWE2008 = readSpecificDatafromAllHRUs(av_swe_df,hru_names_df[0],13670)
max2SWE_obs2008 = [950]
max3SWE2008 = readSpecificDatafromAllHRUs(av_swe_df,hru_names_df[0],13815)
max3SWE_obs2008 = [873]
max4SWE2008 = readSpecificDatafromAllHRUs(av_swe_df,hru_names_df[0],14010)
max4SWE_obs2008 = [894]

max1SWE_residual2008 = (pd.DataFrame(mySubtract(max1SWE2008,max1SWE_obs2008))).T
max2SWE_residual2008 = (pd.DataFrame(mySubtract(max2SWE2008,max2SWE_obs2008))).T
max3SWE_residual2008 = (pd.DataFrame(mySubtract(max3SWE2008,max3SWE_obs2008))).T
max4SWE_residual2008 = (pd.DataFrame(mySubtract(max4SWE2008,max4SWE_obs2008))).T

maxSWE_residual2008_perc = (abs(max1SWE_residual2008)/max1SWE_obs2008[0]+
                            abs(max2SWE_residual2008)/max2SWE_obs2008[0]+
                            abs(max3SWE_residual2008)/max3SWE_obs2008[0]+
                            abs(max4SWE_residual2008)/max4SWE_obs2008[0])/4.

maxSWE_residual2008_perc.columns = hru_names_df[0]
maxSWE_residual2008_rsh = np.reshape(np.array(maxSWE_residual2008_perc),(1,len(hru_names_df))).T
maxSWE_residual2008_perc_df = pd.DataFrame(np.reshape(maxSWE_residual2008_rsh,(len(out_names),len(hruidxID))).T,columns = out_names)#
maxSWE_residual2008_perc_df.index = hruidxID

maxSWE_residual_df_mean = abs(maxSWE_residual2008_perc_df)/2+abs(maxSWE_residual2007_perc_df)/2
maxSWE_residual_df_mean_r_perc = pd.concat([maxSWE_residual_df_mean[out_names[0]],
                                            maxSWE_residual_df_mean[out_names[1]],
                                            maxSWE_residual_df_mean[out_names[2]],
                                            maxSWE_residual_df_mean[out_names[3]]],axis=0)#,
#                                            maxSWE_residual_df_mean[out_names[4]],
#                                            maxSWE_residual_df_mean[out_names[5]]], columns = ['sum2years'])
maxSWE_residual_df_mean_r_perc.index = hru_names_df[0]

maxSWE_residual_df_perc_thresh = maxSWE_residual_df_mean_r_perc[maxSWE_residual_df_mean_r_perc<=0.10]
maxSWE_residual_df_perc_thresh_inx = maxSWE_residual_df_perc_thresh.index

#maxSWE_residual_dfmin = abs(maxSWE_residual_df_sum_r_perc).nsmallest(50,'first')
    
#%%**************************************************************************************************
## *********** calculating snowmelt rate based on SWE #cm/day *************************************
#maxSWE2008 = readSpecificDatafromAllHRUs(av_swe_df,hru_names_df[0],13670)
#2007
sweM1,SWE1date = SWEandSWEDateforSpecificDate(hru_names_df[0],5289,av_swe_df,dosd_df2007)
sweM2,SWE2date = SWEandSWEDateforSpecificDate(hru_names_df[0],5457,av_swe_df,dosd_df2007)
sweM3,SWE3date = SWEandSWEDateforSpecificDate(hru_names_df[0],5799,av_swe_df,dosd_df2007)
sweM4,SWE4date = SWEandSWEDateforSpecificDate(hru_names_df[0],5965,av_swe_df,dosd_df2007)

#2008
sweM5,SWE5date = SWEandSWEDateforSpecificDate(hru_names_df[0],13488,av_swe_df,dosd_df2008)
sweM6,SWE6date = SWEandSWEDateforSpecificDate(hru_names_df[0],14320,av_swe_df,dosd_df2008)
sweM7,SWE7date = SWEandSWEDateforSpecificDate(hru_names_df[0],14485,av_swe_df,dosd_df2008)
sweM8,SWE8date = SWEandSWEDateforSpecificDate(hru_names_df[0],14800,av_swe_df,dosd_df2008)

#%%  #cm/day
meltingrate20071 = meltingRateBetween2days(sweM1,sweM2,SWE1date,SWE2date)
meltingrate20072 = meltingRateBetween2days(sweM2,sweM3,SWE2date,SWE3date)
meltingrate20073 = meltingRateBetween2days(sweM3,sweM4,SWE3date,SWE4date)
meltingrate20074 = np.array(0.1*24)*sweM4 / abs(SWE4date-dosd_df2007.values)

meltingrateAvg_2007 = []
for countermr in range (np.size(meltingrate20071)):
    meltingrateAvg_2007.append((meltingrate20071[countermr]+meltingrate20072[countermr]+
                                meltingrate20073[countermr]+meltingrate20072[countermr])/4.)

meltingrate20081 = meltingRateBetween2days(sweM5,sweM6,SWE5date,SWE6date)
meltingrate20082 = meltingRateBetween2days(sweM6,sweM7,SWE6date,SWE7date)
meltingrate20083 = meltingRateBetween2days(sweM7,sweM8,SWE7date,SWE8date)
meltingrate20084 = np.array(0.1*24)*sweM8 / abs(SWE8date-dosd_df2008.values)

meltingrateAvg_2008 = []
for countermr1 in range (np.size(meltingrate20081)):
    meltingrateAvg_2008.append((meltingrate20081[countermr1]+meltingrate20082[countermr1]+
                                meltingrate20083[countermr1]+meltingrate20072[countermr])/4.)

#%%  #cm/day
sweMR = [711, 550, 309, 84]
mrDate = ['2007-05-09 08:50 5289','2007-05-16 09:00 5457','2007-05-30 09:00 5793','2007-06-06 08:15 5960']  
meltingrate1_obs2007 = np.array([0.1*24*(711-550.)/(5457.-5289)])
meltingrate2_obs2007 = np.array([0.1*24*(550.-309)/(5799.-5457)])
meltingrate3_obs2007 = np.array([0.1*24*(309-84.)/(5965-5799.)])
meltingrate4_obs2007 = np.array([0.1*24*(84.)/(5976-5965)])
#meltingrateAvg_obs2007 = (meltingrate1_obs2007+meltingrate2_obs2007+
#                          meltingrate3_obs2007+meltingrate4_obs2007)/4.
#'2007-05-09 08:50':5289, to '2007-06-06 08:15': 5960, 
#swe_mm = [711, 84]
#meltingRate_obs2007 = [0.1*24*(711-84.)/(5965-5289.)] 
dosd_obs2007 = 5976
dosd_obs2008 = 14976-8760.

meltingrate1_obs2008 = np.array([0.1*24*(977-851.)/(14320.-13488)])
meltingrate2_obs2008 = np.array([0.1*24*(851.-739)/(14485.-14320)])
meltingrate3_obs2008 = np.array([0.1*24*(739-381.)/(14800-14485.)])
meltingrate4_obs2008 = np.array([0.1*24*(381.)/(14976-14800.)])
#meltingrateAvg_obs2008 = (meltingrate1_obs2008+meltingrate2_obs2008+
#                          meltingrate3_obs2008+meltingrate4_obs2008)/4.

#%%
meltRate_residual_mean_perc = (abs(meltingrate20071-meltingrate1_obs2007)/meltingrate1_obs2007 +
                               abs(meltingrate20072-meltingrate2_obs2007)/meltingrate2_obs2007 +
                               abs(meltingrate20073-meltingrate3_obs2007)/meltingrate3_obs2007 +
#                               abs(meltingrate20074-meltingrate4_obs2007)/meltingrate4_obs2007 +
                               abs(meltingrate20081-meltingrate1_obs2008)/meltingrate1_obs2008 +
                               abs(meltingrate20083-meltingrate3_obs2008)/meltingrate3_obs2008 +
#                               abs(meltingrate20084-meltingrate4_obs2008)/meltingrate4_obs2008 +
                               abs(meltingrate20082-meltingrate2_obs2008)/meltingrate2_obs2008)/6.
meltRate_residual_df_mean_perc = pd.DataFrame(meltRate_residual_mean_perc)

meltRate_residual_df_mean_perc.index = hru_names_df[0]
meltRate_residual_df_mean_perc_bestSWE = meltRate_residual_df_mean_perc[0][maxSWE_residual_df_perc_thresh_inx[:]]
ind_df_mean_perc_bestSWE_bestMR = (meltRate_residual_df_mean_perc_bestSWE[meltRate_residual_df_mean_perc_bestSWE<=0.20]).index

#%%
dosd_residual_df_sum_r_bestSWE_bestMR = dosd_residual_df_sum_r[ind_df_mean_perc_bestSWE_bestMR]
ind_df_mean_perc_bestSWE_bestMR_bestSDD = (dosd_residual_df_sum_r_bestSWE_bestMR[dosd_residual_df_sum_r_bestSWE_bestMR<=11]).index

#%% snow surface temp obj func.
sst_obs_morn_2007 = sst_obs_morn_df['2007-03-20':'2007-05-15']
sst_obs_morn_2008 = sst_obs_morn_df['2008-04-01':'2008-05-15']

av_st_morn_2007 = av_st_morn_df['2007-03-20':'2007-05-15']-273.15
av_st_morn_2008 = av_st_morn_df['2008-04-01':'2008-05-15']-273.15

delta_sst_morn2007 = abs(av_st_morn_2007.values[:,1:] - sst_obs_morn_2007.values)/abs(sst_obs_morn_2007.values)
delta_sst_morn2008 = abs(av_st_morn_2008.values[:,1:] - sst_obs_morn_2008.values)/abs(sst_obs_morn_2008.values)

delta_sst_morn2007_perc = np.sum(delta_sst_morn2007,axis =0)/len(delta_sst_morn2007)
delta_sst_morn2008_perc = np.sum(delta_sst_morn2008,axis =0)/len(delta_sst_morn2008)

delta_sst_morn_perc = (delta_sst_morn2007_perc + delta_sst_morn2007_perc)/2. 
delta_sst_morn_perc_df = pd.DataFrame(delta_sst_morn_perc)
delta_sst_morn_perc_df.index = hru_names_df[0]

delta_sst_df_sum_r_bestSWE_bestMR_bestSDD = delta_sst_morn_perc_df[0][ind_df_mean_perc_bestSWE_bestMR_bestSDD]
ind_df_mean_perc_bestSWE_bestMR_bestSDD_bestSST = (delta_sst_df_sum_r_bestSWE_bestMR_bestSDD[delta_sst_df_sum_r_bestSWE_bestMR_bestSDD<=0.25]).index

#%%
safig, saax = plt.subplots(1,1, figsize=(30,20))#
color = []
#for hru1 in range (len(dosd_residual_dfmin_indx)):
#    plt.plot(av_swe_df[dosd_residual_dfmin_indx[hru1]],  'green', linewidth=4)#
#for hru3 in range (len(hru_meltRater)):
#    plt.plot(av_swe_df[hru_meltRater[hru3]], 'pink', linewidth=4)
#for hru2 in range (len(hru_maxSWEr_r)):
#    plt.plot(av_swe_df[hru_maxSWEr_r[hru2]], 'blue', linewidth=4)#'green', 


for hru in ind_df_mean_perc_bestSWE_bestMR_bestSDD_bestSST:
    #print hru
    plt.plot(av_swe_df[hru], 'purple', linewidth=4)#'green', 
    
saax.plot(obs_swe.index, obs_swe , 'ok', markersize=15)#[0:16]

plt.yticks(fontsize=40)
plt.xlabel('Time 2006-2008', fontsize=40)
saax.set_ylabel('SWE(mm)', fontsize=40)

sax = np.arange(0,np.size(date_sa))
sa_xticks = date_sa
plt.xticks(sax[::3000], sa_xticks[::3000], rotation=25, fontsize=40)# 

safig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa2_bestSweSD/sa_sa2_swe_taSweMrSDDSST.png')
#%%
safig4, saax4 = plt.subplots(1,1, figsize=(30,20))#
for hru4 in ind_df_mean_perc_bestSWE_bestMR_bestSDD_bestSST:#
    plt.plot(av_st_morn_df2[hru4]-273.15, linewidth=4)

plt.plot(sst_obs_morn_df2.index, sst_obs_morn_df2['observed_snowSurfaceTemp_morning'], 'k', linewidth=4)#[0:16], markersize=15
plt.yticks(fontsize=40)
plt.xlabel('Time 2006-2008', fontsize=40)
saax4.set_ylabel('snow surface temperature at before sunrise (C)', fontsize=40)
plt.xticks(sax[::2000], sa_xticks[::2000], rotation=25, fontsize=40)# 
safig4.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa2_bestSweSD/sa_sa2_sst5am_4c_lastP_bestSWEMrSST.png')


#%%
safig2, saax2 = plt.subplots(1,1, figsize=(30,20))#

for hru4 in ind_df_mean_perc_bestSWE_bestMR_bestSDD:#len(hru_names_df[0]),
    plt.plot(av_sd_df[hru4], 'violet',linewidth=4)# 
#for hru5 in range (len(hru_meltRater)):#len(hru_names_df[0]),
#    plt.plot(av_sd_df[hru_meltRater[hru5]], 'pink', linewidth=4)#'green', 
#for hru6 in range (len(hru_maxSWEr_r)):#len(hru_names_df[0]),
#    plt.plot(av_sd_df[hru_maxSWEr_r[hru6]], 'blue', linewidth=4)#'green', 

plt.plot(snowdepth_obs_df2.index, snowdepth_obs_df2['observed_snowdepth'], 'k', linewidth=6)#[0:16], markersize=15

plt.yticks(fontsize=40)
plt.xlabel('Time 2006-2008', fontsize=40)
saax2.set_ylabel('snow depth (mm)', fontsize=40)
plt.xticks(sax[::3000], sa_xticks[::3000], rotation=25, fontsize=40)# 

safig2.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa_sa2_VARs_p12FPM_fTCs/sa_sa2_sd_taSweMrSDD6.png')

#hru_maxSWEr2007_r = np.reshape(hru_maxSWEr2007,(20,1))
#%% new criteria-swe before max 

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
  