###       /bin/bash runTestCases_docker.sh  scalarSnowfall scalarRainfall
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

def readVariablefromMultipleNcfilesDatasetasDF2(av_all,variable,hruname_df,hruidxID,out_names,ds):
    
    variableList = []
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
    
    #dosd_residual_df = pd.DataFrame(np.reshape(np.array(dosd_residual),(np.size(out_names),hru_num)).T, columns=out_names)
    dosd_residual_df = pd.DataFrame(np.array(dosd_residual))#,columns=hru_names_df[0]
    dosd_residual_df.index = hru_names_df[0]
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

def averageMaxSWE4dryAndWetYears (av_swe_df_h,hru_names_df_swe):
    # 2220000777
    max1SWE_h_dry = readSpecificDatafromAllHRUs(av_swe_df_h,hru_names_df_swe[0],5289)
    max2SWE_h_dry = readSpecificDatafromAllHRUs(av_swe_df_h,hru_names_df_swe[0],5125)
    max3SWE_h_dry = readSpecificDatafromAllHRUs(av_swe_df_h,hru_names_df_swe[0],4886)
    max4SWE_h_dry = readSpecificDatafromAllHRUs(av_swe_df_h,hru_names_df_swe[0],4766)
    
    maxSWE_h_dry = (np.array(max1SWE_h_dry)+np.array(max2SWE_h_dry)+np.array(max3SWE_h_dry)+np.array(max4SWE_h_dry))/4.
    maxSWE_dry_pd = pd.DataFrame(maxSWE_h_dry)
    maxSWE_dry_pd.index = hru_names_df_swe[0]
    
    #2220000888
    max1SWE_h_wet = readSpecificDatafromAllHRUs(av_swe_df_h,hru_names_df_swe[0],13488)
    max2SWE_h_wet = readSpecificDatafromAllHRUs(av_swe_df_h,hru_names_df_swe[0],13670)
    max3SWE_h_wet = readSpecificDatafromAllHRUs(av_swe_df_h,hru_names_df_swe[0],13815)
    max4SWE_h_wet = readSpecificDatafromAllHRUs(av_swe_df_h,hru_names_df_swe[0],14010)
    
    maxSWE_h_wet = (np.array(max1SWE_h_wet)+np.array(max2SWE_h_wet)+np.array(max3SWE_h_wet)+np.array(max4SWE_h_wet))/4.
    maxSWE_wet_pd = pd.DataFrame(maxSWE_h_wet)
    maxSWE_wet_pd.index = hru_names_df_swe[0]
    
    return maxSWE_dry_pd, maxSWE_wet_pd


def meltRateBased0nSWE (hru_names_df_swe,av_swe_df_h,dosd_df_h_dry,dosd_df_h_wet):
    #2007
    sweM1h,SWE1dateh = SWEandSWEDateforSpecificDate(hru_names_df_swe[0],5289,av_swe_df_h,dosd_df_h_dry)
    sweM2h,SWE2dateh = SWEandSWEDateforSpecificDate(hru_names_df_swe[0],5457,av_swe_df_h,dosd_df_h_dry)
    sweM3h,SWE3dateh = SWEandSWEDateforSpecificDate(hru_names_df_swe[0],5799,av_swe_df_h,dosd_df_h_dry)
    sweM4h,SWE4dateh = SWEandSWEDateforSpecificDate(hru_names_df_swe[0],5965,av_swe_df_h,dosd_df_h_dry)
    
    #2008
    sweM5h,SWE5dateh = SWEandSWEDateforSpecificDate(hru_names_df_swe[0],13488,av_swe_df_h,dosd_df_h_wet)
    sweM6h,SWE6dateh = SWEandSWEDateforSpecificDate(hru_names_df_swe[0],14320,av_swe_df_h,dosd_df_h_wet)
    sweM7h,SWE7dateh = SWEandSWEDateforSpecificDate(hru_names_df_swe[0],14485,av_swe_df_h,dosd_df_h_wet)
    sweM8h,SWE8dateh = SWEandSWEDateforSpecificDate(hru_names_df_swe[0],14800,av_swe_df_h,dosd_df_h_wet)
    
    #cm/day
    meltingrate_h_dry1 = meltingRateBetween2days(sweM1h,sweM2h,SWE1dateh,SWE2dateh)
    meltingrate_h_dry2 = meltingRateBetween2days(sweM2h,sweM3h,SWE2dateh,SWE3dateh)
    meltingrate_h_dry3 = meltingRateBetween2days(sweM3h,sweM4h,SWE3dateh,SWE4dateh)
    meltingrate_h_dry4 = np.array(0.1*24)*sweM4h / abs(SWE4dateh-dosd_df_h_dry.values)
    
    meltingrateAvg_h_dry = []
    for countermr in range (np.size(meltingrate_h_dry1)):
        meltingrateAvg_h_dry.append((meltingrate_h_dry1[countermr]+meltingrate_h_dry2[countermr]+
                                     meltingrate_h_dry3[countermr]+meltingrate_h_dry4[0][countermr])/4.)
    
    meltingrate_h_wet1 = meltingRateBetween2days(sweM5h,sweM6h,SWE5dateh,SWE6dateh)
    meltingrate_h_wet2 = meltingRateBetween2days(sweM6h,sweM7h,SWE6dateh,SWE7dateh)
    meltingrate_h_wet3 = meltingRateBetween2days(sweM7h,sweM8h,SWE7dateh,SWE8dateh)
    meltingrate_h_wet4 = np.array(0.1*24)*sweM8h / abs(SWE8dateh-dosd_df_h_wet.values)
    
    meltingrateAvg_h_wet = []
    for countermr1 in range (np.size(meltingrate_h_wet1)):
        meltingrateAvg_h_wet.append((meltingrate_h_wet1[countermr1]+meltingrate_h_wet2[countermr1]+
                                     meltingrate_h_wet3[countermr1]+meltingrate_h_wet4[0][countermr])/4.)

    meltingrateAvg_h_dry_df = pd.DataFrame(meltingrateAvg_h_dry)
    meltingrateAvg_h_dry_df.index = hru_names_df_swe[0]
    meltingrateAvg_h_wet_df = pd.DataFrame(meltingrateAvg_h_wet)
    meltingrateAvg_h_wet_df.index = hru_names_df_swe[0]

    return meltingrateAvg_h_dry_df, meltingrateAvg_h_wet_df

def calculatingTiming0f50Pecent0fAnnualInput (av_rPm_df_h,hru_names_df_swe):
    av_rPm_df_h_dry = av_rPm_df_h.iloc[0:8760]
    av_rPm_df_h_wet = av_rPm_df_h.iloc[8760:]
    av_rPm_df_h_dry_cum = av_rPm_df_h_dry.cumsum(axis=0)#; av_rPm_df_h_dry_cum.index = date_sa [0:8760]
    av_rPm_df_h_wet_cum = av_rPm_df_h_wet.cumsum(axis=0)#; av_rPm_df_h_wet_cum.index = date_sa [8760:]
    
    time0f50input_h_dry = []
    av_rPm_df_h_dry_cum50 = (av_rPm_df_h_dry_cum.iloc[-1])[1:]/2.
    for hrun in range(len(hru_names_df_swe)):
        inpt = av_rPm_df_h_dry_cum50[hrun]
        time0f50input = ((av_rPm_df_h_dry_cum[hru_names_df_swe[0][hrun]]-inpt).abs()).argmin()
        time0f50input_h_dry.append(time0f50input)
    time0f50input_h_dry_df = pd.DataFrame(time0f50input_h_dry)
    time0f50input_h_dry_df.index = hru_names_df_swe[0]
    
    time0f50input_h_wet = []
    av_rPm_df_h_wet_cum50 = (av_rPm_df_h_wet_cum.iloc[-1])[1:]/2.
    for hrun in range(len(hru_names_df_swe)):
        inpt = av_rPm_df_h_wet_cum50[hrun]
        time0f50input = ((av_rPm_df_h_wet_cum[hru_names_df_swe[0][hrun]]-inpt).abs()).argmin()
        time0f50input_h_wet.append(time0f50input-8760)
    time0f50input_h_wet_df = pd.DataFrame(time0f50input_h_wet)
    time0f50input_h_wet_df.index = hru_names_df_swe[0]

    return time0f50input_h_dry_df,time0f50input_h_wet_df

#%% SWE observation data 
date_swe = ['2006-11-01 11:10','2006-11-30 12:30','2007-01-01 11:10','2007-01-30 10:35','2007-03-05 14:30','2007-03-12 14:00', 
            '2007-03-19 12:30','2007-03-26 12:30','2007-04-02 12:30','2007-04-18 08:35','2007-04-23 10:30','2007-05-02 08:40', 
            '2007-05-09 08:50','2007-05-16 09:00','2007-05-23 08:30','2007-05-30 09:00','2007-06-06 08:15', 
            
            '2007-12-03 10:45','2008-01-01 11:30','2008-01-31 12:00','2008-03-03 14:30','2008-03-24 09:10','2008-04-01 09:55', 
            '2008-04-14 14:45','2008-04-22 12:30','2008-04-28 12:30','2008-05-06 09:15','2008-05-12 12:45','2008-05-19 10:40',
            '2008-05-26 08:45','2008-06-02 12:45','2008-06-08 08:45'] 
            
swe_mm = [58,  169, 267, 315, 499, 523, 503, 549, 611, 678, 654, 660, 711, 550, 443, 309, 84, 
          141, 300, 501, 737, 781, 837, 977, 950, 873, 894, 872, 851, 739, 538, 381]  

obs_swe = pd.DataFrame (swe_mm, columns=['swe_mm'])
obs_swe.set_index(pd.DatetimeIndex(date_swe),inplace=True)

max_swe_obs = max(obs_swe['swe_mm'])
max_swe_date_obs = obs_swe[obs_swe ['swe_mm']== max_swe_obs].index.tolist()  

swe_obs2007 = pd.DataFrame (obs_swe['swe_mm']['2006-11-01':'2007-06-06'], columns=['swe_mm'])
swe_obs2008 = pd.DataFrame (obs_swe['swe_mm']['2007-12-03':'2008-06-08'], columns=['swe_mm'])

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

sa_sd_obs_date = pd.DatetimeIndex(sa_snowdepth[1:len(raw_snowdepth1)-1,0])

snowdepth_obs_df = pd.DataFrame(sa_sd_obs, columns = ['observed_snowdepth']) 
snowdepth_obs_df.set_index(sa_sd_obs_date,inplace=True)

snowdepth_obs_df2 = pd.DataFrame(sa_sd_obs, columns = ['observed_snowdepth']) 
indx_sd = np.arange(0,16056,24)
snowdepth_obs_df2.set_index(indx_sd,inplace=True)

#%% defining hrus_name
years = ['2007','2008']#

# reading index of best parameters for each decision model combination for swe
with open("C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/cc_wrf_swe/p1213_sweBestParam_index.csv") as saprswe:
    reader1swe = csv.reader(saprswe)
    params2swe = [r1swe for r1swe in reader1swe]
params_index_swe = np.array(params2swe[1:])

out_names_swe = ['lsc','lsh','lsp','ssc','ssh','ssp','ljc','ljp','lth','sjh','sjp','stp']
params_index_swe_df = pd.DataFrame(params_index_swe, columns = out_names_swe)


hruid_swe_dic = {'lsc': [], 'lsh': [], 'lsp': [], 'ssc': [], 'ssh': [], 'ssp': [], 
                 'ljc': [], 'ljp': [], 'lth': [], 'sjh': [], 'sjp': [], 'stp': []}
for keys in out_names_swe:
    for counter in range (len(params_index_swe_df)):
        if int(params_index_swe_df[keys][counter])>0:
            hruid_swe_dic[keys].append(int(params_index_swe_df[keys][counter]))


hru_names_dic_swe = {'lsc': [], 'lsh': [], 'lsp': [], 'ssc': [], 'ssh': [], 'ssp': [],  
                     'ljc': [], 'ljp': [], 'lth': [], 'sjh': [], 'sjp': [], 'stp': []}
for i in out_names_swe:
    hru_names_dic_swe[i].append(['{}{}'.format(i, j) for j in hruid_swe_dic[i]])

hru_names_df_swe = pd.concat([pd.DataFrame(hru_names_dic_swe['lsc'][0]),
                              pd.DataFrame(hru_names_dic_swe['lsh'][0]),
                              pd.DataFrame(hru_names_dic_swe['lsp'][0]),
                              pd.DataFrame(hru_names_dic_swe['ssc'][0]),
                              pd.DataFrame(hru_names_dic_swe['ssh'][0]),
                              pd.DataFrame(hru_names_dic_swe['ssp'][0]),
                              pd.DataFrame(hru_names_dic_swe['ljc'][0]),
                              pd.DataFrame(hru_names_dic_swe['ljp'][0]),
                              pd.DataFrame(hru_names_dic_swe['lth'][0]),
                              pd.DataFrame(hru_names_dic_swe['sjh'][0]),
                              pd.DataFrame(hru_names_dic_swe['sjp'][0]),
                              pd.DataFrame(hru_names_dic_swe['stp'][0]),
                              ],ignore_index = True)


# reading index of best parameters for each decision model combination for swe and melt rate
with open("C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/cc_wrf_swe/p1213_sweMrBestParam_index.csv") as saprsweMr:
    reader1sweMr = csv.reader(saprsweMr)
    params2sweMr = [r1sweMr for r1sweMr in reader1sweMr]
params_index_swe_mr = np.array(params2sweMr[1:])

out_names_swe_mr = ['lsc','lsh','lsp','ssc','ssh','ssp','ljc','ljp','sjh','sjp']
params_index_swe_mr_df = pd.DataFrame(params_index_swe_mr, columns = out_names_swe_mr)

hruid_swe_mr_dic = {'lsc': [], 'lsh': [], 'lsp': [], 'ssc': [], 'ssh': [], 'ssp': [], 
                    'ljc': [], 'ljp': [], 'sjh': [], 'sjp': []}
for keysm in out_names_swe_mr:
    for counterm in range (len(params_index_swe_mr_df)):
        if int(params_index_swe_mr_df[keysm][counterm])>0:
            hruid_swe_mr_dic[keysm].append(int(params_index_swe_mr_df[keysm][counterm]))

hru_names_dic_swe_mr = {'lsc': [], 'lsh': [], 'lsp': [], 'ssc': [], 'ssh': [], 'ssp': [],  
                        'ljc': [], 'ljp': [], 'sjh': [], 'sjp': []}
for im in out_names_swe_mr:
    hru_names_dic_swe_mr[im].append(['{}{}'.format(im, jm) for jm in hruid_swe_mr_dic[im]])

hru_names_df_swe_mr = pd.concat([pd.DataFrame(hru_names_dic_swe_mr['lsc'][0]),
                                 pd.DataFrame(hru_names_dic_swe_mr['lsh'][0]),
                                 pd.DataFrame(hru_names_dic_swe_mr['lsp'][0]),
                                 pd.DataFrame(hru_names_dic_swe_mr['ssc'][0]),
                                 pd.DataFrame(hru_names_dic_swe_mr['ssh'][0]),
                                 pd.DataFrame(hru_names_dic_swe_mr['ssp'][0]),
                                 pd.DataFrame(hru_names_dic_swe_mr['ljc'][0]),
                                 pd.DataFrame(hru_names_dic_swe_mr['ljp'][0]),
                                 pd.DataFrame(hru_names_dic_swe_mr['sjh'][0]),
                                 pd.DataFrame(hru_names_dic_swe_mr['sjp'][0]),
                                 ],ignore_index = True)

    
# reading index of best parameters for each decision combination for swe, melt rate, SST
with open("C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/cc_wrf_swe/p1213_sweMrSstBestParam_index.csv") as saprsweMrSst:
    reader1sweMrSst = csv.reader(saprsweMrSst)
    params2sweMrSst = [r1sweMrSst for r1sweMrSst in reader1sweMrSst]
params_index_swe_mr_sst = np.array(params2sweMrSst[1:])

out_names_swe_mr_sst = ['lsc','lsh','lsp','ssc','ssh','ssp','ljc','ljp','sjh']
params_index_swe_mr_sst_df = pd.DataFrame(params_index_swe_mr_sst, columns = out_names_swe_mr_sst)

hruid_swe_mr_sst_dic = {'lsc': [], 'lsh': [], 'lsp': [], 'ssc': [], 'ssh': [], 'ssp': [], 
                        'ljc': [], 'ljp': [], 'sjh': []}
for keysms in out_names_swe_mr_sst:
    for counterms in range (len(params_index_swe_mr_sst_df)):
        if int(params_index_swe_mr_sst_df[keysms][counterms])>0:
            hruid_swe_mr_sst_dic[keysms].append(int(params_index_swe_mr_sst_df[keysms][counterms]))

hru_names_dic_swe_mr_sst = {'lsc': [], 'lsh': [], 'lsp': [], 'ssc': [], 'ssh': [], 'ssp': [],  
                            'ljc': [], 'ljp': [], 'sjh': []}
for ims in out_names_swe_mr_sst:
    hru_names_dic_swe_mr_sst[ims].append(['{}{}'.format(ims, jms) for jms in hruid_swe_mr_sst_dic[ims]])

hru_names_df_swe_mr_sst = pd.concat([pd.DataFrame(hru_names_dic_swe_mr_sst['lsc'][0]),
                                     pd.DataFrame(hru_names_dic_swe_mr_sst['lsh'][0]),
                                     pd.DataFrame(hru_names_dic_swe_mr_sst['lsp'][0]),
                                     pd.DataFrame(hru_names_dic_swe_mr_sst['ssc'][0]),
                                     pd.DataFrame(hru_names_dic_swe_mr_sst['ssh'][0]),
                                     pd.DataFrame(hru_names_dic_swe_mr_sst['ssp'][0]),
                                     pd.DataFrame(hru_names_dic_swe_mr_sst['ljc'][0]),
                                     pd.DataFrame(hru_names_dic_swe_mr_sst['ljp'][0]),
                                     pd.DataFrame(hru_names_dic_swe_mr_sst['sjh'][0]),
                                     ],ignore_index = True)

    
# reading index of best parameters for each decision combination for swe, melt rate, SST, CC
with open("C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/cc_wrf_swe/p1213_sweMrSstCcBestParam_index.csv") as saprsweMrSstCc:
    reader1sweMrSstCc = csv.reader(saprsweMrSstCc)
    params2sweMrSstCc = [r1sweMrSstCc for r1sweMrSstCc in reader1sweMrSstCc]
params_index_swe_mr_sst_cc = np.array(params2sweMrSstCc[1:])

out_names_swe_mr_sst_cc = ['lsc','lsh','lsp','ssc','ssh','ssp']
params_index_swe_mr_sst_cc_df = pd.DataFrame(params_index_swe_mr_sst_cc, columns = out_names_swe_mr_sst_cc)

hruid_swe_mr_sst_cc_dic = {'lsc': [], 'lsh': [], 'lsp': [], 'ssc': [], 'ssh': [], 'ssp': []}
for keysmsc in out_names_swe_mr_sst_cc:
    for countermsc in range (len(params_index_swe_mr_sst_cc_df)):
        if int(params_index_swe_mr_sst_cc_df[keysmsc][countermsc])>0:
            hruid_swe_mr_sst_cc_dic[keysmsc].append(int(params_index_swe_mr_sst_cc_df[keysmsc][countermsc]))

hru_names_dic_swe_mr_sst_cc = {'lsc': [], 'lsh': [], 'lsp': [], 'ssc': [], 'ssh': [], 'ssp': []}
for imsc in out_names_swe_mr_sst_cc:
    hru_names_dic_swe_mr_sst_cc[imsc].append(['{}{}'.format(imsc, jmsc) for jmsc in hruid_swe_mr_sst_cc_dic[imsc]])

hru_names_df_swe_mr_sst_cc = pd.concat([pd.DataFrame(hru_names_dic_swe_mr_sst_cc['lsc'][0]),
                                        pd.DataFrame(hru_names_dic_swe_mr_sst_cc['lsh'][0]),
                                        pd.DataFrame(hru_names_dic_swe_mr_sst_cc['lsp'][0]),
                                        pd.DataFrame(hru_names_dic_swe_mr_sst_cc['ssc'][0]),
                                        pd.DataFrame(hru_names_dic_swe_mr_sst_cc['ssh'][0]),
                                        pd.DataFrame(hru_names_dic_swe_mr_sst_cc['ssp'][0]),
                                        ],ignore_index = True)

#%%  reading output files for historical
from allNcFiles_wrf import av_ncfiles_h
av_all_h = readAllNcfilesAsDataset(av_ncfiles_h)

DateSa21 = date(av_all_h[0],"%Y-%m-%d") #"%Y-%m-%d %H:%M"
DateSa22 = date(av_all_h[1],"%Y-%m-%d")
date_sa = np.append(DateSa21,DateSa22)
sax = np.arange(0,np.size(date_sa))
sa_xticks = date_sa

# calculating historical swe, sd, and 50%input
av_sd_df_h_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_sd_df_h_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_sd_df_h_lsh.drop(['counter'], axis = 1, inplace = True)
av_sd_df_h_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_sd_df_h_lsp.drop(['counter'], axis = 1, inplace = True)
av_sd_df_h_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_sd_df_h_ssc.drop(['counter'], axis = 1, inplace = True)
av_sd_df_h_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_sd_df_h_ssh.drop(['counter'], axis = 1, inplace = True)
av_sd_df_h_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_sd_df_h_ssp.drop(['counter'], axis = 1, inplace = True)
av_sd_df_h_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_sd_df_h_ljc.drop(['counter'], axis = 1, inplace = True)
av_sd_df_h_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_sd_df_h_ljp.drop(['counter'], axis = 1, inplace = True)
av_sd_df_h_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_sd_df_h_lth.drop(['counter'], axis = 1, inplace = True)
av_sd_df_h_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_sd_df_h_sjh.drop(['counter'], axis = 1, inplace = True)
av_sd_df_h_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_sd_df_h_sjp.drop(['counter'], axis = 1, inplace = True)
av_sd_df_h_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_sd_df_h_stp.drop(['counter'], axis = 1, inplace = True)
av_sd_df_h = pd.concat([av_sd_df_h_lsc,av_sd_df_h_lsh,av_sd_df_h_lsp,av_sd_df_h_ssc,av_sd_df_h_ssh,av_sd_df_h_ssp,av_sd_df_h_ljc,av_sd_df_h_ljp,av_sd_df_h_lth,av_sd_df_h_sjh,av_sd_df_h_sjp,av_sd_df_h_stp], axis = 1)

av_swe_df_h_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSWE',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_swe_df_h_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSWE',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_swe_df_h_lsh.drop(['counter'], axis = 1, inplace = True)
av_swe_df_h_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSWE',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_swe_df_h_lsp.drop(['counter'], axis = 1, inplace = True)
av_swe_df_h_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSWE',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_swe_df_h_ssc.drop(['counter'], axis = 1, inplace = True)
av_swe_df_h_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSWE',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_swe_df_h_ssh.drop(['counter'], axis = 1, inplace = True)
av_swe_df_h_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSWE',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_swe_df_h_ssp.drop(['counter'], axis = 1, inplace = True)
av_swe_df_h_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSWE',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_swe_df_h_ljc.drop(['counter'], axis = 1, inplace = True)
av_swe_df_h_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSWE',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_swe_df_h_ljp.drop(['counter'], axis = 1, inplace = True)
av_swe_df_h_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSWE',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_swe_df_h_lth.drop(['counter'], axis = 1, inplace = True)
av_swe_df_h_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSWE',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_swe_df_h_sjh.drop(['counter'], axis = 1, inplace = True)
av_swe_df_h_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSWE',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_swe_df_h_sjp.drop(['counter'], axis = 1, inplace = True)
av_swe_df_h_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSWE',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_swe_df_h_stp.drop(['counter'], axis = 1, inplace = True)
av_swe_df_h = pd.concat([av_swe_df_h_lsc,av_swe_df_h_lsh,av_swe_df_h_lsp,av_swe_df_h_ssc,av_swe_df_h_ssh,av_swe_df_h_ssp,av_swe_df_h_ljc,av_swe_df_h_ljp,av_swe_df_h_lth,av_swe_df_h_sjh,av_swe_df_h_sjp,av_swe_df_h_stp], axis = 1)

av_rPm_df_h_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_rPm_df_h_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_rPm_df_h_lsh.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_h_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_rPm_df_h_lsp.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_h_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_rPm_df_h_ssc.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_h_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_rPm_df_h_ssh.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_h_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_rPm_df_h_ssp.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_h_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_rPm_df_h_ljc.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_h_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_rPm_df_h_ljp.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_h_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_rPm_df_h_lth.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_h_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_rPm_df_h_sjh.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_h_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_rPm_df_h_sjp.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_h_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_rPm_df_h_stp.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_h = pd.concat([av_rPm_df_h_lsc,av_rPm_df_h_lsh,av_rPm_df_h_lsp,av_rPm_df_h_ssc,av_rPm_df_h_ssh,av_rPm_df_h_ssp,av_rPm_df_h_ljc,av_rPm_df_h_ljp,av_rPm_df_h_lth,av_rPm_df_h_sjh,av_rPm_df_h_sjp,av_rPm_df_h_stp], axis = 1)

#hru_names_df_swe_mr_sst_cc
time0f50input_h_dry_df,time0f50input_h_wet_df = calculatingTiming0f50Pecent0fAnnualInput (av_rPm_df_h,hru_names_df_swe)
time0f50input_h_dry_df_mr = time0f50input_h_dry_df[0][hru_names_df_swe_mr[0]]
time0f50input_h_wet_df_mr = time0f50input_h_wet_df[0][hru_names_df_swe_mr[0]] 
time0f50input_h_dry_df_mr_sst = time0f50input_h_dry_df[0][hru_names_df_swe_mr_sst[0]]
time0f50input_h_wet_df_mr_sst = time0f50input_h_wet_df[0][hru_names_df_swe_mr_sst[0]]  
time0f50input_h_dry_df_mr_sst_cc = time0f50input_h_dry_df[0][hru_names_df_swe_mr_sst_cc[0]]
time0f50input_h_wet_df_mr_sst_cc = time0f50input_h_wet_df[0][hru_names_df_swe_mr_sst_cc[0]]   
  
#  day of snow disappearance (based on snowdepth)-final output
dosd_df_h_dry, dosd_residual_df_h_dry = calculatingSDD(av_sd_df_h[:][5000:8737],hru_names_df_swe,5976,'2007',3737,5000)
dosd_df_h_wet, dosd_residual_df_h_wet = calculatingSDD(av_sd_df_h[:][14000:],hru_names_df_swe,14976,'2008',3521,14000)
dosd_h_dry_df_mr = dosd_df_h_dry[hru_names_df_swe_mr[0]]
dosd_h_wet_df_mr = dosd_df_h_wet[hru_names_df_swe_mr[0]] 
dosd_h_dry_df_mr_sst = dosd_df_h_dry[hru_names_df_swe_mr_sst[0]]
dosd_h_wet_df_mr_sst = dosd_df_h_wet[hru_names_df_swe_mr_sst[0]]  
dosd_h_dry_df_mr_sst_cc = dosd_df_h_dry[hru_names_df_swe_mr_sst_cc[0]]
dosd_h_wet_df_mr_sst_cc = dosd_df_h_wet[hru_names_df_swe_mr_sst_cc[0]]   

# ***********************  finding max corespondance swe for 'h_dry and h_wet'***********************
maxSWE_df_h_dry, maxSWE_df_h_wet = averageMaxSWE4dryAndWetYears (av_swe_df_h,hru_names_df_swe)
maxSWE_h_dry_df_mr = maxSWE_df_h_dry[0][hru_names_df_swe_mr[0]]
maxSWE_h_wet_df_mr = maxSWE_df_h_wet[0][hru_names_df_swe_mr[0]] 
maxSWE_h_dry_df_mr_sst = maxSWE_df_h_dry[0][hru_names_df_swe_mr_sst[0]]
maxSWE_h_wet_df_mr_sst = maxSWE_df_h_wet[0][hru_names_df_swe_mr_sst[0]]  
maxSWE_h_dry_df_mr_sst_cc = maxSWE_df_h_dry[0][hru_names_df_swe_mr_sst_cc[0]]
maxSWE_h_wet_df_mr_sst_cc = maxSWE_df_h_wet[0][hru_names_df_swe_mr_sst_cc[0]]   
   
## *********** calculating snowmelt rate based on SWE #cm/day *************************************
meltingrateAvg_h_dry_df, meltingrateAvg_h_wet_df = meltRateBased0nSWE (hru_names_df_swe,av_swe_df_h,dosd_df_h_dry,dosd_df_h_wet)
meltingrate_h_dry_df_mr = meltingrateAvg_h_dry_df[0][hru_names_df_swe_mr[0]]
meltingrate_h_wet_df_mr = meltingrateAvg_h_wet_df[0][hru_names_df_swe_mr[0]] 
meltingrate_h_dry_df_mr_sst = meltingrateAvg_h_dry_df[0][hru_names_df_swe_mr_sst[0]]
meltingrate_h_wet_df_mr_sst = meltingrateAvg_h_wet_df[0][hru_names_df_swe_mr_sst[0]]  
meltingrate_h_dry_df_mr_sst_cc = meltingrateAvg_h_dry_df[0][hru_names_df_swe_mr_sst_cc[0]]
meltingrate_h_wet_df_mr_sst_cc = meltingrateAvg_h_wet_df[0][hru_names_df_swe_mr_sst_cc[0]]   

#################################################################################################
#%% calculating CC WRF T swe, sd, and 50%input
#################################################################################################
from allNcFiles_wrf import av_ncfiles_T
av_all_T = readAllNcfilesAsDataset(av_ncfiles_T)

# calculating swe, sd, and 50%input
av_sd_df_T_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_sd_df_T_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_sd_df_T_lsh.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_sd_df_T_lsp.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_sd_df_T_ssc.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_sd_df_T_ssh.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_sd_df_T_ssp.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_sd_df_T_ljc.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_sd_df_T_ljp.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_sd_df_T_lth.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_sd_df_T_sjh.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_sd_df_T_sjp.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_sd_df_T_stp.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T = pd.concat([av_sd_df_T_lsc,av_sd_df_T_lsh,av_sd_df_T_lsp,av_sd_df_T_ssc,av_sd_df_T_ssh,av_sd_df_T_ssp,av_sd_df_T_ljc,av_sd_df_T_ljp,av_sd_df_T_lth,av_sd_df_T_sjh,av_sd_df_T_sjp,av_sd_df_T_stp], axis = 1)

av_swe_df_T_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSWE',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_swe_df_T_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSWE',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_swe_df_T_lsh.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSWE',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_swe_df_T_lsp.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSWE',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_swe_df_T_ssc.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSWE',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_swe_df_T_ssh.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSWE',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_swe_df_T_ssp.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSWE',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_swe_df_T_ljc.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSWE',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_swe_df_T_ljp.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSWE',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_swe_df_T_lth.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSWE',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_swe_df_T_sjh.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSWE',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_swe_df_T_sjp.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSWE',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_swe_df_T_stp.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T = pd.concat([av_swe_df_T_lsc,av_swe_df_T_lsh,av_swe_df_T_lsp,av_swe_df_T_ssc,av_swe_df_T_ssh,av_swe_df_T_ssp,av_swe_df_T_ljc,av_swe_df_T_ljp,av_swe_df_T_lth,av_swe_df_T_sjh,av_swe_df_T_sjp,av_swe_df_T_stp], axis = 1)

av_rPm_df_T_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_rPm_df_T_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_rPm_df_T_lsh.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_rPm_df_T_lsp.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_rPm_df_T_ssc.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_rPm_df_T_ssh.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_rPm_df_T_ssp.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_rPm_df_T_ljc.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_rPm_df_T_ljp.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_rPm_df_T_lth.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_rPm_df_T_sjh.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_rPm_df_T_sjp.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_rPm_df_T_stp.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T = pd.concat([av_rPm_df_T_lsc,av_rPm_df_T_lsh,av_rPm_df_T_lsp,av_rPm_df_T_ssc,av_rPm_df_T_ssh,av_rPm_df_T_ssp,av_rPm_df_T_ljc,av_rPm_df_T_ljp,av_rPm_df_T_lth,av_rPm_df_T_sjh,av_rPm_df_T_sjp,av_rPm_df_T_stp], axis = 1)


#  timing of 50 percent of annual input (melt + rain)
time0f50input_T_dry_df,time0f50input_T_wet_df = calculatingTiming0f50Pecent0fAnnualInput (av_rPm_df_T,hru_names_df_swe)
time0f50input_T_dry_df_mr = time0f50input_T_dry_df[0][hru_names_df_swe_mr[0]]
time0f50input_T_wet_df_mr = time0f50input_T_wet_df[0][hru_names_df_swe_mr[0]] 
time0f50input_T_dry_df_mr_sst = time0f50input_T_dry_df[0][hru_names_df_swe_mr_sst[0]]
time0f50input_T_wet_df_mr_sst = time0f50input_T_wet_df[0][hru_names_df_swe_mr_sst[0]]  
time0f50input_T_dry_df_mr_sst_cc = time0f50input_T_dry_df[0][hru_names_df_swe_mr_sst_cc[0]]
time0f50input_T_wet_df_mr_sst_cc = time0f50input_T_wet_df[0][hru_names_df_swe_mr_sst_cc[0]]   
  
#  day of snow disappearance (based on snowdepth)-final output
dosd_df_T_dry, dosd_residual_df_T_dry = calculatingSDD(av_sd_df_T[:][5000:8737],hru_names_df_swe,5976,'2007',3737,5000)
dosd_df_T_wet, dosd_residual_df_T_wet = calculatingSDD(av_sd_df_T[:][14000:],hru_names_df_swe,14976,'2008',3521,14000)
dosd_T_dry_df_mr = dosd_df_T_dry[hru_names_df_swe_mr[0]]
dosd_T_wet_df_mr = dosd_df_T_wet[hru_names_df_swe_mr[0]] 
dosd_T_dry_df_mr_sst = dosd_df_T_dry[hru_names_df_swe_mr_sst[0]]
dosd_T_wet_df_mr_sst = dosd_df_T_wet[hru_names_df_swe_mr_sst[0]]  
dosd_T_dry_df_mr_sst_cc = dosd_df_T_dry[hru_names_df_swe_mr_sst_cc[0]]
dosd_T_wet_df_mr_sst_cc = dosd_df_T_wet[hru_names_df_swe_mr_sst_cc[0]]   

# ***********************  finding max corespondance swe for 'h_dry and h_wet'***********************
maxSWE_df_T_dry, maxSWE_df_T_wet = averageMaxSWE4dryAndWetYears (av_swe_df_T,hru_names_df_swe)
maxSWE_T_dry_df_mr = maxSWE_df_T_dry[0][hru_names_df_swe_mr[0]]
maxSWE_T_wet_df_mr = maxSWE_df_T_wet[0][hru_names_df_swe_mr[0]] 
maxSWE_T_dry_df_mr_sst = maxSWE_df_T_dry[0][hru_names_df_swe_mr_sst[0]]
maxSWE_T_wet_df_mr_sst = maxSWE_df_T_wet[0][hru_names_df_swe_mr_sst[0]]  
maxSWE_T_dry_df_mr_sst_cc = maxSWE_df_T_dry[0][hru_names_df_swe_mr_sst_cc[0]]
maxSWE_T_wet_df_mr_sst_cc = maxSWE_df_T_wet[0][hru_names_df_swe_mr_sst_cc[0]]   
   
## *********** calculating snowmelt rate based on SWE #cm/day *************************************
meltingrateAvg_T_dry_df, meltingrateAvg_T_wet_df = meltRateBased0nSWE (hru_names_df_swe,av_swe_df_T,dosd_df_T_dry,dosd_df_T_wet)
meltingrate_T_dry_df_mr = meltingrateAvg_T_dry_df[0][hru_names_df_swe_mr[0]]
meltingrate_T_wet_df_mr = meltingrateAvg_T_wet_df[0][hru_names_df_swe_mr[0]] 
meltingrate_T_dry_df_mr_sst = meltingrateAvg_T_dry_df[0][hru_names_df_swe_mr_sst[0]]
meltingrate_T_wet_df_mr_sst = meltingrateAvg_T_wet_df[0][hru_names_df_swe_mr_sst[0]]  
meltingrate_T_dry_df_mr_sst_cc = meltingrateAvg_T_dry_df[0][hru_names_df_swe_mr_sst_cc[0]]
meltingrate_T_wet_df_mr_sst_cc = meltingrateAvg_T_wet_df[0][hru_names_df_swe_mr_sst_cc[0]]   

#################################################################################################
#%% calculating CC WRF T swe, sd, and 50%input
#################################################################################################
from allNcFiles_wrf import av_ncfiles_T_al
av_all_T_al = readAllNcfilesAsDataset(av_ncfiles_T_al)

# calculating swe, sd, and 50%input
av_sd_df_T_al_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_sd_df_T_al_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_sd_df_T_al_lsh.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_al_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_sd_df_T_al_lsp.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_al_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_sd_df_T_al_ssc.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_al_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_sd_df_T_al_ssh.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_al_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_sd_df_T_al_ssp.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_al_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_sd_df_T_al_ljc.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_al_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_sd_df_T_al_ljp.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_al_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_sd_df_T_al_lth.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_al_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_sd_df_T_al_sjh.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_al_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_sd_df_T_al_sjp.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_al_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_sd_df_T_al_stp.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_al = pd.concat([av_sd_df_T_al_lsc,av_sd_df_T_al_lsh,av_sd_df_T_al_lsp,av_sd_df_T_al_ssc,av_sd_df_T_al_ssh,av_sd_df_T_al_ssp,av_sd_df_T_al_ljc,av_sd_df_T_al_ljp,av_sd_df_T_al_lth,av_sd_df_T_al_sjh,av_sd_df_T_al_sjp,av_sd_df_T_al_stp], axis = 1)

av_swe_df_T_al_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSWE',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_swe_df_T_al_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSWE',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_swe_df_T_al_lsh.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_al_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSWE',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_swe_df_T_al_lsp.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_al_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSWE',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_swe_df_T_al_ssc.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_al_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSWE',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_swe_df_T_al_ssh.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_al_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSWE',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_swe_df_T_al_ssp.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_al_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSWE',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_swe_df_T_al_ljc.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_al_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSWE',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_swe_df_T_al_ljp.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_al_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSWE',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_swe_df_T_al_lth.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_al_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSWE',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_swe_df_T_al_sjh.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_al_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSWE',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_swe_df_T_al_sjp.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_al_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSWE',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_swe_df_T_al_stp.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_al = pd.concat([av_swe_df_T_al_lsc,av_swe_df_T_al_lsh,av_swe_df_T_al_lsp,av_swe_df_T_al_ssc,av_swe_df_T_al_ssh,av_swe_df_T_al_ssp,av_swe_df_T_al_ljc,av_swe_df_T_al_ljp,av_swe_df_T_al_lth,av_swe_df_T_al_sjh,av_swe_df_T_al_sjp,av_swe_df_T_al_stp], axis = 1)

av_rPm_df_T_al_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_rPm_df_T_al_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_rPm_df_T_al_lsh.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_al_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_rPm_df_T_al_lsp.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_al_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_rPm_df_T_al_ssc.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_al_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_rPm_df_T_al_ssh.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_al_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_rPm_df_T_al_ssp.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_al_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_rPm_df_T_al_ljc.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_al_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_rPm_df_T_al_ljp.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_al_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_rPm_df_T_al_lth.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_al_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_rPm_df_T_al_sjh.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_al_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_rPm_df_T_al_sjp.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_al_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_rPm_df_T_al_stp.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_al = pd.concat([av_rPm_df_T_al_lsc,av_rPm_df_T_al_lsh,av_rPm_df_T_al_lsp,av_rPm_df_T_al_ssc,av_rPm_df_T_al_ssh,av_rPm_df_T_al_ssp,av_rPm_df_T_al_ljc,av_rPm_df_T_al_ljp,av_rPm_df_T_al_lth,av_rPm_df_T_al_sjh,av_rPm_df_T_al_sjp,av_rPm_df_T_al_stp], axis = 1)


#  timing of 50 percent of annual input (melt + rain)
time0f50input_T_al_dry_df,time0f50input_T_al_wet_df = calculatingTiming0f50Pecent0fAnnualInput (av_rPm_df_T_al,hru_names_df_swe)
time0f50input_T_al_dry_df_mr = time0f50input_T_al_dry_df[0][hru_names_df_swe_mr[0]]
time0f50input_T_al_wet_df_mr = time0f50input_T_al_wet_df[0][hru_names_df_swe_mr[0]] 
time0f50input_T_al_dry_df_mr_sst = time0f50input_T_al_dry_df[0][hru_names_df_swe_mr_sst[0]]
time0f50input_T_al_wet_df_mr_sst = time0f50input_T_al_wet_df[0][hru_names_df_swe_mr_sst[0]]  
time0f50input_T_al_dry_df_mr_sst_cc = time0f50input_T_al_dry_df[0][hru_names_df_swe_mr_sst_cc[0]]
time0f50input_T_al_wet_df_mr_sst_cc = time0f50input_T_al_wet_df[0][hru_names_df_swe_mr_sst_cc[0]]   
  
#  day of snow disappearance (based on snowdepth)-final output
dosd_df_T_al_dry, dosd_residual_df_T_al_dry = calculatingSDD(av_sd_df_T_al[:][5000:8737],hru_names_df_swe,5976,'2007',3737,5000)
dosd_df_T_al_wet, dosd_residual_df_T_al_wet = calculatingSDD(av_sd_df_T_al[:][14000:],hru_names_df_swe,14976,'2008',3521,14000)
dosd_T_al_dry_df_mr = dosd_df_T_al_dry[hru_names_df_swe_mr[0]]
dosd_T_al_wet_df_mr = dosd_df_T_al_wet[hru_names_df_swe_mr[0]] 
dosd_T_al_dry_df_mr_sst = dosd_df_T_al_dry[hru_names_df_swe_mr_sst[0]]
dosd_T_al_wet_df_mr_sst = dosd_df_T_al_wet[hru_names_df_swe_mr_sst[0]]  
dosd_T_al_dry_df_mr_sst_cc = dosd_df_T_al_dry[hru_names_df_swe_mr_sst_cc[0]]
dosd_T_al_wet_df_mr_sst_cc = dosd_df_T_al_wet[hru_names_df_swe_mr_sst_cc[0]]   

# ***********************  finding max corespondance swe for 'h_dry and h_wet'***********************
maxSWE_df_T_al_dry, maxSWE_df_T_al_wet = averageMaxSWE4dryAndWetYears (av_swe_df_T_al,hru_names_df_swe)
maxSWE_T_al_dry_df_mr = maxSWE_df_T_al_dry[0][hru_names_df_swe_mr[0]]
maxSWE_T_al_wet_df_mr = maxSWE_df_T_al_wet[0][hru_names_df_swe_mr[0]] 
maxSWE_T_al_dry_df_mr_sst = maxSWE_df_T_al_dry[0][hru_names_df_swe_mr_sst[0]]
maxSWE_T_al_wet_df_mr_sst = maxSWE_df_T_al_wet[0][hru_names_df_swe_mr_sst[0]]  
maxSWE_T_al_dry_df_mr_sst_cc = maxSWE_df_T_al_dry[0][hru_names_df_swe_mr_sst_cc[0]]
maxSWE_T_al_wet_df_mr_sst_cc = maxSWE_df_T_al_wet[0][hru_names_df_swe_mr_sst_cc[0]]   
   
## *********** calculating snowmelt rate based on SWE #cm/day *************************************
meltingrateAvg_T_al_dry_df, meltingrateAvg_T_al_wet_df = meltRateBased0nSWE (hru_names_df_swe,av_swe_df_T_al,dosd_df_T_al_dry,dosd_df_T_al_wet)
meltingrate_T_al_dry_df_mr = meltingrateAvg_T_al_dry_df[0][hru_names_df_swe_mr[0]]
meltingrate_T_al_wet_df_mr = meltingrateAvg_T_al_wet_df[0][hru_names_df_swe_mr[0]] 
meltingrate_T_al_dry_df_mr_sst = meltingrateAvg_T_al_dry_df[0][hru_names_df_swe_mr_sst[0]]
meltingrate_T_al_wet_df_mr_sst = meltingrateAvg_T_al_wet_df[0][hru_names_df_swe_mr_sst[0]]  
meltingrate_T_al_dry_df_mr_sst_cc = meltingrateAvg_T_al_dry_df[0][hru_names_df_swe_mr_sst_cc[0]]
meltingrate_T_al_wet_df_mr_sst_cc = meltingrateAvg_T_al_wet_df[0][hru_names_df_swe_mr_sst_cc[0]]   

#################################################################################################
#%% calculating CC WRF T swe, sd, and 50%input
#################################################################################################
from allNcFiles_wrf import av_ncfiles_T_al_P
av_all_T_al_P = readAllNcfilesAsDataset(av_ncfiles_T_al_P)

# calculating swe, sd, and 50%input
av_sd_df_T_al_P_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_sd_df_T_al_P_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_sd_df_T_al_P_lsh.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_al_P_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_sd_df_T_al_P_lsp.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_al_P_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_sd_df_T_al_P_ssc.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_al_P_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_sd_df_T_al_P_ssh.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_al_P_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_sd_df_T_al_P_ssp.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_al_P_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_sd_df_T_al_P_ljc.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_al_P_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_sd_df_T_al_P_ljp.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_al_P_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_sd_df_T_al_P_lth.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_al_P_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_sd_df_T_al_P_sjh.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_al_P_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_sd_df_T_al_P_sjp.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_al_P_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_sd_df_T_al_P_stp.drop(['counter'], axis = 1, inplace = True)
av_sd_df_T_al_P = pd.concat([av_sd_df_T_al_P_lsc,av_sd_df_T_al_P_lsh,av_sd_df_T_al_P_lsp,av_sd_df_T_al_P_ssc,av_sd_df_T_al_P_ssh,av_sd_df_T_al_P_ssp,av_sd_df_T_al_P_ljc,av_sd_df_T_al_P_ljp,av_sd_df_T_al_P_lth,av_sd_df_T_al_P_sjh,av_sd_df_T_al_P_sjp,av_sd_df_T_al_P_stp], axis = 1)

av_swe_df_T_al_P_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSWE',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_swe_df_T_al_P_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSWE',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_swe_df_T_al_P_lsh.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_al_P_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSWE',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_swe_df_T_al_P_lsp.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_al_P_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSWE',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_swe_df_T_al_P_ssc.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_al_P_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSWE',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_swe_df_T_al_P_ssh.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_al_P_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSWE',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_swe_df_T_al_P_ssp.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_al_P_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSWE',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_swe_df_T_al_P_ljc.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_al_P_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSWE',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_swe_df_T_al_P_ljp.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_al_P_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSWE',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_swe_df_T_al_P_lth.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_al_P_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSWE',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_swe_df_T_al_P_sjh.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_al_P_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSWE',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_swe_df_T_al_P_sjp.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_al_P_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSWE',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_swe_df_T_al_P_stp.drop(['counter'], axis = 1, inplace = True)
av_swe_df_T_al_P = pd.concat([av_swe_df_T_al_P_lsc,av_swe_df_T_al_P_lsh,av_swe_df_T_al_P_lsp,av_swe_df_T_al_P_ssc,av_swe_df_T_al_P_ssh,av_swe_df_T_al_P_ssp,av_swe_df_T_al_P_ljc,av_swe_df_T_al_P_ljp,av_swe_df_T_al_P_lth,av_swe_df_T_al_P_sjh,av_swe_df_T_al_P_sjp,av_swe_df_T_al_P_stp], axis = 1)

av_rPm_df_T_al_P_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_rPm_df_T_al_P_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_rPm_df_T_al_P_lsh.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_al_P_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_rPm_df_T_al_P_lsp.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_al_P_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_rPm_df_T_al_P_ssc.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_al_P_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_rPm_df_T_al_P_ssh.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_al_P_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_rPm_df_T_al_P_ssp.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_al_P_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_rPm_df_T_al_P_ljc.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_al_P_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_rPm_df_T_al_P_ljp.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_al_P_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_rPm_df_T_al_P_lth.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_al_P_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_rPm_df_T_al_P_sjh.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_al_P_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_rPm_df_T_al_P_sjp.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_al_P_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_rPm_df_T_al_P_stp.drop(['counter'], axis = 1, inplace = True)
av_rPm_df_T_al_P = pd.concat([av_rPm_df_T_al_P_lsc,av_rPm_df_T_al_P_lsh,av_rPm_df_T_al_P_lsp,av_rPm_df_T_al_P_ssc,av_rPm_df_T_al_P_ssh,av_rPm_df_T_al_P_ssp,av_rPm_df_T_al_P_ljc,av_rPm_df_T_al_P_ljp,av_rPm_df_T_al_P_lth,av_rPm_df_T_al_P_sjh,av_rPm_df_T_al_P_sjp,av_rPm_df_T_al_P_stp], axis = 1)

#  timing of 50 percent of annual input (melt + rain)
time0f50input_T_al_P_dry_df,time0f50input_T_al_P_wet_df = calculatingTiming0f50Pecent0fAnnualInput (av_rPm_df_T_al_P,hru_names_df_swe)
time0f50input_T_al_P_dry_df_mr = time0f50input_T_al_P_dry_df[0][hru_names_df_swe_mr[0]]
time0f50input_T_al_P_wet_df_mr = time0f50input_T_al_P_wet_df[0][hru_names_df_swe_mr[0]] 
time0f50input_T_al_P_dry_df_mr_sst = time0f50input_T_al_P_dry_df[0][hru_names_df_swe_mr_sst[0]]
time0f50input_T_al_P_wet_df_mr_sst = time0f50input_T_al_P_wet_df[0][hru_names_df_swe_mr_sst[0]]  
time0f50input_T_al_P_dry_df_mr_sst_cc = time0f50input_T_al_P_dry_df[0][hru_names_df_swe_mr_sst_cc[0]]
time0f50input_T_al_P_wet_df_mr_sst_cc = time0f50input_T_al_P_wet_df[0][hru_names_df_swe_mr_sst_cc[0]]   
  
#  day of snow disappearance (based on snowdepth)-final output
dosd_df_T_al_P_dry, dosd_residual_df_T_al_P_dry = calculatingSDD(av_sd_df_T_al_P[:][5000:8737],hru_names_df_swe,5976,'2007',3737,5000)
dosd_df_T_al_P_wet, dosd_residual_df_T_al_P_wet = calculatingSDD(av_sd_df_T_al_P[:][14000:],hru_names_df_swe,14976,'2008',3521,14000)
dosd_T_al_P_dry_df_mr = dosd_df_T_al_P_dry[hru_names_df_swe_mr[0]]
dosd_T_al_P_wet_df_mr = dosd_df_T_al_P_wet[hru_names_df_swe_mr[0]] 
dosd_T_al_P_dry_df_mr_sst = dosd_df_T_al_P_dry[hru_names_df_swe_mr_sst[0]]
dosd_T_al_P_wet_df_mr_sst = dosd_df_T_al_P_wet[hru_names_df_swe_mr_sst[0]]  
dosd_T_al_P_dry_df_mr_sst_cc = dosd_df_T_al_P_dry[hru_names_df_swe_mr_sst_cc[0]]
dosd_T_al_P_wet_df_mr_sst_cc = dosd_df_T_al_P_wet[hru_names_df_swe_mr_sst_cc[0]]   

# ***********************  finding max corespondance swe for 'h_dry and h_wet'***********************
maxSWE_df_T_al_P_dry, maxSWE_df_T_al_P_wet = averageMaxSWE4dryAndWetYears (av_swe_df_T_al_P,hru_names_df_swe)
maxSWE_T_al_P_dry_df_mr = maxSWE_df_T_al_P_dry[0][hru_names_df_swe_mr[0]]
maxSWE_T_al_P_wet_df_mr = maxSWE_df_T_al_P_wet[0][hru_names_df_swe_mr[0]] 
maxSWE_T_al_P_dry_df_mr_sst = maxSWE_df_T_al_P_dry[0][hru_names_df_swe_mr_sst[0]]
maxSWE_T_al_P_wet_df_mr_sst = maxSWE_df_T_al_P_wet[0][hru_names_df_swe_mr_sst[0]]  
maxSWE_T_al_P_dry_df_mr_sst_cc = maxSWE_df_T_al_P_dry[0][hru_names_df_swe_mr_sst_cc[0]]
maxSWE_T_al_P_wet_df_mr_sst_cc = maxSWE_df_T_al_P_wet[0][hru_names_df_swe_mr_sst_cc[0]]   
   
## *********** calculating snowmelt rate based on SWE #cm/day *************************************
meltingrateAvg_T_al_P_dry_df, meltingrateAvg_T_al_P_wet_df = meltRateBased0nSWE (hru_names_df_swe,av_swe_df_T_al_P,dosd_df_T_al_P_dry,dosd_df_T_al_P_wet)
meltingrate_T_al_P_dry_df_mr = meltingrateAvg_T_al_P_dry_df[0][hru_names_df_swe_mr[0]]
meltingrate_T_al_P_wet_df_mr = meltingrateAvg_T_al_P_wet_df[0][hru_names_df_swe_mr[0]] 
meltingrate_T_al_P_dry_df_mr_sst = meltingrateAvg_T_al_P_dry_df[0][hru_names_df_swe_mr_sst[0]]
meltingrate_T_al_P_wet_df_mr_sst = meltingrateAvg_T_al_P_wet_df[0][hru_names_df_swe_mr_sst[0]]  
meltingrate_T_al_P_dry_df_mr_sst_cc = meltingrateAvg_T_al_P_dry_df[0][hru_names_df_swe_mr_sst_cc[0]]
meltingrate_T_al_P_wet_df_mr_sst_cc = meltingrateAvg_T_al_P_wet_df[0][hru_names_df_swe_mr_sst_cc[0]]   


#%% plot max swe
n_groups = 24
#x11 = ['his_wet','T_wet','T_al_wet','T_al_P_wet',
#       'his_dry','T_dry','T_al_dry','T_al_P_dry',
#       
#       'his_wet','T_wet','T_al_wet','T_al_P_wet',
#       'his_dry','T_dry','T_al_dry','T_al_P_dry',
#       
##       'his_wet','T_wet','T_al_wet','T_al_P_wet',
##       'his_dry','T_dry','T_al_dry','T_al_P_dry',
#       
#       'his_wet','T_wet','T_al_wet','T_al_P_wet',
#       'his_dry','T_dry','T_al_dry','T_al_P_dry',
#       ]

x11 = ['historical','Future Temp','T_al_wet','T_al_P_wet',
       'his_dry','T_dry','T_al_dry','T_al_P_dry',
       'his_wet','T_wet','T_al_wet','T_al_P_wet',
       ]

d11 = [maxSWE_df_h_wet[0],maxSWE_df_T_wet[0], 
       maxSWE_df_T_al_wet[0],maxSWE_df_T_al_P_wet[0],
       maxSWE_df_h_dry[0],maxSWE_df_T_dry[0], 
       maxSWE_df_T_al_dry[0],maxSWE_df_T_al_P_dry[0],
       
       maxSWE_h_wet_df_mr,maxSWE_T_wet_df_mr,
       maxSWE_T_al_wet_df_mr,maxSWE_T_al_P_wet_df_mr,
       maxSWE_h_dry_df_mr,maxSWE_T_dry_df_mr,
       maxSWE_T_al_dry_df_mr,maxSWE_T_al_P_dry_df_mr,
       
#       maxSWE_h_wet_df_mr_sst,maxSWE_T_wet_df_mr_sst,
#       maxSWE_T_al_wet_df_mr_sst,maxSWE_T_al_P_wet_df_mr_sst,
#       maxSWE_h_dry_df_mr_sst,maxSWE_T_dry_df_mr_sst,
#       maxSWE_T_al_dry_df_mr_sst,maxSWE_T_al_P_dry_df_mr_sst,
       
       maxSWE_h_wet_df_mr_sst_cc,maxSWE_T_wet_df_mr_sst_cc,
       maxSWE_T_al_wet_df_mr_sst_cc,maxSWE_T_al_P_wet_df_mr_sst_cc,
       maxSWE_h_dry_df_mr_sst_cc,maxSWE_T_dry_df_mr_sst_cc,
       maxSWE_T_al_dry_df_mr_sst_cc,maxSWE_T_al_P_dry_df_mr_sst_cc,
       ]

index = np.arange(n_groups)
bar_width = 0.5

safig, saax = plt.subplots(1,1, figsize=(120,40))#
position = [1,2,3,4,5,6,7,8, 11,12,13,14,15,16,17,18, 21,22,23,24,25,26,27,28]#, 32,33,34,35,36,37,38,39
bp3 = saax.boxplot(d11, patch_artist=True, positions = position, capprops = {'linewidth':5.0},
                   flierprops = dict(marker='o', markersize=16, linestyle='none', markeredgecolor='k'),
                   whiskerprops = {'linewidth':5.0})

bp3['boxes'][0].set(linewidth=8, facecolor = 'deepskyblue', hatch = '*')
bp3['boxes'][1].set(linewidth=8, facecolor = 'deepskyblue', hatch = '*')
bp3['boxes'][2].set(linewidth=8, facecolor = 'deepskyblue', hatch = '*')
bp3['boxes'][3].set(linewidth=8, facecolor = 'deepskyblue', hatch = '*')
bp3['boxes'][4].set(linewidth=8, facecolor = 'lightcyan', hatch = '*')
bp3['boxes'][5].set(linewidth=8, facecolor = 'lightcyan', hatch = '*')
bp3['boxes'][6].set(linewidth=8, facecolor = 'lightcyan', hatch = '*')
bp3['boxes'][7].set(linewidth=8, facecolor = 'lightcyan', hatch = '*')

bp3['boxes'][8].set(linewidth=8, facecolor = 'deepskyblue', hatch = '/')
bp3['boxes'][9].set(linewidth=8, facecolor = 'deepskyblue', hatch = '/')
bp3['boxes'][10].set(linewidth=8, facecolor = 'deepskyblue', hatch = '/')
bp3['boxes'][11].set(linewidth=8, facecolor = 'deepskyblue', hatch = '/')
bp3['boxes'][12].set(linewidth=8, facecolor = 'lightcyan', hatch = '/')
bp3['boxes'][13].set(linewidth=8, facecolor = 'lightcyan', hatch = '/')
bp3['boxes'][14].set(linewidth=8, facecolor = 'lightcyan', hatch = '/')
bp3['boxes'][15].set(linewidth=8, facecolor = 'lightcyan', hatch = '/')

bp3['boxes'][16].set(linewidth=8, facecolor = 'deepskyblue', hatch = 'o')
bp3['boxes'][17].set(linewidth=8, facecolor = 'deepskyblue', hatch = 'o')
bp3['boxes'][18].set(linewidth=8, facecolor = 'deepskyblue', hatch = 'o')
bp3['boxes'][19].set(linewidth=8, facecolor = 'deepskyblue', hatch = 'o')
bp3['boxes'][20].set(linewidth=8, facecolor = 'lightcyan', hatch = 'o')
bp3['boxes'][21].set(linewidth=8, facecolor = 'lightcyan', hatch = 'o')
bp3['boxes'][22].set(linewidth=8, facecolor = 'lightcyan', hatch = 'o')
bp3['boxes'][23].set(linewidth=8, facecolor = 'lightcyan', hatch = 'o')

#bp3['boxes'][24].set(linewidth=8, facecolor = 'deepskyblue', hatch = '\\')
#bp3['boxes'][25].set(linewidth=8, facecolor = 'deepskyblue', hatch = '\\')
#bp3['boxes'][26].set(linewidth=8, facecolor = 'deepskyblue', hatch = '\\')
#bp3['boxes'][27].set(linewidth=8, facecolor = 'deepskyblue', hatch = '\\')
#bp3['boxes'][28].set(linewidth=8, facecolor = 'lightcyan', hatch = '\\')
#bp3['boxes'][29].set(linewidth=8, facecolor = 'lightcyan', hatch = '\\')
#bp3['boxes'][30].set(linewidth=8, facecolor = 'lightcyan', hatch = '\\')
#bp3['boxes'][31].set(linewidth=8, facecolor = 'lightcyan', hatch = '\\')

plt.ylabel('max SWE (mm)', fontsize=80)
plt.yticks(fontsize=80)
plt.xticks(position,x11, fontsize=60, rotation = '25')
plt.title('OF: maxSWE                                                             OF: maxSWE + meltRate                                            OF: maxSWE+meltRate+coldContent', fontsize=90, loc='center')

plt.savefig('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/cc_wrf_swe/sa_cc_swe_swe_mr_sst_cc_all2.png')


#%% plot dosd
d12 = [((dosd_df_h_wet.T-8737)/24.-30.)[0],((dosd_df_T_wet.T-8737)/24.-30.)[0],
       ((dosd_df_T_al_wet.T-8737)/24.-30.)[0],((dosd_df_T_al_P_wet.T-8737)/24.-30.)[0],
       (dosd_df_h_dry.T/24.-30.)[0],(dosd_df_T_dry.T/24.-30.)[0],
       (dosd_df_T_al_dry.T/24.-30.)[0],(dosd_df_T_al_P_dry.T/24.-30.)[0],
       
       ((dosd_h_wet_df_mr.T-8737)/24.-30.)[0],((dosd_T_wet_df_mr.T-8737)/24.-30.)[0],
       ((dosd_T_al_wet_df_mr.T-8737)/24.-30.)[0],((dosd_T_al_P_wet_df_mr.T-8737)/24.-30.)[0],
       (dosd_h_dry_df_mr.T/24.-30.)[0],(dosd_T_dry_df_mr.T/24.-30.)[0],
       (dosd_T_al_dry_df_mr.T/24.-30.)[0],(dosd_T_al_P_dry_df_mr.T/24.-30.)[0],
       
#       ((dosd_h_wet_df_mr_sst.T-8737)/24.-30.)[0],((dosd_T_wet_df_mr_sst.T-8737)/24.-30.)[0],
#       ((dosd_T_al_wet_df_mr_sst.T-8737)/24.-30.)[0],((dosd_T_al_P_wet_df_mr_sst.T-8737)/24.-30.)[0],
#       (dosd_h_dry_df_mr_sst.T/24.-30.)[0],(dosd_T_dry_df_mr_sst.T/24.-30.)[0],
#       (dosd_T_al_dry_df_mr_sst.T/24.-30.)[0],(dosd_T_al_P_dry_df_mr_sst.T/24.-30.)[0],
       
       ((dosd_h_wet_df_mr_sst_cc.T-8737)/24.-30.)[0],((dosd_T_wet_df_mr_sst_cc.T-8737)/24.-30.)[0],
       ((dosd_T_al_wet_df_mr_sst_cc.T-8737)/24.-30.)[0],((dosd_T_al_P_wet_df_mr_sst_cc.T-8737)/24.-30.)[0],
       (dosd_h_dry_df_mr_sst_cc.T/24.-30.)[0],(dosd_T_dry_df_mr_sst_cc.T/24.-30.)[0],
       (dosd_T_al_dry_df_mr_sst_cc.T/24.-30.)[0],(dosd_T_al_P_dry_df_mr_sst_cc.T/24.-30.)[0],
       
       ]

safig2, saax2 = plt.subplots(1,1, figsize=(120,40))#
bp4 = saax2.boxplot(d12, patch_artist=True, positions = position, capprops = {'linewidth':5.0},
                    flierprops = dict(marker='o', markersize=16, linestyle='none', markeredgecolor='k'),
                    whiskerprops = {'linewidth':5.0})

bp4['boxes'][0].set(linewidth=8, facecolor = 'mistyrose', hatch = '*')
bp4['boxes'][1].set(linewidth=8, facecolor = 'mistyrose', hatch = '*')
bp4['boxes'][2].set(linewidth=8, facecolor = 'mistyrose', hatch = '*')
bp4['boxes'][3].set(linewidth=8, facecolor = 'mistyrose', hatch = '*')
bp4['boxes'][4].set(linewidth=8, facecolor = 'salmon', hatch = '*')
bp4['boxes'][5].set(linewidth=8, facecolor = 'salmon', hatch = '*')
bp4['boxes'][6].set(linewidth=8, facecolor = 'salmon', hatch = '*')
bp4['boxes'][7].set(linewidth=8, facecolor = 'salmon', hatch = '*')

bp4['boxes'][8].set(linewidth=8, facecolor = 'mistyrose', hatch = '/')
bp4['boxes'][9].set(linewidth=8, facecolor = 'mistyrose', hatch = '/')
bp4['boxes'][10].set(linewidth=8, facecolor = 'mistyrose', hatch = '/')
bp4['boxes'][11].set(linewidth=8, facecolor = 'mistyrose', hatch = '/')
bp4['boxes'][12].set(linewidth=8, facecolor = 'salmon', hatch = '/')
bp4['boxes'][13].set(linewidth=8, facecolor = 'salmon', hatch = '/')
bp4['boxes'][14].set(linewidth=8, facecolor = 'salmon', hatch = '/')
bp4['boxes'][15].set(linewidth=8, facecolor = 'salmon', hatch = '/')

bp4['boxes'][16].set(linewidth=8, facecolor = 'mistyrose', hatch = 'o')
bp4['boxes'][17].set(linewidth=8, facecolor = 'mistyrose', hatch = 'o')
bp4['boxes'][18].set(linewidth=8, facecolor = 'mistyrose', hatch = 'o')
bp4['boxes'][19].set(linewidth=8, facecolor = 'mistyrose', hatch = 'o')
bp4['boxes'][20].set(linewidth=8, facecolor = 'salmon', hatch = 'o')
bp4['boxes'][21].set(linewidth=8, facecolor = 'salmon', hatch = 'o')
bp4['boxes'][22].set(linewidth=8, facecolor = 'salmon', hatch = 'o')
bp4['boxes'][23].set(linewidth=8, facecolor = 'salmon', hatch = 'o')

#bp4['boxes'][24].set(linewidth=8, facecolor = 'mistyrose', hatch = '\\')
#bp4['boxes'][25].set(linewidth=8, facecolor = 'mistyrose', hatch = '\\')
#bp4['boxes'][26].set(linewidth=8, facecolor = 'mistyrose', hatch = '\\')
#bp4['boxes'][27].set(linewidth=8, facecolor = 'mistyrose', hatch = '\\')
#bp4['boxes'][28].set(linewidth=8, facecolor = 'salmon', hatch = '\\')
#bp4['boxes'][29].set(linewidth=8, facecolor = 'salmon', hatch = '\\')
#bp4['boxes'][30].set(linewidth=8, facecolor = 'salmon', hatch = '\\')
#bp4['boxes'][31].set(linewidth=8, facecolor = 'salmon', hatch = '\\')

plt.ylabel('day of snow disappearance (day of the water year)', fontsize=80)
plt.yticks(fontsize=80)
plt.xticks(position,x11, fontsize=60, rotation = '25')
plt.title('OF: maxSWE                                                             OF: maxSWE + meltRate                                            OF: maxSWE+meltRate+coldContent', fontsize=90, loc='center')

plt.savefig('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/cc_wrf_swe/sa_cc_dosd_swe_mr_sst_cc_all2.png')

#%% timing of 50 input
d13 = [((time0f50input_h_wet_df)/24.)[0],((time0f50input_T_wet_df)/24.)[0],
       ((time0f50input_T_al_wet_df)/24.)[0],((time0f50input_T_al_P_wet_df)/24.)[0],
       (time0f50input_h_dry_df/24.)[0],(time0f50input_T_dry_df/24.)[0],
       (time0f50input_T_al_dry_df/24.)[0],(time0f50input_T_al_P_dry_df/24.)[0],       
       
       (time0f50input_h_wet_df_mr)/24.,((time0f50input_T_wet_df_mr)/24.),
       ((time0f50input_T_al_wet_df_mr)/24.),((time0f50input_T_al_P_wet_df_mr)/24.),
       (time0f50input_h_dry_df_mr/24.),(time0f50input_T_dry_df_mr/24.),
       (time0f50input_T_al_dry_df_mr/24.),(time0f50input_T_al_P_dry_df_mr/24.),
       
#       ((time0f50input_h_wet_df_mr_sst)/24.),((time0f50input_T_wet_df_mr_sst)/24.),
#       ((time0f50input_T_al_wet_df_mr_sst)/24.),((time0f50input_T_al_P_wet_df_mr_sst)/24.),
#       (time0f50input_h_dry_df_mr_sst/24.),(time0f50input_T_dry_df_mr_sst/24.),
#       (time0f50input_T_al_dry_df_mr_sst/24.),(time0f50input_T_al_P_dry_df_mr_sst/24.),
       
       ((time0f50input_h_wet_df_mr_sst_cc)/24.),((time0f50input_T_wet_df_mr_sst_cc)/24.),
       ((time0f50input_T_al_wet_df_mr_sst_cc)/24.),((time0f50input_T_al_P_wet_df_mr_sst_cc)/24.),
       (time0f50input_h_dry_df_mr_sst_cc/24.),(time0f50input_T_dry_df_mr_sst_cc/24.),
       (time0f50input_T_al_dry_df_mr_sst_cc/24.),(time0f50input_T_al_P_dry_df_mr_sst_cc/24.),
       ]

safig3, saax3 = plt.subplots(1,1, figsize=(120,40))#
bp5 = saax3.boxplot(d13, patch_artist=True, positions = position, capprops = {'linewidth':5.0},
                    flierprops = dict(marker='o', markersize=16, linestyle='none', markeredgecolor='k'),
                    whiskerprops = {'linewidth':5.0})

bp5['boxes'][0].set(linewidth=8, facecolor = 'green', hatch = '*')
bp5['boxes'][1].set(linewidth=8, facecolor = 'green', hatch = '*')
bp5['boxes'][2].set(linewidth=8, facecolor = 'green', hatch = '*')
bp5['boxes'][3].set(linewidth=8, facecolor = 'green', hatch = '*')
bp5['boxes'][4].set(linewidth=8, facecolor = 'lightgreen', hatch = '*')
bp5['boxes'][5].set(linewidth=8, facecolor = 'lightgreen', hatch = '*')
bp5['boxes'][6].set(linewidth=8, facecolor = 'lightgreen', hatch = '*')
bp5['boxes'][7].set(linewidth=8, facecolor = 'lightgreen', hatch = '*')

bp5['boxes'][8].set(linewidth=8, facecolor = 'green', hatch = '/')
bp5['boxes'][9].set(linewidth=8, facecolor = 'green', hatch = '/')
bp5['boxes'][10].set(linewidth=8, facecolor = 'green', hatch = '/')
bp5['boxes'][11].set(linewidth=8, facecolor = 'green', hatch = '/')
bp5['boxes'][12].set(linewidth=8, facecolor = 'lightgreen', hatch = '/')
bp5['boxes'][13].set(linewidth=8, facecolor = 'lightgreen', hatch = '/')
bp5['boxes'][14].set(linewidth=8, facecolor = 'lightgreen', hatch = '/')
bp5['boxes'][15].set(linewidth=8, facecolor = 'lightgreen', hatch = '/')

bp5['boxes'][16].set(linewidth=8, facecolor = 'green', hatch = 'o')
bp5['boxes'][17].set(linewidth=8, facecolor = 'green', hatch = 'o')
bp5['boxes'][18].set(linewidth=8, facecolor = 'green', hatch = 'o')
bp5['boxes'][19].set(linewidth=8, facecolor = 'green', hatch = 'o')
bp5['boxes'][20].set(linewidth=8, facecolor = 'lightgreen', hatch = 'o')
bp5['boxes'][21].set(linewidth=8, facecolor = 'lightgreen', hatch = 'o')
bp5['boxes'][22].set(linewidth=8, facecolor = 'lightgreen', hatch = 'o')
bp5['boxes'][23].set(linewidth=8, facecolor = 'lightgreen', hatch = 'o')

#bp5['boxes'][24].set(linewidth=8, facecolor = 'green', hatch = '\\')
#bp5['boxes'][25].set(linewidth=8, facecolor = 'green', hatch = '\\')
#bp5['boxes'][26].set(linewidth=8, facecolor = 'green', hatch = '\\')
#bp5['boxes'][27].set(linewidth=8, facecolor = 'green', hatch = '\\')
#bp5['boxes'][28].set(linewidth=8, facecolor = 'lightgreen', hatch = '\\')
#bp5['boxes'][29].set(linewidth=8, facecolor = 'lightgreen', hatch = '\\')
#bp5['boxes'][30].set(linewidth=8, facecolor = 'lightgreen', hatch = '\\')
#bp5['boxes'][31].set(linewidth=8, facecolor = 'lightgreen', hatch = '\\')

plt.ylabel('timing of 50% input (rain + melt) from Sep 1st (day)', fontsize=80)
plt.yticks(fontsize=80)
plt.xticks(position,x11, fontsize=60, rotation = '25')
plt.title('OF: maxSWE                                                             OF: maxSWE + meltRate                                            OF: maxSWE+meltRate+coldContent', fontsize=90, loc='center')

plt.savefig('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/cc_wrf_swe/sa_cc_time0f50input_swe_mr_sst_cc_all2.png')


#%% timing of 50 input
d14 = [meltingrateAvg_h_wet_df[0],meltingrateAvg_T_wet_df[0],
       meltingrateAvg_T_al_wet_df[0],meltingrateAvg_T_al_P_wet_df[0],
       meltingrateAvg_h_dry_df[0],meltingrateAvg_T_dry_df[0],
       meltingrateAvg_T_al_dry_df[0],meltingrateAvg_T_al_P_dry_df[0],
       
       meltingrate_h_wet_df_mr,meltingrate_T_wet_df_mr,
       meltingrate_T_al_wet_df_mr,meltingrate_T_al_P_wet_df_mr,
       meltingrate_h_dry_df_mr,meltingrate_T_dry_df_mr,
       meltingrate_T_al_dry_df_mr,meltingrate_T_al_P_dry_df_mr,
       
#       meltingrate_h_wet_df_mr_sst,meltingrate_T_wet_df_mr_sst,
#       meltingrate_T_al_wet_df_mr_sst,meltingrate_T_al_P_wet_df_mr_sst,
#       meltingrate_h_dry_df_mr_sst,meltingrate_T_dry_df_mr_sst,
#       meltingrate_T_al_dry_df_mr_sst,meltingrate_T_al_P_dry_df_mr_sst,
       
       meltingrate_h_wet_df_mr_sst_cc,meltingrate_T_wet_df_mr_sst_cc,
       meltingrate_T_al_wet_df_mr_sst_cc,meltingrate_T_al_P_wet_df_mr_sst_cc,
       meltingrate_h_dry_df_mr_sst_cc,meltingrate_T_dry_df_mr_sst_cc,
       meltingrate_T_al_dry_df_mr_sst_cc,meltingrate_T_al_P_dry_df_mr_sst_cc,
       ]

safig4, saax4 = plt.subplots(1,1, figsize=(120,40))#
bp6 = saax4.boxplot(d14, patch_artist=True, positions = position, capprops = {'linewidth':5.0},
                    flierprops = dict(marker='o', markersize=16, linestyle='none', markeredgecolor='k'),
                    whiskerprops = {'linewidth':5.0})

bp6['boxes'][0].set(linewidth=8, facecolor = 'orchid', hatch = '*')
bp6['boxes'][1].set(linewidth=8, facecolor = 'orchid', hatch = '*')
bp6['boxes'][2].set(linewidth=8, facecolor = 'orchid', hatch = '*')
bp6['boxes'][3].set(linewidth=8, facecolor = 'orchid', hatch = '*')
bp6['boxes'][4].set(linewidth=8, facecolor = 'pink', hatch = '*')
bp6['boxes'][5].set(linewidth=8, facecolor = 'pink', hatch = '*')
bp6['boxes'][6].set(linewidth=8, facecolor = 'pink', hatch = '*')
bp6['boxes'][7].set(linewidth=8, facecolor = 'pink', hatch = '*')

bp6['boxes'][8].set(linewidth=8, facecolor = 'orchid', hatch = '/')
bp6['boxes'][9].set(linewidth=8, facecolor = 'orchid', hatch = '/')
bp6['boxes'][10].set(linewidth=8, facecolor = 'orchid', hatch = '/')
bp6['boxes'][11].set(linewidth=8, facecolor = 'orchid', hatch = '/')
bp6['boxes'][12].set(linewidth=8, facecolor = 'pink', hatch = '/')
bp6['boxes'][13].set(linewidth=8, facecolor = 'pink', hatch = '/')
bp6['boxes'][14].set(linewidth=8, facecolor = 'pink', hatch = '/')
bp6['boxes'][15].set(linewidth=8, facecolor = 'pink', hatch = '/')

bp6['boxes'][16].set(linewidth=8, facecolor = 'orchid', hatch = 'o')
bp6['boxes'][17].set(linewidth=8, facecolor = 'orchid', hatch = 'o')
bp6['boxes'][18].set(linewidth=8, facecolor = 'orchid', hatch = 'o')
bp6['boxes'][19].set(linewidth=8, facecolor = 'orchid', hatch = 'o')
bp6['boxes'][20].set(linewidth=8, facecolor = 'pink', hatch = 'o')
bp6['boxes'][21].set(linewidth=8, facecolor = 'pink', hatch = 'o')
bp6['boxes'][22].set(linewidth=8, facecolor = 'pink', hatch = 'o')
bp6['boxes'][23].set(linewidth=8, facecolor = 'pink', hatch = 'o')

#bp6['boxes'][24].set(linewidth=8, facecolor = 'orchid', hatch = '\\')
#bp6['boxes'][25].set(linewidth=8, facecolor = 'orchid', hatch = '\\')
#bp6['boxes'][26].set(linewidth=8, facecolor = 'orchid', hatch = '\\')
#bp6['boxes'][27].set(linewidth=8, facecolor = 'orchid', hatch = '\\')
#bp6['boxes'][28].set(linewidth=8, facecolor = 'pink', hatch = '\\')
#bp6['boxes'][29].set(linewidth=8, facecolor = 'pink', hatch = '\\')
#bp6['boxes'][30].set(linewidth=8, facecolor = 'pink', hatch = '\\')
#bp6['boxes'][31].set(linewidth=8, facecolor = 'pink', hatch = '\\')

plt.ylabel('melting rate (cm/day)', fontsize=80)
plt.yticks(fontsize=80)
plt.xticks(position,x11, fontsize=60, rotation = '25')
plt.title('OF: maxSWE                                                             OF: maxSWE + meltRate                                            OF: maxSWE+meltRate+coldContent', fontsize=90, loc='center')

plt.savefig('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/cc_wrf_swe/sa_cc_meltingrate_swe_mr_sst_cc_all2.png')

#%%
#def plotBoxPlotin2panels(d11_dry,d11_wet,savepath,color,roundn,ylabel,text_position,text_color):
#    
#    x11 = ['historical','Future_Temp','Future_Temp&Energy','Future_Temp&Energy&precip',
#           'historical','Future_Temp','Future_Temp&Energy','Future_Temp&Energy&precip',
#           'historical','Future_Temp','Future_Temp&Energy','Future_Temp&Energy&precip',
#           ]
#
#    safig, saax = plt.subplots(2,1, figsize=(60,40))#
#    saax[0].text(1.5, text_position[0], 'OF: max SWE', fontsize=55, verticalalignment='top')
#    saax[0].text(6.9, text_position[1], 'OF: max SWE & melt rate', fontsize=55, verticalalignment='top')
#    saax[0].text(11.6, text_position[2], 'OF: max SWE&melt rate$cold content', fontsize=55, verticalalignment='top')
#    saax[0].text(7.7, text_position[3], 'Dry Year', color = text_color, fontsize=75, verticalalignment='top')
#    saax[1].text(7.7, text_position[4], 'Wet Year', color = text_color, fontsize=75, verticalalignment='top')
#    
#    position = [1,2,3,4, 7,8,9,10, 13,14,15,16]#, 32,33,34,35,36,37,38,39
#    bp0 = saax[0].boxplot(d11_dry, patch_artist=True, positions = position, capprops = {'linewidth':5.0},
#                          flierprops = dict(marker='o', markersize=16, linestyle='none', markeredgecolor='k'),
#                          whiskerprops = {'linewidth':5.0})
#    bp0['boxes'][0].set(linewidth=8, facecolor = color[0], hatch = '\\')
#    bp0['boxes'][1].set(linewidth=8, facecolor = color[0], hatch = '\\')
#    bp0['boxes'][2].set(linewidth=8, facecolor = color[0], hatch = '\\')
#    bp0['boxes'][3].set(linewidth=8, facecolor = color[0], hatch = '\\')
#    bp0['boxes'][4].set(linewidth=8, facecolor = color[1], hatch = '*')
#    bp0['boxes'][5].set(linewidth=8, facecolor = color[1], hatch = '*')
#    bp0['boxes'][6].set(linewidth=8, facecolor = color[1], hatch = '*')
#    bp0['boxes'][7].set(linewidth=8, facecolor = color[1], hatch = '*')
#    bp0['boxes'][8].set(linewidth=8, facecolor = color[2], hatch = '/')
#    bp0['boxes'][9].set(linewidth=8, facecolor = color[2], hatch = '/')
#    bp0['boxes'][10].set(linewidth=8, facecolor = color[2], hatch = '/')
#    bp0['boxes'][11].set(linewidth=8, facecolor = color[2], hatch = '/')
#    y_tick0=np.round(saax[0].get_yticks(),roundn)
#    saax[0].set_ylabel(ylabel, fontsize=44)
#    saax[0].set_yticklabels(y_tick0,fontsize=40)
#    saax[0].set_xticklabels([], fontsize=50, rotation = '25')#position,
#    
#    bp1 = saax[1].boxplot(d11_wet, patch_artist=True, positions = position, capprops = {'linewidth':5.0},
#                       flierprops = dict(marker='o', markersize=16, linestyle='none', markeredgecolor='k'),
#                       whiskerprops = {'linewidth':5.0})
#    bp1['boxes'][0].set(linewidth=8, facecolor = color[0], hatch = '\\')
#    bp1['boxes'][1].set(linewidth=8, facecolor = color[0], hatch = '\\')
#    bp1['boxes'][2].set(linewidth=8, facecolor = color[0], hatch = '\\')
#    bp1['boxes'][3].set(linewidth=8, facecolor = color[0], hatch = '\\')
#    bp1['boxes'][4].set(linewidth=8, facecolor = color[1], hatch = '*')
#    bp1['boxes'][5].set(linewidth=8, facecolor = color[1], hatch = '*')
#    bp1['boxes'][6].set(linewidth=8, facecolor = color[1], hatch = '*')
#    bp1['boxes'][7].set(linewidth=8, facecolor = color[1], hatch = '*')
#    bp1['boxes'][8].set(linewidth=8, facecolor = color[2], hatch = '/')
#    bp1['boxes'][9].set(linewidth=8, facecolor = color[2], hatch = '/')
#    bp1['boxes'][10].set(linewidth=8, facecolor = color[2], hatch = '/')
#    bp1['boxes'][11].set(linewidth=8, facecolor = color[2], hatch = '/')
#    
#    y_tick1=np.round(saax[1].get_yticks(),roundn)
#    saax[1].set_ylabel(ylabel, fontsize=44)
#    saax[1].set_yticklabels(y_tick1,fontsize=40)
#    saax[1].set_xticklabels(x11, fontsize=50, rotation = '25')#position,
#    
#    plt.savefig(savepath)

#%%
def plotBoxPlotin2panels(d11_dry,d11_wet,savepath,color,ylabel,text_position,text_color,bottom1,top1,roundn): #
    
    x11 = ['historical','Future_Temp','Future_Temp&Energy','Future_Temp&Energy&precip',
           'historical','Future_Temp','Future_Temp&Energy','Future_Temp&Energy&precip',
           'historical','Future_Temp','Future_Temp&Energy','Future_Temp&Energy&precip',
           ]

    safig, saax = plt.subplots(2,1, figsize=(60,40))#
    saax[0].text(0.9, text_position[0], 'OF: max SWE', fontsize=55, verticalalignment='top')
    saax[0].text(6.8, text_position[1], 'OF: max SWE & melt rate', fontsize=55, verticalalignment='top')
    saax[0].text(11.6, text_position[2], 'OF: max SWE&melt rate$cold content', fontsize=55, verticalalignment='top')
    saax[0].text(7.7, text_position[3], 'Dry Year', color = text_color, fontsize=75, verticalalignment='top')
    saax[1].text(7.7, text_position[4], 'Wet Year', color = text_color, fontsize=75, verticalalignment='top')
    
    position = [1,2,3,4, 7,8,9,10, 13,14,15,16]#, 32,33,34,35,36,37,38,39
    bp0 = saax[0].boxplot(d11_dry, patch_artist=True, positions = position, capprops = {'linewidth':5.0},
                          flierprops = dict(marker='o', markersize=16, linestyle='none', markeredgecolor='k'),
                          whiskerprops = {'linewidth':5.0})
    bp0['boxes'][0].set(linewidth=8, facecolor = color[0], hatch = '\\')
    bp0['boxes'][1].set(linewidth=8, facecolor = color[0], hatch = '\\')
    bp0['boxes'][2].set(linewidth=8, facecolor = color[0], hatch = '\\')
    bp0['boxes'][3].set(linewidth=8, facecolor = color[0], hatch = '\\')
    bp0['boxes'][4].set(linewidth=8, facecolor = color[1], hatch = '*')
    bp0['boxes'][5].set(linewidth=8, facecolor = color[1], hatch = '*')
    bp0['boxes'][6].set(linewidth=8, facecolor = color[1], hatch = '*')
    bp0['boxes'][7].set(linewidth=8, facecolor = color[1], hatch = '*')
    bp0['boxes'][8].set(linewidth=8, facecolor = color[2], hatch = '/')
    bp0['boxes'][9].set(linewidth=8, facecolor = color[2], hatch = '/')
    bp0['boxes'][10].set(linewidth=8, facecolor = color[2], hatch = '/')
    bp0['boxes'][11].set(linewidth=8, facecolor = color[2], hatch = '/')
    saax[0].set_ylim([bottom1, top1])
    y_tick0=np.round(saax[0].get_yticks(),roundn)
    #y_tick = np.round(np.arange(bottom1-((top1-bottom1)/6.), top1+(top1-bottom1)/6.,(top1-bottom1)/6.),roundn)
    #print y_tick
    saax[0].set_ylabel(ylabel, fontsize=44)
    saax[0].set_yticklabels(y_tick0,fontsize=40)
    saax[0].set_xticklabels([], fontsize=50, rotation = '25')#position,
    
    bp1 = saax[1].boxplot(d11_wet, patch_artist=True, positions = position, capprops = {'linewidth':5.0},
                       flierprops = dict(marker='o', markersize=16, linestyle='none', markeredgecolor='k'),
                       whiskerprops = {'linewidth':5.0})
    bp1['boxes'][0].set(linewidth=8, facecolor = color[0], hatch = '\\')
    bp1['boxes'][1].set(linewidth=8, facecolor = color[0], hatch = '\\')
    bp1['boxes'][2].set(linewidth=8, facecolor = color[0], hatch = '\\')
    bp1['boxes'][3].set(linewidth=8, facecolor = color[0], hatch = '\\')
    bp1['boxes'][4].set(linewidth=8, facecolor = color[1], hatch = '*')
    bp1['boxes'][5].set(linewidth=8, facecolor = color[1], hatch = '*')
    bp1['boxes'][6].set(linewidth=8, facecolor = color[1], hatch = '*')
    bp1['boxes'][7].set(linewidth=8, facecolor = color[1], hatch = '*')
    bp1['boxes'][8].set(linewidth=8, facecolor = color[2], hatch = '/')
    bp1['boxes'][9].set(linewidth=8, facecolor = color[2], hatch = '/')
    bp1['boxes'][10].set(linewidth=8, facecolor = color[2], hatch = '/')
    bp1['boxes'][11].set(linewidth=8, facecolor = color[2], hatch = '/')

    saax[1].set_ylim([bottom1, top1])
    y_tick1=np.round(saax[1].get_yticks(),roundn)
    saax[1].set_ylabel(ylabel, fontsize=44)
    saax[1].set_yticklabels(y_tick1,fontsize=40)
    saax[1].set_xticklabels(x11, fontsize=50, rotation = '25')#position,
    
    plt.savefig(savepath)
#%%
d110_dry = [maxSWE_df_h_dry[0],maxSWE_df_T_dry[0], 
           maxSWE_df_T_al_dry[0],maxSWE_df_T_al_P_dry[0],
       
           maxSWE_h_dry_df_mr,maxSWE_T_dry_df_mr, 
           maxSWE_T_al_dry_df_mr,maxSWE_T_al_P_dry_df_mr,
           
           maxSWE_h_dry_df_mr_sst_cc,maxSWE_T_dry_df_mr_sst_cc, 
           maxSWE_T_al_dry_df_mr_sst_cc,maxSWE_T_al_P_dry_df_mr_sst_cc,
           ]

d110_wet = [maxSWE_df_h_wet[0],maxSWE_df_T_wet[0], 
           maxSWE_df_T_al_wet[0],maxSWE_df_T_al_P_wet[0],
       
           maxSWE_h_wet_df_mr,maxSWE_T_wet_df_mr, 
           maxSWE_T_al_wet_df_mr,maxSWE_T_al_P_wet_df_mr,
           
           maxSWE_h_wet_df_mr_sst_cc,maxSWE_T_wet_df_mr_sst_cc, 
           maxSWE_T_al_wet_df_mr_sst_cc,maxSWE_T_al_P_wet_df_mr_sst_cc,
           ]

text_position1 = [1172, 1172, 1172, 1290, 1220]
savepath1 = 'C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/cc_wrf_swe/maxSWE_2panel2.png'
ylabel1 = 'Max SWE (mm)'
color1 = ['navy','deepskyblue','lightcyan']
plotBoxPlotin2panels(d110_dry,d110_wet,savepath1,color1,ylabel1,text_position1,'red',-100,1100.,0)

#%% 

d111_dry = [(dosd_df_h_dry.T/24.)[0],(dosd_df_T_dry.T/24.)[0],
            (dosd_df_T_al_dry.T/24.)[0],(dosd_df_T_al_P_dry.T/24.)[0],
       
            (dosd_h_dry_df_mr.T/24.)[0],(dosd_T_dry_df_mr.T/24.)[0],
            (dosd_T_al_dry_df_mr.T/24.)[0],(dosd_T_al_P_dry_df_mr.T/24.)[0],   
            
            (dosd_h_dry_df_mr_sst_cc.T/24.)[0],(dosd_T_dry_df_mr_sst_cc.T/24.)[0],
            (dosd_T_al_dry_df_mr_sst_cc.T/24.)[0],(dosd_T_al_P_dry_df_mr_sst_cc.T/24.)[0],
           ]

d111_wet = [((dosd_df_h_wet.T-8760)/24.)[0],((dosd_df_T_wet.T-8760)/24.)[0],
            ((dosd_df_T_al_wet.T-8760)/24.)[0],((dosd_df_T_al_P_wet.T-8760)/24.)[0],
       
            ((dosd_h_wet_df_mr.T-8760)/24.)[0],((dosd_T_wet_df_mr.T-8760)/24.)[0],
            ((dosd_T_al_wet_df_mr.T-8760)/24.)[0],((dosd_T_al_P_wet_df_mr.T-8760)/24.)[0],
      
            ((dosd_h_wet_df_mr_sst_cc.T-8760)/24.)[0],((dosd_T_wet_df_mr_sst_cc.T-8760)/24.)[0],
            ((dosd_T_al_wet_df_mr_sst_cc.T-8760)/24.)[0],((dosd_T_al_P_wet_df_mr_sst_cc.T-8760)/24.)[0],
            ]

text_position2 = [275, 275, 275, 282, 304]
savepath2 = 'C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/cc_wrf_swe/sdd_2panel1.png'
ylabel2 = 'Day of snow disappearance (day of water year)'
color2 = ['firebrick','salmon','mistyrose']
plotBoxPlotin2panels(d111_dry,d111_wet,savepath2,color2,ylabel2,text_position2,'green',-100,1100.,0)
#%%



d112_dry = [(time0f50input_h_dry_df/24.)[0],(time0f50input_T_dry_df/24.)[0],
            (time0f50input_T_al_dry_df/24.)[0],(time0f50input_T_al_P_dry_df/24.)[0],       
            
            (time0f50input_h_dry_df_mr/24.),(time0f50input_T_dry_df_mr/24.),
            (time0f50input_T_al_dry_df_mr/24.),(time0f50input_T_al_P_dry_df_mr/24.),

           (time0f50input_h_dry_df_mr_sst_cc/24.),(time0f50input_T_dry_df_mr_sst_cc/24.),
           (time0f50input_T_al_dry_df_mr_sst_cc/24.),(time0f50input_T_al_P_dry_df_mr_sst_cc/24.),
           ]

d112_wet = [((time0f50input_h_wet_df)/24.)[0],((time0f50input_T_wet_df)/24.)[0],
            ((time0f50input_T_al_wet_df)/24.)[0],((time0f50input_T_al_P_wet_df)/24.)[0],
       
            (time0f50input_h_wet_df_mr)/24.,((time0f50input_T_wet_df_mr)/24.),
            ((time0f50input_T_al_wet_df_mr)/24.),((time0f50input_T_al_P_wet_df_mr)/24.),
      
            ((time0f50input_h_wet_df_mr_sst_cc)/24.),((time0f50input_T_wet_df_mr_sst_cc)/24.),
            ((time0f50input_T_al_wet_df_mr_sst_cc)/24.),((time0f50input_T_al_P_wet_df_mr_sst_cc)/24.),
            ]

text_position3 = [340, 340, 340, 370, 325]
savepath3 = 'C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/cc_wrf_swe/time0f50input_2panel1.png'
ylabel3 = 'timing of 50% input (rain + melt, day of water year)'
color3 = ['darkgreen','yellowgreen','honeydew']
plotBoxPlotin2panels(d112_dry,d112_wet,savepath3,color3,0,ylabel3,text_position3,'red')


d113_dry = [meltingrateAvg_h_dry_df[0],meltingrateAvg_T_dry_df[0],
            meltingrateAvg_T_al_dry_df[0],meltingrateAvg_T_al_P_dry_df[0],

            meltingrate_h_dry_df_mr,meltingrate_T_dry_df_mr,
            meltingrate_T_al_dry_df_mr,meltingrate_T_al_P_dry_df_mr,
            
            meltingrate_h_dry_df_mr_sst_cc,meltingrate_T_dry_df_mr_sst_cc,
            meltingrate_T_al_dry_df_mr_sst_cc,meltingrate_T_al_P_dry_df_mr_sst_cc,
            ]

d113_wet = [meltingrateAvg_h_wet_df[0],meltingrateAvg_T_wet_df[0],
            meltingrateAvg_T_al_wet_df[0],meltingrateAvg_T_al_P_wet_df[0],
       
            meltingrate_h_wet_df_mr,meltingrate_T_wet_df_mr,
            meltingrate_T_al_wet_df_mr,meltingrate_T_al_P_wet_df_mr,
       
            meltingrate_h_wet_df_mr_sst_cc,meltingrate_T_wet_df_mr_sst_cc,
            meltingrate_T_al_wet_df_mr_sst_cc,meltingrate_T_al_P_wet_df_mr_sst_cc,
            ]

text_position4 = [2.59, 2.59, 2.59, 2.81, 3.5]
savepath4 = 'C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/cc_wrf_swe/meltRate_2panel1.png'
ylabel4 = 'melting rate (cm/day)'
color4 = ['rebeccapurple','mediumpurple','lavender']
plotBoxPlotin2panels(d113_dry,d113_wet,savepath4,color4,2,ylabel4,text_position4,'green')




























 