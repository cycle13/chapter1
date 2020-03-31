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

def calculatingSnowFractionfromSnowfallAndRainfall_dryYear(param_dic,snow_h_swe_dic,rain_h_swe_dic):
    snowFraction_h_swe_dry = []  
    for pdm in range(len(param_dic.keys())):
        snowFractionTimeSeries = np.zeros((len(snow_h_swe_dic[param_dic.keys()[pdm]]),1)) #
        for ts in range (len(snow_h_swe_dic[param_dic.keys()[pdm]])):
            snow_accum = np.sum(snow_h_swe_dic[param_dic.keys()[pdm]][ts][0:5832])
            rain_accum = np.sum(rain_h_swe_dic[param_dic.keys()[pdm]][ts][0:5832])
            snowFractionParam = snow_accum/ (snow_accum+rain_accum)
            snowFractionTimeSeries[ts] = snowFractionParam
        snowFraction_h_swe_dry.append(snowFractionTimeSeries)
    return snowFraction_h_swe_dry

def calculatingSnowFractionfromSnowfallAndRainfall_wetYear(param_dic,snow_h_swe_dic,rain_h_swe_dic):
    snowFraction_h_swe_wet = []  
    for pdm in range(len(param_dic.keys())):
        snowFractionTimeSeries = np.zeros((len(snow_h_swe_dic[param_dic.keys()[pdm]]),1)) #
        for ts in range (len(snow_h_swe_dic[param_dic.keys()[pdm]])):
            snow_accum = np.sum(snow_h_swe_dic[param_dic.keys()[pdm]][ts][8760:14617])
            rain_accum = np.sum(rain_h_swe_dic[param_dic.keys()[pdm]][ts][8760:14617])
            snowFractionParam = snow_accum/ (snow_accum+rain_accum)
            snowFractionTimeSeries[ts] = snowFractionParam
        snowFraction_h_swe_wet.append(snowFractionTimeSeries)
    return snowFraction_h_swe_wet

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

#%% all parameters
with open("C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/cc_wrf_swe/STAR_out_P12_233_L.csv") as safd:
    reader = csv.reader(safd)
    params0 = [r for r in reader]
params1 = params0[1:]
sa_fd_column = []
for csv_counter1 in range (len (params1)):
    for csv_counter2 in range (21):
        sa_fd_column.append(float(params1[csv_counter1][csv_counter2]))
params_sa0=np.reshape(sa_fd_column,(len (params1),21))
params_sa_df12p = pd.DataFrame(params_sa0)#,columns = params0[0]

with open("C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/cc_wrf_swe/STAR_out_P13_233_L.csv") as safd3:
    reader3 = csv.reader(safd3)
    params03 = [r3 for r3 in reader3]
params13 = params03[1:]
sa_fd_column3 = []
for csv_counter13 in range (len (params13)):
    for csv_counter23 in range (22):
        sa_fd_column3.append(float(params13[csv_counter13][csv_counter23]))
params_sa03=np.reshape(sa_fd_column3,(len (params13),22))
params_sa_df13p = pd.DataFrame(params_sa03)#,columns = params0[0]


# reading index of best parameters for each decision model combination 
with open("C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/cc_wrf_swe/p1213_sweBestParam_index.csv") as sapr:
    reader1 = csv.reader(sapr)
    params2 = [r1 for r1 in reader1]
params_index = np.array(params2[1:])
params_index_df = pd.DataFrame(params_index, columns = ['lsc','lsh','lsp','ssc','ssh','ssp',
                                                        'ljc','ljp','lth','sjh','sjp','stp'])

#lsh453  lsp441	lsc457	ssh405	ssp412	ssc401	ljp190	ljc290	lth121	sjh162	sjp182	stp116
hruid_dic = {'lsc': [], 'lsh': [], 'lsp': [], 'ssc': [], 'ssh': [], 'ssp': [],  
             'ljc': [], 'ljp': [], 'lth': [], 'sjh': [], 'sjp': [], 'stp': []}
for keys in params_index_df.columns:
    for counter in range (len(params_index_df)):
        if int(params_index_df[keys][counter])>0:
            hruid_dic[keys].append(int(params_index_df[keys][counter]))

index_dic = {'lsc': [], 'lsh': [], 'lsp': [], 'ssc': [], 'ssh': [], 'ssp': [],  
             'ljc': [], 'ljp': [], 'lth': [], 'sjh': [], 'sjp': [], 'stp': []}
for indx in params_index_df.columns:
    index_array = np.array(hruid_dic[indx])-10000
    index_dic[indx].append(index_array)

param_dic = {'lsc': [], 'lsh': [], 'lsp': [], 'ssc': [], 'ssh': [], 'ssp': [],  
             'ljc': [], 'ljp': [], 'lth': [], 'sjh': [], 'sjp': [], 'stp': []}
for prms in params_index_df.columns:
    if prms[1]=='s':
        params00 = params_sa_df13p.iloc[index_dic[prms][0]]
        param_dic[prms].append(params00)
    else:
        params00 = params_sa_df12p.iloc[index_dic[prms][0]]
        param_dic[prms].append(params00)
    

param_df_ljc = params_sa_df12p.iloc[index_dic['ljc'][0]]
hruidxID = hruid_dic['ljc']
hru_num = np.size(hruidxID)


#%%defining hrus_name

# reading index of best parameters for each decision model combination 
years = ['2007','2008']#

# reading index of best parameters for each decision model combination for swe 0F
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

    
#%% reading index of best parameters for each decision model combination for swe and melt rate 0F
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

index_swe_mr_dic = {'lsc': [], 'lsh': [], 'lsp': [], 'ssc': [], 'ssh': [], 'ssp': [], 
                    'ljc': [], 'ljp': [], 'sjh': [], 'sjp': []}
for indx2 in params_index_swe_mr_df.columns:
    index_array2 = np.array(hruid_swe_mr_dic[indx2])-10000
    index_swe_mr_dic[indx2].append(index_array2)

param_swe_mr_dic = {'lsc': [], 'lsh': [], 'lsp': [], 'ssc': [], 'ssh': [], 'ssp': [], 
                    'ljc': [], 'ljp': [], 'sjh': [], 'sjp': []}
for prms2 in params_index_swe_mr_df.columns:
    if prms2[1]=='s':
        params000 = params_sa_df13p.iloc[index_swe_mr_dic[prms2][0]]
        param_swe_mr_dic[prms2].append(params000)
    else:
        params000 = params_sa_df12p.iloc[index_swe_mr_dic[prms2][0]]
        param_swe_mr_dic[prms2].append(params000)
   
#%% reading index of best parameters for each decision combination for swe, melt rate, SST, CC 0F
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

index_swe_mr_sst_cc_dic = {'lsc': [], 'lsh': [], 'lsp': [], 'ssc': [], 'ssh': [], 'ssp': []}
for indx4 in params_index_swe_mr_sst_cc_df.columns:
    index_array4 = np.array(hruid_swe_mr_sst_cc_dic[indx4])-10000
    index_swe_mr_sst_cc_dic[indx4].append(index_array4)

param_swe_mr_sst_cc_dic = {'lsc': [], 'lsh': [], 'lsp': [], 'ssc': [], 'ssh': [], 'ssp': []}
for prms4 in params_index_swe_mr_sst_cc_df.columns:
    if prms4[1]=='s':
        params00000 = params_sa_df13p.iloc[index_swe_mr_sst_cc_dic[prms4][0]]
        param_swe_mr_sst_cc_dic[prms4].append(params00000)
    else:
        params00000 = params_sa_df12p.iloc[index_swe_mr_sst_cc_dic[prms4][0]]
        param_swe_mr_sst_cc_dic[prms4].append(params00000)

#%%  reading output files for historical
from allNcFiles_wrf import av_ncfiles_h
av_all_h = readAllNcfilesAsDataset(av_ncfiles_h)

DateSa21 = date(av_all_h[0],"%Y-%m-%d") #"%Y-%m-%d %H:%M"
DateSa22 = date(av_all_h[1],"%Y-%m-%d")
date_sa = np.append(DateSa21,DateSa22)
sax = np.arange(0,np.size(date_sa))
sa_xticks = date_sa

# calculating historical swe, sd, and 50%input
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

av_lh_df_h_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_lh_df_h_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_lh_df_h_lsh.drop(['counter'], axis = 1, inplace = True)
av_lh_df_h_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_lh_df_h_lsp.drop(['counter'], axis = 1, inplace = True)
av_lh_df_h_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_lh_df_h_ssc.drop(['counter'], axis = 1, inplace = True)
av_lh_df_h_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_lh_df_h_ssh.drop(['counter'], axis = 1, inplace = True)
av_lh_df_h_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_lh_df_h_ssp.drop(['counter'], axis = 1, inplace = True)
av_lh_df_h_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_lh_df_h_ljc.drop(['counter'], axis = 1, inplace = True)
av_lh_df_h_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_lh_df_h_ljp.drop(['counter'], axis = 1, inplace = True)
av_lh_df_h_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_lh_df_h_lth.drop(['counter'], axis = 1, inplace = True)
av_lh_df_h_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_lh_df_h_sjh.drop(['counter'], axis = 1, inplace = True)
av_lh_df_h_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_lh_df_h_sjp.drop(['counter'], axis = 1, inplace = True)
av_lh_df_h_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_lh_df_h_stp.drop(['counter'], axis = 1, inplace = True)
av_lh_df_h = pd.concat([av_lh_df_h_lsc,av_lh_df_h_lsh,av_lh_df_h_lsp,av_lh_df_h_ssc,av_lh_df_h_ssh,av_lh_df_h_ssp,av_lh_df_h_ljc,av_lh_df_h_ljp,av_lh_df_h_lth,av_lh_df_h_sjh,av_lh_df_h_sjp,av_lh_df_h_stp], axis = 1)

av_gne_df_h_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_gne_df_h_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_gne_df_h_lsh.drop(['counter'], axis = 1, inplace = True)
av_gne_df_h_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_gne_df_h_lsp.drop(['counter'], axis = 1, inplace = True)
av_gne_df_h_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_gne_df_h_ssc.drop(['counter'], axis = 1, inplace = True)
av_gne_df_h_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_gne_df_h_ssh.drop(['counter'], axis = 1, inplace = True)
av_gne_df_h_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_gne_df_h_ssp.drop(['counter'], axis = 1, inplace = True)
av_gne_df_h_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_gne_df_h_ljc.drop(['counter'], axis = 1, inplace = True)
av_gne_df_h_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_gne_df_h_ljp.drop(['counter'], axis = 1, inplace = True)
av_gne_df_h_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_gne_df_h_lth.drop(['counter'], axis = 1, inplace = True)
av_gne_df_h_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_gne_df_h_sjh.drop(['counter'], axis = 1, inplace = True)
av_gne_df_h_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_gne_df_h_sjp.drop(['counter'], axis = 1, inplace = True)
av_gne_df_h_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_gne_df_h_stp.drop(['counter'], axis = 1, inplace = True)
av_gne_df_h = pd.concat([av_gne_df_h_lsc,av_gne_df_h_lsh,av_gne_df_h_lsp,av_gne_df_h_ssc,av_gne_df_h_ssh,av_gne_df_h_ssp,av_gne_df_h_ljc,av_gne_df_h_ljp,av_gne_df_h_lth,av_gne_df_h_sjh,av_gne_df_h_sjp,av_gne_df_h_stp], axis = 1)

av_nlwr_df_h_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_nlwr_df_h_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_nlwr_df_h_lsh.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_h_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_nlwr_df_h_lsp.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_h_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_nlwr_df_h_ssc.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_h_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_nlwr_df_h_ssh.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_h_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_nlwr_df_h_ssp.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_h_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_nlwr_df_h_ljc.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_h_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_nlwr_df_h_ljp.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_h_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_nlwr_df_h_lth.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_h_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_nlwr_df_h_sjh.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_h_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_nlwr_df_h_sjp.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_h_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_nlwr_df_h_stp.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_h = pd.concat([av_nlwr_df_h_lsc,av_nlwr_df_h_lsh,av_nlwr_df_h_lsp,av_nlwr_df_h_ssc,av_nlwr_df_h_ssh,av_nlwr_df_h_ssp,av_nlwr_df_h_ljc,av_nlwr_df_h_ljp,av_nlwr_df_h_lth,av_nlwr_df_h_sjh,av_nlwr_df_h_sjp,av_nlwr_df_h_stp], axis = 1)

av_sh_df_h_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_sh_df_h_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_sh_df_h_lsh.drop(['counter'], axis = 1, inplace = True)
av_sh_df_h_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_sh_df_h_lsp.drop(['counter'], axis = 1, inplace = True)
av_sh_df_h_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_sh_df_h_ssc.drop(['counter'], axis = 1, inplace = True)
av_sh_df_h_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_sh_df_h_ssh.drop(['counter'], axis = 1, inplace = True)
av_sh_df_h_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_sh_df_h_ssp.drop(['counter'], axis = 1, inplace = True)
av_sh_df_h_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_sh_df_h_ljc.drop(['counter'], axis = 1, inplace = True)
av_sh_df_h_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_sh_df_h_ljp.drop(['counter'], axis = 1, inplace = True)
av_sh_df_h_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_sh_df_h_lth.drop(['counter'], axis = 1, inplace = True)
av_sh_df_h_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_sh_df_h_sjh.drop(['counter'], axis = 1, inplace = True)
av_sh_df_h_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_sh_df_h_sjp.drop(['counter'], axis = 1, inplace = True)
av_sh_df_h_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_sh_df_h_stp.drop(['counter'], axis = 1, inplace = True)
av_sh_df_h = pd.concat([av_sh_df_h_lsc,av_sh_df_h_lsh,av_sh_df_h_lsp,av_sh_df_h_ssc,av_sh_df_h_ssh,av_sh_df_h_ssp,av_sh_df_h_ljc,av_sh_df_h_ljp,av_sh_df_h_lth,av_sh_df_h_sjh,av_sh_df_h_sjp,av_sh_df_h_stp], axis = 1)

#%% Dec 1st to June  30th 10 percentile of water input as starting of melting season
rPm_season_h_dry = pd.concat([av_rPm_df_h[z][1464:6552][(av_rPm_df_h[z][1464:6552]>0) & (av_swe_df_h[z][1464:6552]>0) ] for z in av_swe_df_h.columns[1:]],axis = 1) #[av_swe_df_h.columns[1:]]
#rPm_season_h_dry2 = av_rPm_df_h[1464:6552][(av_rPm_df_h[1464:6552]>0) & (av_swe_df_h[1464:6552]>0)] 
start_melt_h_dry = rPm_season_h_dry.quantile(0.1)
aaaa2 = 61+((rPm_season_h_dry['lsc10001']-start_melt_h_dry[0]).abs().sort_values().index[0])/24
#aaaa3 = ((rPm_season['lsc10001']-start_meltSeason[0]).abs().dropna().sort_values()).iloc[0]
#start_meltSeason = pd.concat([pd.DataFrame(61+((rPm_season[j]-start_melt[j]).abs().sort_values().index[0])/24) for j in rPm_season.columns[:]]) #(rPm_season-start_melt).abs().sort_values(axis = 1)
start_meltSeason_h_dry = []
for j in rPm_season_h_dry.columns[:]:
    time50p = 61+((rPm_season_h_dry[j]-start_melt_h_dry[j]).abs().sort_values().index[0])/24
    start_meltSeason_h_dry.append(time50p)
start_meltSeason_df_h_dry = pd.DataFrame(start_meltSeason_h_dry); start_meltSeason_df_h_dry.index = hru_names_df_swe[0]
start_meltSeason_df_h_dry_mr = start_meltSeason_df_h_dry[0][hru_names_df_swe_mr[0]]
start_meltSeason_df_h_dry_mr_cc = start_meltSeason_df_h_dry[0][hru_names_df_swe_mr_sst_cc[0]]

rPm_season_wet = pd.concat([av_rPm_df_h[z][10223:15336][(av_rPm_df_h[z][10223:15336]>0) & (av_swe_df_h[z][10223:15336]>0) ] for z in av_swe_df_h.columns[1:]],axis = 1) #[av_swe_df_h.columns[1:]]
start_melt_wet = rPm_season_wet.quantile(0.1)
start_meltSeason_h_wet = []
for j in rPm_season_wet.columns[:]:
    time50p = 61+(((rPm_season_wet[j]-start_melt_wet[j]).abs().sort_values().index[0])-8760)/24
    start_meltSeason_h_wet.append(time50p)
start_meltSeason_df_h_wet = pd.DataFrame(start_meltSeason_h_wet); start_meltSeason_df_h_wet.index = hru_names_df_swe[0]
start_meltSeason_df_h_wet_mr = start_meltSeason_df_h_wet[0][hru_names_df_swe_mr[0]]
start_meltSeason_df_h_wet_mr_cc = start_meltSeason_df_h_wet[0][hru_names_df_swe_mr_sst_cc[0]]

sms_timeStep_h_dry = pd.DataFrame(start_meltSeason_h_dry)*24.; sms_timeStep_h_dry.index = av_swe_df_h.columns[1:]
sms_timeStep_h_wet = pd.DataFrame(start_meltSeason_h_wet)*24 + 8760.; sms_timeStep_h_dry.index = av_swe_df_h.columns[1:]

#%% Dec 1st to June 30th, sublimation
lh_h_dry = pd.concat([av_lh_df_h[z][0:7000][(av_lh_df_h[z][0:7000]<0) & (av_swe_df_h[z][0:7000]>0) ] for z in av_lh_df_h.columns[1:]],axis = 1) #[av_swe_df_h.columns[1:]]
sublimation_h_dry = lh_h_dry/-25000. #mm/hr
sublimation_yr_h_dry = sublimation_h_dry.sum(axis = 0)
total_precip_h_dry = 486.890424 # total winter precip
sublim_wntrPrecip_h_dry = sublimation_yr_h_dry/total_precip_h_dry*100

lh_h_wet = pd.concat([av_lh_df_h[z][8760:][(av_lh_df_h[z][8760:]<0) & (av_swe_df_h[z][8760:]>0) ] for z in av_lh_df_h.columns[1:]],axis = 1) #[av_swe_df_h.columns[1:]]
sublimation_h_wet = lh_h_wet/-25000. #mm/hr
sublimation_yr_h_wet = sublimation_h_wet.sum(axis = 0)
total_precip_h_dry = 826.5645864 # total winter precip
sublim_wntrPrecip_h_wet = sublimation_yr_h_dry/total_precip_h_dry*100

sublim_wntrPrecip_df_h_dry = pd.DataFrame(sublim_wntrPrecip_h_dry); sublim_wntrPrecip_df_h_dry.index = hru_names_df_swe[0]
sublim_wntrPrecip_df_h_dry_mr = sublim_wntrPrecip_df_h_dry[0][hru_names_df_swe_mr[0]]
sublim_wntrPrecip_df_h_dry_mr_cc = sublim_wntrPrecip_df_h_dry[0][hru_names_df_swe_mr_sst_cc[0]]
sublim_wntrPrecip_df_h_wet = pd.DataFrame(sublim_wntrPrecip_h_wet); sublim_wntrPrecip_df_h_wet.index = hru_names_df_swe[0]
sublim_wntrPrecip_df_h_wet_mr = sublim_wntrPrecip_df_h_wet[0][hru_names_df_swe_mr[0]]
sublim_wntrPrecip_df_h_wet_mr_cc = sublim_wntrPrecip_df_h_wet[0][hru_names_df_swe_mr_sst_cc[0]]

#%% energy fluxes criteria for snow season
av_nswr_df_h = av_gne_df_h - (av_nlwr_df_h + av_lh_df_h + av_sh_df_h)
av_nr_df_h = av_nswr_df_h + av_nlwr_df_h

nr_ss_df_h_dry = pd.concat([av_nr_df_h[z][0:7000][(av_swe_df_h[z][0:7000]>0)] for z in av_swe_df_h.columns[1:]],axis = 1) # ner radiation for snow season  #nr_ss_df_h2 = av_nr_df_h[0:7000][(av_swe_df_h[0:7000]>0)]
nerRad_yr_h_dry = (nr_ss_df_h_dry.sum(axis = 0))/1000. #Kj s-1 m-2
nr_ss_df_h_wet = pd.concat([av_nr_df_h[z][8760:15760][(av_swe_df_h[z][8760:15760]>0)] for z in av_swe_df_h.columns[1:]],axis = 1) # ner radiation for snow season  #nr_ss_df_h2 = av_nr_df_h[0:7000][(av_swe_df_h[0:7000]>0)]
nerRad_yr_h_wet = (nr_ss_df_h_wet.sum(axis = 0))/1000. #Kj s-1 m-2

nswr_ss_df_h_dry = pd.concat([av_nswr_df_h[z][0:7000][(av_swe_df_h[z][0:7000]>0)] for z in av_swe_df_h.columns[1:]],axis = 1) # ner radiation for snow season  #nr_ss_df_h2 = av_nr_df_h[0:7000][(av_swe_df_h[0:7000]>0)]
nswr_yr_h_dry = (nswr_ss_df_h_dry.sum(axis = 0))/1000. #Kj s-1 m-2
nswr_ss_df_h_wet = pd.concat([av_nswr_df_h[z][8760:15760][(av_swe_df_h[z][8760:15760]>0)] for z in av_swe_df_h.columns[1:]],axis = 1) # ner radiation for snow season  #nr_ss_df_h2 = av_nr_df_h[0:7000][(av_swe_df_h[0:7000]>0)]
nswr_yr_h_wet = (nswr_ss_df_h_wet.sum(axis = 0))/1000. #Kj s-1 m-2

nlwr_ss_df_h_dry = pd.concat([av_nlwr_df_h[z][0:7000][(av_swe_df_h[z][0:7000]>0)] for z in av_swe_df_h.columns[1:]],axis = 1) # ner radiation for snow season  #nr_ss_df_h2 = av_nr_df_h[0:7000][(av_swe_df_h[0:7000]>0)]
nlwr_yr_h_dry = (nlwr_ss_df_h_dry.sum(axis = 0))/1000. #Kj s-1 m-2
nlwr_ss_df_h_wet = pd.concat([av_nlwr_df_h[z][8760:15760][(av_swe_df_h[z][8760:15760]>0)] for z in av_swe_df_h.columns[1:]],axis = 1) # ner radiation for snow season  #nr_ss_df_h2 = av_nr_df_h[0:7000][(av_swe_df_h[0:7000]>0)]
nlwr_yr_h_wet = (nlwr_ss_df_h_wet.sum(axis = 0))/1000. #Kj s-1 m-2




#bowen ratio
lh_ss_df_h_dry = pd.concat([av_lh_df_h[z][0:7000][(av_swe_df_h[z][0:7000]>0)] for z in av_swe_df_h.columns[1:]],axis = 1) # ner radiation for snow season  #nr_ss_df_h2 = av_nr_df_h[0:7000][(av_swe_df_h[0:7000]>0)]
lh_yr_h_dry = (lh_ss_df_h_dry.sum(axis = 0))/1000. #Kj s-1 m-2
lh_ss_df_h_wet = pd.concat([av_lh_df_h[z][8760:15760][(av_swe_df_h[z][8760:15760]>0)] for z in av_swe_df_h.columns[1:]],axis = 1) # ner radiation for snow season  #nr_ss_df_h2 = av_nr_df_h[0:7000][(av_swe_df_h[0:7000]>0)]
lh_yr_h_wet = (lh_ss_df_h_wet.sum(axis = 0))/1000. #Kj s-1 m-2

sh_ss_df_h_dry = pd.concat([av_sh_df_h[z][0:7000][(av_swe_df_h[z][0:7000]>0)] for z in av_swe_df_h.columns[1:]],axis = 1) # ner radiation for snow season  #nr_ss_df_h2 = av_nr_df_h[0:7000][(av_swe_df_h[0:7000]>0)]
sh_yr_h_dry = (sh_ss_df_h_dry.sum(axis = 0))/1000. #Kj s-1 m-2
sh_ss_df_h_wet = pd.concat([av_sh_df_h[z][8760:15760][(av_swe_df_h[z][8760:15760]>0)] for z in av_swe_df_h.columns[1:]],axis = 1) # ner radiation for snow season  #nr_ss_df_h2 = av_nr_df_h[0:7000][(av_swe_df_h[0:7000]>0)]
sh_yr_h_wet = (sh_ss_df_h_wet.sum(axis = 0))/1000. #Kj s-1 m-2

BowenRatio_h_dry = sh_yr_h_dry / lh_yr_h_dry
BowenRatio_h_wet = sh_yr_h_wet / lh_yr_h_wet

BowenRatio_df_h_dry = pd.DataFrame(BowenRatio_h_dry); BowenRatio_df_h_dry.index = hru_names_df_swe[0]
BowenRatio_df_h_dry_mr = BowenRatio_df_h_dry[0][hru_names_df_swe_mr[0]]
BowenRatio_df_h_dry_mr_cc = BowenRatio_df_h_dry[0][hru_names_df_swe_mr_sst_cc[0]]
BowenRatio_df_h_wet = pd.DataFrame(BowenRatio_h_wet); BowenRatio_df_h_wet.index = hru_names_df_swe[0]
BowenRatio_df_h_wet_mr = BowenRatio_df_h_wet[0][hru_names_df_swe_mr[0]]
BowenRatio_df_h_wet_mr_cc = BowenRatio_df_h_wet[0][hru_names_df_swe_mr_sst_cc[0]]

#%% energy fluxes criteria for meting season

nr_ms_df_h_dry = pd.concat([av_nr_df_h[z][0:7000][(av_rPm_df_h[z][0:7000]>0) & (av_swe_df_h[z][0:7000]>0)] for z in av_swe_df_h.columns[1:]],axis = 1) # ner radiation for snow season  #nr_ms_df_h2 = av_nr_df_h[0:7000][(av_swe_df_h[0:7000]>0)]
nerRad_accumms_h_dry = (nr_ms_df_h_dry.sum(axis = 0))/1000. #Kj s-1 m-2
nr_ms_df_h_wet = pd.concat([av_nr_df_h[z][8760:15760][(av_rPm_df_h[z][8760:15760]>0) & (av_swe_df_h[z][8760:15760]>0)] for z in av_swe_df_h.columns[1:]],axis = 1) # ner radiation for snow season  #nr_ms_df_h2 = av_nr_df_h[0:7000][(av_swe_df_h[0:7000]>0)]
nerRad_accumms_h_wet = (nr_ms_df_h_wet.sum(axis = 0))/1000. #Kj s-1 m-2

nswr_ms_df_h_dry = pd.concat([av_nswr_df_h[z][0:7000][(av_rPm_df_h[z][0:7000]>0) & (av_swe_df_h[z][0:7000]>0)] for z in av_swe_df_h.columns[1:]],axis = 1) # ner radiation for snow season  #nr_ms_df_h2 = av_nr_df_h[0:7000][(av_swe_df_h[0:7000]>0)]
nswr_accumms_h_dry = (nswr_ms_df_h_dry.sum(axis = 0))/1000. #Kj s-1 m-2
nswr_ms_df_h_wet = pd.concat([av_nswr_df_h[z][8760:15760][(av_rPm_df_h[z][8760:15760]>0) & (av_swe_df_h[z][8760:15760]>0)] for z in av_swe_df_h.columns[1:]],axis = 1) # ner radiation for snow season  #nr_ms_df_h2 = av_nr_df_h[0:7000][(av_swe_df_h[0:7000]>0)]
nswr_accumms_h_wet = (nswr_ms_df_h_wet.sum(axis = 0))/1000. #Kj s-1 m-2

nlwr_ms_df_h_dry = pd.concat([av_nlwr_df_h[z][0:7000][(av_rPm_df_h[z][0:7000]>0) & (av_swe_df_h[z][0:7000]>0)] for z in av_swe_df_h.columns[1:]],axis = 1) # ner radiation for snow season  #nr_ms_df_h2 = av_nr_df_h[0:7000][(av_swe_df_h[0:7000]>0)]
nlwr_accumms_h_dry = (nlwr_ms_df_h_dry.sum(axis = 0))/1000. #Kj s-1 m-2
nlwr_ms_df_h_wet = pd.concat([av_nlwr_df_h[z][8760:15760][(av_rPm_df_h[z][8760:15760]>0) & (av_swe_df_h[z][8760:15760]>0)] for z in av_swe_df_h.columns[1:]],axis = 1) # ner radiation for snow season  #nr_ms_df_h2 = av_nr_df_h[0:7000][(av_swe_df_h[0:7000]>0)]
nlwr_accumms_h_wet = (nlwr_ms_df_h_wet.sum(axis = 0))/1000. #Kj s-1 m-2

radiation_ratio_sw_df_h_wet = pd.DataFrame(nswr_accumms_h_wet/nerRad_accumms_h_wet);radiation_ratio_sw_df_h_wet.index = hru_names_df_swe[0]
radiation_ratio_lw_df_h_wet = pd.DataFrame(nlwr_accumms_h_wet/nerRad_accumms_h_wet);radiation_ratio_lw_df_h_wet.index = hru_names_df_swe[0]
radiation_ratio_sw_df_h_dry = pd.DataFrame(nswr_accumms_h_dry/nerRad_accumms_h_dry);radiation_ratio_sw_df_h_dry.index = hru_names_df_swe[0]
radiation_ratio_lw_df_h_dry = pd.DataFrame(nlwr_accumms_h_dry/nerRad_accumms_h_dry);radiation_ratio_lw_df_h_dry.index = hru_names_df_swe[0]

radiation_ratio_sw_df_h_dry_mr = radiation_ratio_sw_df_h_dry[0][hru_names_df_swe_mr[0]]
radiation_ratio_sw_df_h_dry_mr_cc = radiation_ratio_sw_df_h_dry[0][hru_names_df_swe_mr_sst_cc[0]]
radiation_ratio_sw_df_h_wet_mr = radiation_ratio_sw_df_h_wet[0][hru_names_df_swe_mr[0]]
radiation_ratio_sw_df_h_wet_mr_cc = radiation_ratio_sw_df_h_wet[0][hru_names_df_swe_mr_sst_cc[0]]
radiation_ratio_lw_df_h_dry_mr = radiation_ratio_lw_df_h_dry[0][hru_names_df_swe_mr[0]]
radiation_ratio_lw_df_h_dry_mr_cc = radiation_ratio_lw_df_h_dry[0][hru_names_df_swe_mr_sst_cc[0]]
radiation_ratio_lw_df_h_wet_mr = radiation_ratio_lw_df_h_wet[0][hru_names_df_swe_mr[0]]
radiation_ratio_lw_df_h_wet_mr_cc = radiation_ratio_lw_df_h_wet[0][hru_names_df_swe_mr_sst_cc[0]]

#bowen ratio
lh_ms_df_h_dry = pd.concat([av_lh_df_h[z][0:7000][(av_rPm_df_h[z][0:7000]>0) & (av_swe_df_h[z][0:7000]>0)] for z in av_swe_df_h.columns[1:]],axis = 1) # ner radiation for snow season  #nr_ms_df_h2 = av_nr_df_h[0:7000][(av_swe_df_h[0:7000]>0)]
lh_accumms_h_dry = (lh_ms_df_h_dry.sum(axis = 0))/1000. #Kj s-1 m-2
lh_ms_df_h_wet = pd.concat([av_lh_df_h[z][8760:15760][(av_rPm_df_h[z][8760:15760]>0) & (av_swe_df_h[z][8760:15760]>0)] for z in av_swe_df_h.columns[1:]],axis = 1) # ner radiation for snow season  #nr_ms_df_h2 = av_nr_df_h[0:7000][(av_swe_df_h[0:7000]>0)]
lh_accumms_h_wet = (lh_ms_df_h_wet.sum(axis = 0))/1000. #Kj s-1 m-2

sh_ms_df_h_dry = pd.concat([av_sh_df_h[z][0:7000][(av_rPm_df_h[z][0:7000]>0) & (av_swe_df_h[z][0:7000]>0)] for z in av_swe_df_h.columns[1:]],axis = 1) # ner radiation for snow season  #nr_ms_df_h2 = av_nr_df_h[0:7000][(av_swe_df_h[0:7000]>0)]
sh_accumms_h_dry = (sh_ms_df_h_dry.sum(axis = 0))/1000. #Kj s-1 m-2
sh_ms_df_h_wet = pd.concat([av_sh_df_h[z][8760:15760][(av_rPm_df_h[z][8760:15760]>0) & (av_swe_df_h[z][8760:15760]>0)] for z in av_swe_df_h.columns[1:]],axis = 1) # ner radiation for snow season  #nr_ms_df_h2 = av_nr_df_h[0:7000][(av_swe_df_h[0:7000]>0)]
sh_accumms_h_wet = (sh_ms_df_h_wet.sum(axis = 0))/1000. #Kj s-1 m-2

BowenRatio_ms_h_dry = sh_accumms_h_dry / lh_accumms_h_dry
BowenRatio_ms_h_wet = sh_accumms_h_wet / lh_accumms_h_wet

BowenRatio_ms_df_h_dry = pd.DataFrame(BowenRatio_ms_h_dry); BowenRatio_ms_df_h_dry.index = hru_names_df_swe[0]
BowenRatio_ms_df_h_dry_mr = BowenRatio_ms_df_h_dry[0][hru_names_df_swe_mr[0]]
BowenRatio_ms_df_h_dry_mr_cc = BowenRatio_ms_df_h_dry[0][hru_names_df_swe_mr_sst_cc[0]]
BowenRatio_ms_df_h_wet = pd.DataFrame(BowenRatio_ms_h_wet); BowenRatio_ms_df_h_wet.index = hru_names_df_swe[0]
BowenRatio_ms_df_h_wet_mr = BowenRatio_ms_df_h_wet[0][hru_names_df_swe_mr[0]]
BowenRatio_ms_df_h_wet_mr_cc = BowenRatio_ms_df_h_wet[0][hru_names_df_swe_mr_sst_cc[0]]

#################################################################################################
#%% calculating CC WRF T swe, sd, and 50%input
#################################################################################################
from allNcFiles_wrf import av_ncfiles_T
av_all_T = readAllNcfilesAsDataset(av_ncfiles_T)

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

av_lh_df_T_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_lh_df_T_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_lh_df_T_lsh.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_lh_df_T_lsp.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_lh_df_T_ssc.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_lh_df_T_ssh.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_lh_df_T_ssp.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_lh_df_T_ljc.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_lh_df_T_ljp.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_lh_df_T_lth.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_lh_df_T_sjh.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_lh_df_T_sjp.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_lh_df_T_stp.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T = pd.concat([av_lh_df_T_lsc,av_lh_df_T_lsh,av_lh_df_T_lsp,av_lh_df_T_ssc,av_lh_df_T_ssh,av_lh_df_T_ssp,av_lh_df_T_ljc,av_lh_df_T_ljp,av_lh_df_T_lth,av_lh_df_T_sjh,av_lh_df_T_sjp,av_lh_df_T_stp], axis = 1)

av_gne_df_T_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_gne_df_T_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_gne_df_T_lsh.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_gne_df_T_lsp.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_gne_df_T_ssc.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_gne_df_T_ssh.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_gne_df_T_ssp.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_gne_df_T_ljc.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_gne_df_T_ljp.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_gne_df_T_lth.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_gne_df_T_sjh.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_gne_df_T_sjp.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_gne_df_T_stp.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T = pd.concat([av_gne_df_T_lsc,av_gne_df_T_lsh,av_gne_df_T_lsp,av_gne_df_T_ssc,av_gne_df_T_ssh,av_gne_df_T_ssp,av_gne_df_T_ljc,av_gne_df_T_ljp,av_gne_df_T_lth,av_gne_df_T_sjh,av_gne_df_T_sjp,av_gne_df_T_stp], axis = 1)

av_nlwr_df_T_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_nlwr_df_T_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_nlwr_df_T_lsh.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_nlwr_df_T_lsp.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_nlwr_df_T_ssc.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_nlwr_df_T_ssh.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_nlwr_df_T_ssp.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_nlwr_df_T_ljc.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_nlwr_df_T_ljp.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_nlwr_df_T_lth.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_nlwr_df_T_sjh.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_nlwr_df_T_sjp.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_nlwr_df_T_stp.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T = pd.concat([av_nlwr_df_T_lsc,av_nlwr_df_T_lsh,av_nlwr_df_T_lsp,av_nlwr_df_T_ssc,av_nlwr_df_T_ssh,av_nlwr_df_T_ssp,av_nlwr_df_T_ljc,av_nlwr_df_T_ljp,av_nlwr_df_T_lth,av_nlwr_df_T_sjh,av_nlwr_df_T_sjp,av_nlwr_df_T_stp], axis = 1)

av_sh_df_T_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_sh_df_T_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_sh_df_T_lsh.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_sh_df_T_lsp.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_sh_df_T_ssc.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_sh_df_T_ssh.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_sh_df_T_ssp.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_sh_df_T_ljc.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_sh_df_T_ljp.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_sh_df_T_lth.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_sh_df_T_sjh.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_sh_df_T_sjp.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_sh_df_T_stp.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T = pd.concat([av_sh_df_T_lsc,av_sh_df_T_lsh,av_sh_df_T_lsp,av_sh_df_T_ssc,av_sh_df_T_ssh,av_sh_df_T_ssp,av_sh_df_T_ljc,av_sh_df_T_ljp,av_sh_df_T_lth,av_sh_df_T_sjh,av_sh_df_T_sjp,av_sh_df_T_stp], axis = 1)

#%% Dec 1st to June  30th 10 percentile of water input as starting of melting season
rPm_season_T_dry = pd.concat([av_rPm_df_T[z][1464:6552][(av_rPm_df_T[z][1464:6552]>0) & (av_swe_df_T[z][1464:6552]>0) ] for z in av_swe_df_T.columns[1:]],axis = 1) 
start_melt_T_dry = rPm_season_T_dry.quantile(0.1)
start_meltSeason_T_dry = []
for j in rPm_season_T_dry.columns[:]:
    time50p = 61+((rPm_season_T_dry[j]-start_melt_T_dry[j]).abs().sort_values().index[0])/24
    start_meltSeason_T_dry.append(time50p)

rPm_season_wet = pd.concat([av_rPm_df_T[z][10223:15336][(av_rPm_df_T[z][10223:15336]>0) & (av_swe_df_T[z][10223:15336]>0) ] for z in av_swe_df_T.columns[1:]],axis = 1) 
start_melt_wet = rPm_season_wet.quantile(0.1)
start_meltSeason_T_wet = []
for j in rPm_season_wet.columns[:]:
    time50p = 61+(((rPm_season_wet[j]-start_melt_wet[j]).abs().sort_values().index[0])-8760)/24
    start_meltSeason_T_wet.append(time50p)

sms_timeStep_T_dry = pd.DataFrame(start_meltSeason_T_dry)*24.; sms_timeStep_T_dry.index = av_swe_df_T.columns[1:]
sms_timeStep_T_wet = pd.DataFrame(start_meltSeason_T_wet)*24 + 8760.; sms_timeStep_T_dry.index = av_swe_df_T.columns[1:]

#%% Dec 1st to June 30th, sublimation
lh_T_dry = pd.concat([av_lh_df_T[z][0:7000][(av_lh_df_T[z][0:7000]<0) & (av_swe_df_T[z][0:7000]>0) ] for z in av_lh_df_T.columns[1:]],axis = 1) 
sublimation_T_dry = lh_T_dry/-25000. #mm/hr
sublimation_yr_T_dry = sublimation_T_dry.sum(axis = 0)
total_precip_T_dry = 486.890424
sublim_wntrPrecip_T_dry = sublimation_yr_T_dry/total_precip_T_dry*100

lh_T_wet = pd.concat([av_lh_df_T[z][8760:][(av_lh_df_T[z][8760:]<0) & (av_swe_df_T[z][8760:]>0) ] for z in av_lh_df_T.columns[1:]],axis = 1) 
sublimation_T_wet = lh_T_wet/-25000. #mm/hr
sublimation_yr_T_wet = sublimation_T_wet.sum(axis = 0)
total_precip_T_wet = 826.5645864
sublim_wntrPrecip_T_wet = sublimation_yr_T_dry/total_precip_T_dry*100

#%% energy fluxes criteria for snow season
av_nswr_df_T = av_gne_df_T - (av_nlwr_df_T + av_lh_df_T + av_sh_df_T)
av_nr_df_T = av_nswr_df_T + av_nlwr_df_T

nr_ss_df_T_dry = pd.concat([av_nr_df_T[z][0:7000][(av_swe_df_T[z][0:7000]>0)] for z in av_swe_df_T.columns[1:]],axis = 1) 
nerRad_yr_T_dry = (nr_ss_df_T_dry.sum(axis = 0))/1000. #Kj s-1 m-2
nr_ss_df_T_wet = pd.concat([av_nr_df_T[z][8760:15760][(av_swe_df_T[z][8760:15760]>0)] for z in av_swe_df_T.columns[1:]],axis = 1) 
nerRad_yr_T_wet = (nr_ss_df_T_wet.sum(axis = 0))/1000. #Kj s-1 m-2

nswr_ss_df_T_dry = pd.concat([av_nswr_df_T[z][0:7000][(av_swe_df_T[z][0:7000]>0)] for z in av_swe_df_T.columns[1:]],axis = 1) 
nswr_yr_T_dry = (nswr_ss_df_T_dry.sum(axis = 0))/1000. #Kj s-1 m-2
nswr_ss_df_T_wet = pd.concat([av_nswr_df_T[z][8760:15760][(av_swe_df_T[z][8760:15760]>0)] for z in av_swe_df_T.columns[1:]],axis = 1) 
nswr_yr_T_wet = (nswr_ss_df_T_wet.sum(axis = 0))/1000. #Kj s-1 m-2

nlwr_ss_df_T_dry = pd.concat([av_nlwr_df_T[z][0:7000][(av_swe_df_T[z][0:7000]>0)] for z in av_swe_df_T.columns[1:]],axis = 1) 
nlwr_yr_T_dry = (nlwr_ss_df_T_dry.sum(axis = 0))/1000. #Kj s-1 m-2
nlwr_ss_df_T_wet = pd.concat([av_nlwr_df_T[z][8760:15760][(av_swe_df_T[z][8760:15760]>0)] for z in av_swe_df_T.columns[1:]],axis = 1) 
nlwr_yr_T_wet = (nlwr_ss_df_T_wet.sum(axis = 0))/1000. #Kj s-1 m-2


#bowen ratio
lh_ss_df_T_dry = pd.concat([av_lh_df_T[z][0:7000][(av_swe_df_T[z][0:7000]>0)] for z in av_swe_df_T.columns[1:]],axis = 1) 
lh_yr_T_dry = (lh_ss_df_T_dry.sum(axis = 0))/1000. #Kj s-1 m-2
lh_ss_df_T_wet = pd.concat([av_lh_df_T[z][8760:15760][(av_swe_df_T[z][8760:15760]>0)] for z in av_swe_df_T.columns[1:]],axis = 1) 
lh_yr_T_wet = (lh_ss_df_T_wet.sum(axis = 0))/1000. #Kj s-1 m-2

sh_ss_df_T_dry = pd.concat([av_sh_df_T[z][0:7000][(av_swe_df_T[z][0:7000]>0)] for z in av_swe_df_T.columns[1:]],axis = 1) 
sh_yr_T_dry = (sh_ss_df_T_dry.sum(axis = 0))/1000. #Kj s-1 m-2
sh_ss_df_T_wet = pd.concat([av_sh_df_T[z][8760:15760][(av_swe_df_T[z][8760:15760]>0)] for z in av_swe_df_T.columns[1:]],axis = 1) 
sh_yr_T_wet = (sh_ss_df_T_wet.sum(axis = 0))/1000. #Kj s-1 m-2

BowenRatio_T_dry = sh_yr_T_dry / lh_yr_T_dry
BowenRatio_T_wet = sh_yr_T_wet / lh_yr_T_wet

#%% energy fluxes criteria for meting season

nr_ms_df_T_dry = pd.concat([av_nr_df_T[z][0:7000][(av_rPm_df_T[z][0:7000]>0) & (av_swe_df_T[z][0:7000]>0)] for z in av_swe_df_T.columns[1:]],axis = 1) 
nerRad_accumms_T_dry = (nr_ms_df_T_dry.sum(axis = 0))/1000. #Kj s-1 m-2
nr_ms_df_T_wet = pd.concat([av_nr_df_T[z][8760:15760][(av_rPm_df_T[z][8760:15760]>0) & (av_swe_df_T[z][8760:15760]>0)] for z in av_swe_df_T.columns[1:]],axis = 1) 
nerRad_accumms_T_wet = (nr_ms_df_T_wet.sum(axis = 0))/1000. #Kj s-1 m-2

nswr_ms_df_T_dry = pd.concat([av_nswr_df_T[z][0:7000][(av_rPm_df_T[z][0:7000]>0) & (av_swe_df_T[z][0:7000]>0)] for z in av_swe_df_T.columns[1:]],axis = 1) 
nswr_accumms_T_dry = (nswr_ms_df_T_dry.sum(axis = 0))/1000. #Kj s-1 m-2
nswr_ms_df_T_wet = pd.concat([av_nswr_df_T[z][8760:15760][(av_rPm_df_T[z][8760:15760]>0) & (av_swe_df_T[z][8760:15760]>0)] for z in av_swe_df_T.columns[1:]],axis = 1) 
nswr_accumms_T_wet = (nswr_ms_df_T_wet.sum(axis = 0))/1000. #Kj s-1 m-2

nlwr_ms_df_T_dry = pd.concat([av_nlwr_df_T[z][0:7000][(av_rPm_df_T[z][0:7000]>0) & (av_swe_df_T[z][0:7000]>0)] for z in av_swe_df_T.columns[1:]],axis = 1) 
nlwr_accumms_T_dry = (nlwr_ms_df_T_dry.sum(axis = 0))/1000. #Kj s-1 m-2
nlwr_ms_df_T_wet = pd.concat([av_nlwr_df_T[z][8760:15760][(av_rPm_df_T[z][8760:15760]>0) & (av_swe_df_T[z][8760:15760]>0)] for z in av_swe_df_T.columns[1:]],axis = 1) 
nlwr_accumms_T_wet = (nlwr_ms_df_T_wet.sum(axis = 0))/1000. #Kj s-1 m-2

#radiation_ratio
radiation_ratio_sw_df_T_wet = pd.DataFrame(nswr_accumms_T_wet/nerRad_accumms_T_wet);radiation_ratio_sw_df_T_wet.index = hru_names_df_swe[0]
radiation_ratio_lw_df_T_wet = pd.DataFrame(nlwr_accumms_T_wet/nerRad_accumms_T_wet);radiation_ratio_lw_df_T_wet.index = hru_names_df_swe[0]
radiation_ratio_sw_df_T_dry = pd.DataFrame(nswr_accumms_T_dry/nerRad_accumms_T_dry);radiation_ratio_sw_df_T_dry.index = hru_names_df_swe[0]
radiation_ratio_lw_df_T_dry = pd.DataFrame(nlwr_accumms_T_dry/nerRad_accumms_T_dry);radiation_ratio_lw_df_T_dry.index = hru_names_df_swe[0]

radiation_ratio_sw_df_T_dry_mr = radiation_ratio_sw_df_T_dry[0][hru_names_df_swe_mr[0]]
radiation_ratio_sw_df_T_dry_mr_cc = radiation_ratio_sw_df_T_dry[0][hru_names_df_swe_mr_sst_cc[0]]
radiation_ratio_sw_df_T_wet_mr = radiation_ratio_sw_df_T_wet[0][hru_names_df_swe_mr[0]]
radiation_ratio_sw_df_T_wet_mr_cc = radiation_ratio_sw_df_T_wet[0][hru_names_df_swe_mr_sst_cc[0]]
radiation_ratio_lw_df_T_dry_mr = radiation_ratio_lw_df_T_dry[0][hru_names_df_swe_mr[0]]
radiation_ratio_lw_df_T_dry_mr_cc = radiation_ratio_lw_df_T_dry[0][hru_names_df_swe_mr_sst_cc[0]]
radiation_ratio_lw_df_T_wet_mr = radiation_ratio_lw_df_T_wet[0][hru_names_df_swe_mr[0]]
radiation_ratio_lw_df_T_wet_mr_cc = radiation_ratio_lw_df_T_wet[0][hru_names_df_swe_mr_sst_cc[0]]



#bowen ratio
lh_ms_df_T_dry = pd.concat([av_lh_df_T[z][0:7000][(av_rPm_df_T[z][0:7000]>0) & (av_swe_df_T[z][0:7000]>0)] for z in av_swe_df_T.columns[1:]],axis = 1) 
lh_accumms_T_dry = (lh_ms_df_T_dry.sum(axis = 0))/1000. #Kj s-1 m-2
lh_ms_df_T_wet = pd.concat([av_lh_df_T[z][8760:15760][(av_rPm_df_T[z][8760:15760]>0) & (av_swe_df_T[z][8760:15760]>0)] for z in av_swe_df_T.columns[1:]],axis = 1) 
lh_accumms_T_wet = (lh_ms_df_T_wet.sum(axis = 0))/1000. #Kj s-1 m-2

sh_ms_df_T_dry = pd.concat([av_sh_df_T[z][0:7000][(av_rPm_df_T[z][0:7000]>0) & (av_swe_df_T[z][0:7000]>0)] for z in av_swe_df_T.columns[1:]],axis = 1) 
sh_accumms_T_dry = (sh_ms_df_T_dry.sum(axis = 0))/1000. #Kj s-1 m-2
sh_ms_df_T_wet = pd.concat([av_sh_df_T[z][8760:15760][(av_rPm_df_T[z][8760:15760]>0) & (av_swe_df_T[z][8760:15760]>0)] for z in av_swe_df_T.columns[1:]],axis = 1) 
sh_accumms_T_wet = (sh_ms_df_T_wet.sum(axis = 0))/1000. #Kj s-1 m-2

BowenRatio_ms_T_dry = sh_accumms_T_dry / lh_accumms_T_dry
BowenRatio_ms_T_wet = sh_accumms_T_wet / lh_accumms_T_wet

#%%###################################################################################################
# applying different objective functions
start_meltSeason_df_T_dry = pd.DataFrame(start_meltSeason_T_dry); start_meltSeason_df_T_dry.index = hru_names_df_swe[0]
start_meltSeason_df_T_dry_mr = start_meltSeason_df_T_dry[0][hru_names_df_swe_mr[0]]
start_meltSeason_df_T_dry_mr_cc = start_meltSeason_df_T_dry[0][hru_names_df_swe_mr_sst_cc[0]]
start_meltSeason_df_T_wet = pd.DataFrame(start_meltSeason_T_wet); start_meltSeason_df_T_wet.index = hru_names_df_swe[0]
start_meltSeason_df_T_wet_mr = start_meltSeason_df_T_wet[0][hru_names_df_swe_mr[0]]
start_meltSeason_df_T_wet_mr_cc = start_meltSeason_df_T_wet[0][hru_names_df_swe_mr_sst_cc[0]]

sublim_wntrPrecip_df_T_dry = pd.DataFrame(sublim_wntrPrecip_T_dry); sublim_wntrPrecip_df_T_dry.index = hru_names_df_swe[0]
sublim_wntrPrecip_df_T_dry_mr = sublim_wntrPrecip_df_T_dry[0][hru_names_df_swe_mr[0]]
sublim_wntrPrecip_df_T_dry_mr_cc = sublim_wntrPrecip_df_T_dry[0][hru_names_df_swe_mr_sst_cc[0]]
sublim_wntrPrecip_df_T_wet = pd.DataFrame(sublim_wntrPrecip_T_wet); sublim_wntrPrecip_df_T_wet.index = hru_names_df_swe[0]
sublim_wntrPrecip_df_T_wet_mr = sublim_wntrPrecip_df_T_wet[0][hru_names_df_swe_mr[0]]
sublim_wntrPrecip_df_T_wet_mr_cc = sublim_wntrPrecip_df_T_wet[0][hru_names_df_swe_mr_sst_cc[0]]

BowenRatio_df_T_dry = pd.DataFrame(BowenRatio_T_dry); BowenRatio_df_T_dry.index = hru_names_df_swe[0]
BowenRatio_df_T_dry_mr = BowenRatio_df_T_dry[0][hru_names_df_swe_mr[0]]
BowenRatio_df_T_dry_mr_cc = BowenRatio_df_T_dry[0][hru_names_df_swe_mr_sst_cc[0]]
BowenRatio_df_T_wet = pd.DataFrame(BowenRatio_T_wet); BowenRatio_df_T_wet.index = hru_names_df_swe[0]
BowenRatio_df_T_wet_mr = BowenRatio_df_T_wet[0][hru_names_df_swe_mr[0]]
BowenRatio_df_T_wet_mr_cc = BowenRatio_df_T_wet[0][hru_names_df_swe_mr_sst_cc[0]]

BowenRatio_ms_df_T_dry = pd.DataFrame(BowenRatio_ms_T_dry); BowenRatio_ms_df_T_dry.index = hru_names_df_swe[0]
BowenRatio_ms_df_T_dry_mr = BowenRatio_ms_df_T_dry[0][hru_names_df_swe_mr[0]]
BowenRatio_ms_df_T_dry_mr_cc = BowenRatio_ms_df_T_dry[0][hru_names_df_swe_mr_sst_cc[0]]
BowenRatio_ms_df_T_wet = pd.DataFrame(BowenRatio_ms_T_wet); BowenRatio_ms_df_T_wet.index = hru_names_df_swe[0]
BowenRatio_ms_df_T_wet_mr = BowenRatio_ms_df_T_wet[0][hru_names_df_swe_mr[0]]
BowenRatio_ms_df_T_wet_mr_cc = BowenRatio_ms_df_T_wet[0][hru_names_df_swe_mr_sst_cc[0]]

#################################################################################################
#%% calculating CC WRF T swe, sd, and 50%input
#################################################################################################
from allNcFiles_wrf import av_ncfiles_T_al
av_all_T_al = readAllNcfilesAsDataset(av_ncfiles_T_al)

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

av_lh_df_T_al_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_lh_df_T_al_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_lh_df_T_al_lsh.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_al_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_lh_df_T_al_lsp.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_al_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_lh_df_T_al_ssc.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_al_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_lh_df_T_al_ssh.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_al_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_lh_df_T_al_ssp.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_al_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_lh_df_T_al_ljc.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_al_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_lh_df_T_al_ljp.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_al_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_lh_df_T_al_lth.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_al_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_lh_df_T_al_sjh.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_al_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_lh_df_T_al_sjp.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_al_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_lh_df_T_al_stp.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_al = pd.concat([av_lh_df_T_al_lsc,av_lh_df_T_al_lsh,av_lh_df_T_al_lsp,av_lh_df_T_al_ssc,av_lh_df_T_al_ssh,av_lh_df_T_al_ssp,av_lh_df_T_al_ljc,av_lh_df_T_al_ljp,av_lh_df_T_al_lth,av_lh_df_T_al_sjh,av_lh_df_T_al_sjp,av_lh_df_T_al_stp], axis = 1)

av_gne_df_T_al_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_gne_df_T_al_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_gne_df_T_al_lsh.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_al_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_gne_df_T_al_lsp.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_al_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_gne_df_T_al_ssc.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_al_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_gne_df_T_al_ssh.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_al_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_gne_df_T_al_ssp.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_al_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_gne_df_T_al_ljc.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_al_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_gne_df_T_al_ljp.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_al_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_gne_df_T_al_lth.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_al_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_gne_df_T_al_sjh.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_al_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_gne_df_T_al_sjp.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_al_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_gne_df_T_al_stp.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_al = pd.concat([av_gne_df_T_al_lsc,av_gne_df_T_al_lsh,av_gne_df_T_al_lsp,av_gne_df_T_al_ssc,av_gne_df_T_al_ssh,av_gne_df_T_al_ssp,av_gne_df_T_al_ljc,av_gne_df_T_al_ljp,av_gne_df_T_al_lth,av_gne_df_T_al_sjh,av_gne_df_T_al_sjp,av_gne_df_T_al_stp], axis = 1)

av_nlwr_df_T_al_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_nlwr_df_T_al_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_nlwr_df_T_al_lsh.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_al_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_nlwr_df_T_al_lsp.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_al_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_nlwr_df_T_al_ssc.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_al_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_nlwr_df_T_al_ssh.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_al_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_nlwr_df_T_al_ssp.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_al_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_nlwr_df_T_al_ljc.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_al_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_nlwr_df_T_al_ljp.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_al_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_nlwr_df_T_al_lth.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_al_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_nlwr_df_T_al_sjh.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_al_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_nlwr_df_T_al_sjp.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_al_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_nlwr_df_T_al_stp.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_al = pd.concat([av_nlwr_df_T_al_lsc,av_nlwr_df_T_al_lsh,av_nlwr_df_T_al_lsp,av_nlwr_df_T_al_ssc,av_nlwr_df_T_al_ssh,av_nlwr_df_T_al_ssp,av_nlwr_df_T_al_ljc,av_nlwr_df_T_al_ljp,av_nlwr_df_T_al_lth,av_nlwr_df_T_al_sjh,av_nlwr_df_T_al_sjp,av_nlwr_df_T_al_stp], axis = 1)

av_sh_df_T_al_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_sh_df_T_al_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_sh_df_T_al_lsh.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_al_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_sh_df_T_al_lsp.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_al_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_sh_df_T_al_ssc.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_al_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_sh_df_T_al_ssh.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_al_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_sh_df_T_al_ssp.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_al_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_sh_df_T_al_ljc.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_al_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_sh_df_T_al_ljp.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_al_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_sh_df_T_al_lth.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_al_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_sh_df_T_al_sjh.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_al_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_sh_df_T_al_sjp.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_al_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_sh_df_T_al_stp.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_al = pd.concat([av_sh_df_T_al_lsc,av_sh_df_T_al_lsh,av_sh_df_T_al_lsp,av_sh_df_T_al_ssc,av_sh_df_T_al_ssh,av_sh_df_T_al_ssp,av_sh_df_T_al_ljc,av_sh_df_T_al_ljp,av_sh_df_T_al_lth,av_sh_df_T_al_sjh,av_sh_df_T_al_sjp,av_sh_df_T_al_stp], axis = 1)

#%% Dec 1st to June  30th 10 percentile of water input as starting of melting season
rPm_season_T_al_dry = pd.concat([av_rPm_df_T_al[z][1464:6552][(av_rPm_df_T_al[z][1464:6552]>0) & (av_swe_df_T_al[z][1464:6552]>0) ] for z in av_swe_df_T_al.columns[1:]],axis = 1) 
start_melt_T_al_dry = rPm_season_T_al_dry.quantile(0.1)
start_meltSeason_T_al_dry = []
for j in rPm_season_T_al_dry.columns[:]:
    time50p = 61+((rPm_season_T_al_dry[j]-start_melt_T_al_dry[j]).abs().sort_values().index[0])/24
    start_meltSeason_T_al_dry.append(time50p)

rPm_season_wet = pd.concat([av_rPm_df_T_al[z][10223:15336][(av_rPm_df_T_al[z][10223:15336]>0) & (av_swe_df_T_al[z][10223:15336]>0) ] for z in av_swe_df_T_al.columns[1:]],axis = 1) 
start_melt_wet = rPm_season_wet.quantile(0.1)
start_meltSeason_T_al_wet = []
for j in rPm_season_wet.columns[:]:
    time50p = 61+(((rPm_season_wet[j]-start_melt_wet[j]).abs().sort_values().index[0])-8760)/24
    start_meltSeason_T_al_wet.append(time50p)

sms_timeStep_T_al_dry = pd.DataFrame(start_meltSeason_T_al_dry)*24.; sms_timeStep_T_al_dry.index = av_swe_df_T_al.columns[1:]
sms_timeStep_T_al_wet = pd.DataFrame(start_meltSeason_T_al_wet)*24 + 8760.; sms_timeStep_T_al_wet.index = av_swe_df_T_al.columns[1:]

#%% Dec 1st to June 30th, sublimation
lh_T_al_dry = pd.concat([av_lh_df_T_al[z][0:7000][(av_lh_df_T_al[z][0:7000]<0) & (av_swe_df_T_al[z][0:7000]>0) ] for z in av_lh_df_T_al.columns[1:]],axis = 1) 
sublimation_T_al_dry = lh_T_al_dry/-25000. #mm/hr
sublimation_yr_T_al_dry = sublimation_T_al_dry.sum(axis = 0)
total_precip_T_al_dry = 486.890424
sublim_wntrPrecip_T_al_dry = sublimation_yr_T_al_dry/total_precip_T_al_dry*100

lh_T_al_wet = pd.concat([av_lh_df_T_al[z][8760:][(av_lh_df_T_al[z][8760:]<0) & (av_swe_df_T_al[z][8760:]>0) ] for z in av_lh_df_T_al.columns[1:]],axis = 1) 
sublimation_T_al_wet = lh_T_al_wet/-25000. #mm/hr
sublimation_yr_T_al_wet = sublimation_T_al_wet.sum(axis = 0)
total_precip_T_al_wet = 826.5645864
sublim_wntrPrecip_T_al_wet = sublimation_yr_T_al_dry/total_precip_T_al_dry*100

#%% energy fluxes criteria for snow season
av_nswr_df_T_al = av_gne_df_T_al - (av_nlwr_df_T_al + av_lh_df_T_al + av_sh_df_T_al)
av_nr_df_T_al = av_nswr_df_T_al + av_nlwr_df_T_al

nr_ss_df_T_al_dry = pd.concat([av_nr_df_T_al[z][0:7000][(av_swe_df_T_al[z][0:7000]>0)] for z in av_swe_df_T_al.columns[1:]],axis = 1) 
nerRad_yr_T_al_dry = (nr_ss_df_T_al_dry.sum(axis = 0))/1000. #Kj s-1 m-2
nr_ss_df_T_al_wet = pd.concat([av_nr_df_T_al[z][8760:15760][(av_swe_df_T_al[z][8760:15760]>0)] for z in av_swe_df_T_al.columns[1:]],axis = 1) 
nerRad_yr_T_al_wet = (nr_ss_df_T_al_wet.sum(axis = 0))/1000. #Kj s-1 m-2

nswr_ss_df_T_al_dry = pd.concat([av_nswr_df_T_al[z][0:7000][(av_swe_df_T_al[z][0:7000]>0)] for z in av_swe_df_T_al.columns[1:]],axis = 1) 
nswr_yr_T_al_dry = (nswr_ss_df_T_al_dry.sum(axis = 0))/1000. #Kj s-1 m-2
nswr_ss_df_T_al_wet = pd.concat([av_nswr_df_T_al[z][8760:15760][(av_swe_df_T_al[z][8760:15760]>0)] for z in av_swe_df_T_al.columns[1:]],axis = 1) 
nswr_yr_T_al_wet = (nswr_ss_df_T_al_wet.sum(axis = 0))/1000. #Kj s-1 m-2

nlwr_ss_df_T_al_dry = pd.concat([av_nlwr_df_T_al[z][0:7000][(av_swe_df_T_al[z][0:7000]>0)] for z in av_swe_df_T_al.columns[1:]],axis = 1) 
nlwr_yr_T_al_dry = (nlwr_ss_df_T_al_dry.sum(axis = 0))/1000. #Kj s-1 m-2
nlwr_ss_df_T_al_wet = pd.concat([av_nlwr_df_T_al[z][8760:15760][(av_swe_df_T_al[z][8760:15760]>0)] for z in av_swe_df_T_al.columns[1:]],axis = 1) 
nlwr_yr_T_al_wet = (nlwr_ss_df_T_al_wet.sum(axis = 0))/1000. #Kj s-1 m-2

#bowen ratio
lh_ss_df_T_al_dry = pd.concat([av_lh_df_T_al[z][0:7000][(av_swe_df_T_al[z][0:7000]>0)] for z in av_swe_df_T_al.columns[1:]],axis = 1) 
lh_yr_T_al_dry = (lh_ss_df_T_al_dry.sum(axis = 0))/1000. #Kj s-1 m-2
lh_ss_df_T_al_wet = pd.concat([av_lh_df_T_al[z][8760:15760][(av_swe_df_T_al[z][8760:15760]>0)] for z in av_swe_df_T_al.columns[1:]],axis = 1) 
lh_yr_T_al_wet = (lh_ss_df_T_al_wet.sum(axis = 0))/1000. #Kj s-1 m-2

sh_ss_df_T_al_dry = pd.concat([av_sh_df_T_al[z][0:7000][(av_swe_df_T_al[z][0:7000]>0)] for z in av_swe_df_T_al.columns[1:]],axis = 1) 
sh_yr_T_al_dry = (sh_ss_df_T_al_dry.sum(axis = 0))/1000. #Kj s-1 m-2
sh_ss_df_T_al_wet = pd.concat([av_sh_df_T_al[z][8760:15760][(av_swe_df_T_al[z][8760:15760]>0)] for z in av_swe_df_T_al.columns[1:]],axis = 1) 
sh_yr_T_al_wet = (sh_ss_df_T_al_wet.sum(axis = 0))/1000. #Kj s-1 m-2

BowenRatio_T_al_dry = sh_yr_T_al_dry / lh_yr_T_al_dry
BowenRatio_T_al_wet = sh_yr_T_al_wet / lh_yr_T_al_wet

#%% energy fluxes criteria for meting season

nr_ms_df_T_al_dry = pd.concat([av_nr_df_T_al[z][0:7000][(av_rPm_df_T_al[z][0:7000]>0) & (av_swe_df_T_al[z][0:7000]>0)] for z in av_swe_df_T_al.columns[1:]],axis = 1) 
nerRad_accumms_T_al_dry = (nr_ms_df_T_al_dry.sum(axis = 0))/1000. #Kj s-1 m-2
nr_ms_df_T_al_wet = pd.concat([av_nr_df_T_al[z][8760:15760][(av_rPm_df_T_al[z][8760:15760]>0) & (av_swe_df_T_al[z][8760:15760]>0)] for z in av_swe_df_T_al.columns[1:]],axis = 1) 
nerRad_accumms_T_al_wet = (nr_ms_df_T_al_wet.sum(axis = 0))/1000. #Kj s-1 m-2

nswr_ms_df_T_al_dry = pd.concat([av_nswr_df_T_al[z][0:7000][(av_rPm_df_T_al[z][0:7000]>0) & (av_swe_df_T_al[z][0:7000]>0)] for z in av_swe_df_T_al.columns[1:]],axis = 1) 
nswr_accumms_T_al_dry = (nswr_ms_df_T_al_dry.sum(axis = 0))/1000. #Kj s-1 m-2
nswr_ms_df_T_al_wet = pd.concat([av_nswr_df_T_al[z][8760:15760][(av_rPm_df_T_al[z][8760:15760]>0) & (av_swe_df_T_al[z][8760:15760]>0)] for z in av_swe_df_T_al.columns[1:]],axis = 1) 
nswr_accumms_T_al_wet = (nswr_ms_df_T_al_wet.sum(axis = 0))/1000. #Kj s-1 m-2

nlwr_ms_df_T_al_dry = pd.concat([av_nlwr_df_T_al[z][0:7000][(av_rPm_df_T_al[z][0:7000]>0) & (av_swe_df_T_al[z][0:7000]>0)] for z in av_swe_df_T_al.columns[1:]],axis = 1) 
nlwr_accumms_T_al_dry = (nlwr_ms_df_T_al_dry.sum(axis = 0))/1000. #Kj s-1 m-2
nlwr_ms_df_T_al_wet = pd.concat([av_nlwr_df_T_al[z][8760:15760][(av_rPm_df_T_al[z][8760:15760]>0) & (av_swe_df_T_al[z][8760:15760]>0)] for z in av_swe_df_T_al.columns[1:]],axis = 1) 
nlwr_accumms_T_al_wet = (nlwr_ms_df_T_al_wet.sum(axis = 0))/1000. #Kj s-1 m-2

#radiation_ratio
radiation_ratio_sw_df_T_al_wet = pd.DataFrame(nswr_accumms_T_al_wet/nerRad_accumms_T_al_wet);radiation_ratio_sw_df_T_al_wet.index = hru_names_df_swe[0]
radiation_ratio_lw_df_T_al_wet = pd.DataFrame(nlwr_accumms_T_al_wet/nerRad_accumms_T_al_wet);radiation_ratio_lw_df_T_al_wet.index = hru_names_df_swe[0]
radiation_ratio_sw_df_T_al_dry = pd.DataFrame(nswr_accumms_T_al_dry/nerRad_accumms_T_al_dry);radiation_ratio_sw_df_T_al_dry.index = hru_names_df_swe[0]
radiation_ratio_lw_df_T_al_dry = pd.DataFrame(nlwr_accumms_T_al_dry/nerRad_accumms_T_al_dry);radiation_ratio_lw_df_T_al_dry.index = hru_names_df_swe[0]

radiation_ratio_sw_df_T_al_dry_mr = radiation_ratio_sw_df_T_al_dry[0][hru_names_df_swe_mr[0]]
radiation_ratio_sw_df_T_al_dry_mr_cc = radiation_ratio_sw_df_T_al_dry[0][hru_names_df_swe_mr_sst_cc[0]]
radiation_ratio_sw_df_T_al_wet_mr = radiation_ratio_sw_df_T_al_wet[0][hru_names_df_swe_mr[0]]
radiation_ratio_sw_df_T_al_wet_mr_cc = radiation_ratio_sw_df_T_al_wet[0][hru_names_df_swe_mr_sst_cc[0]]
radiation_ratio_lw_df_T_al_dry_mr = radiation_ratio_lw_df_T_al_dry[0][hru_names_df_swe_mr[0]]
radiation_ratio_lw_df_T_al_dry_mr_cc = radiation_ratio_lw_df_T_al_dry[0][hru_names_df_swe_mr_sst_cc[0]]
radiation_ratio_lw_df_T_al_wet_mr = radiation_ratio_lw_df_T_al_wet[0][hru_names_df_swe_mr[0]]
radiation_ratio_lw_df_T_al_wet_mr_cc = radiation_ratio_lw_df_T_al_wet[0][hru_names_df_swe_mr_sst_cc[0]]


#bowen ratio
lh_ms_df_T_al_dry = pd.concat([av_lh_df_T_al[z][0:7000][(av_rPm_df_T_al[z][0:7000]>0) & (av_swe_df_T_al[z][0:7000]>0)] for z in av_swe_df_T_al.columns[1:]],axis = 1) 
lh_accumms_T_al_dry = (lh_ms_df_T_al_dry.sum(axis = 0))/1000. #Kj s-1 m-2
lh_ms_df_T_al_wet = pd.concat([av_lh_df_T_al[z][8760:15760][(av_rPm_df_T_al[z][8760:15760]>0) & (av_swe_df_T_al[z][8760:15760]>0)] for z in av_swe_df_T_al.columns[1:]],axis = 1) 
lh_accumms_T_al_wet = (lh_ms_df_T_al_wet.sum(axis = 0))/1000. #Kj s-1 m-2

sh_ms_df_T_al_dry = pd.concat([av_sh_df_T_al[z][0:7000][(av_rPm_df_T_al[z][0:7000]>0) & (av_swe_df_T_al[z][0:7000]>0)] for z in av_swe_df_T_al.columns[1:]],axis = 1) 
sh_accumms_T_al_dry = (sh_ms_df_T_al_dry.sum(axis = 0))/1000. #Kj s-1 m-2
sh_ms_df_T_al_wet = pd.concat([av_sh_df_T_al[z][8760:15760][(av_rPm_df_T_al[z][8760:15760]>0) & (av_swe_df_T_al[z][8760:15760]>0)] for z in av_swe_df_T_al.columns[1:]],axis = 1) 
sh_accumms_T_al_wet = (sh_ms_df_T_al_wet.sum(axis = 0))/1000. #Kj s-1 m-2

BowenRatio_ms_T_al_dry = sh_accumms_T_al_dry / lh_accumms_T_al_dry
BowenRatio_ms_T_al_wet = sh_accumms_T_al_wet / lh_accumms_T_al_wet

 #%%###################################################################################################
# applying different objective functions
start_meltSeason_df_T_al_dry = pd.DataFrame(start_meltSeason_T_al_dry); start_meltSeason_df_T_al_dry.index = hru_names_df_swe[0]
start_meltSeason_df_T_al_dry_mr = start_meltSeason_df_T_al_dry[0][hru_names_df_swe_mr[0]]
start_meltSeason_df_T_al_dry_mr_cc = start_meltSeason_df_T_al_dry[0][hru_names_df_swe_mr_sst_cc[0]]
start_meltSeason_df_T_al_wet = pd.DataFrame(start_meltSeason_T_al_wet); start_meltSeason_df_T_al_wet.index = hru_names_df_swe[0]
start_meltSeason_df_T_al_wet_mr = start_meltSeason_df_T_al_wet[0][hru_names_df_swe_mr[0]]
start_meltSeason_df_T_al_wet_mr_cc = start_meltSeason_df_T_al_wet[0][hru_names_df_swe_mr_sst_cc[0]]

sublim_wntrPrecip_df_T_al_dry = pd.DataFrame(sublim_wntrPrecip_T_al_dry); sublim_wntrPrecip_df_T_al_dry.index = hru_names_df_swe[0]
sublim_wntrPrecip_df_T_al_dry_mr = sublim_wntrPrecip_df_T_al_dry[0][hru_names_df_swe_mr[0]]
sublim_wntrPrecip_df_T_al_dry_mr_cc = sublim_wntrPrecip_df_T_al_dry[0][hru_names_df_swe_mr_sst_cc[0]]
sublim_wntrPrecip_df_T_al_wet = pd.DataFrame(sublim_wntrPrecip_T_al_wet); sublim_wntrPrecip_df_T_al_wet.index = hru_names_df_swe[0]
sublim_wntrPrecip_df_T_al_wet_mr = sublim_wntrPrecip_df_T_al_wet[0][hru_names_df_swe_mr[0]]
sublim_wntrPrecip_df_T_al_wet_mr_cc = sublim_wntrPrecip_df_T_al_wet[0][hru_names_df_swe_mr_sst_cc[0]]

BowenRatio_df_T_al_dry = pd.DataFrame(BowenRatio_T_al_dry); BowenRatio_df_T_al_dry.index = hru_names_df_swe[0]
BowenRatio_df_T_al_dry_mr = BowenRatio_df_T_al_dry[0][hru_names_df_swe_mr[0]]
BowenRatio_df_T_al_dry_mr_cc = BowenRatio_df_T_al_dry[0][hru_names_df_swe_mr_sst_cc[0]]
BowenRatio_df_T_al_wet = pd.DataFrame(BowenRatio_T_al_wet); BowenRatio_df_T_al_wet.index = hru_names_df_swe[0]
BowenRatio_df_T_al_wet_mr = BowenRatio_df_T_al_wet[0][hru_names_df_swe_mr[0]]
BowenRatio_df_T_al_wet_mr_cc = BowenRatio_df_T_al_wet[0][hru_names_df_swe_mr_sst_cc[0]]

BowenRatio_ms_df_T_al_dry = pd.DataFrame(BowenRatio_ms_T_al_dry); BowenRatio_ms_df_T_al_dry.index = hru_names_df_swe[0]
BowenRatio_ms_df_T_al_dry_mr = BowenRatio_ms_df_T_al_dry[0][hru_names_df_swe_mr[0]]
BowenRatio_ms_df_T_al_dry_mr_cc = BowenRatio_ms_df_T_al_dry[0][hru_names_df_swe_mr_sst_cc[0]]
BowenRatio_ms_df_T_al_wet = pd.DataFrame(BowenRatio_ms_T_al_wet); BowenRatio_ms_df_T_al_wet.index = hru_names_df_swe[0]
BowenRatio_ms_df_T_al_wet_mr = BowenRatio_ms_df_T_al_wet[0][hru_names_df_swe_mr[0]]
BowenRatio_ms_df_T_al_wet_mr_cc = BowenRatio_ms_df_T_al_wet[0][hru_names_df_swe_mr_sst_cc[0]]
 
#################################################################################################
#%% calculating CC WRF T swe, sd, and 50%input
#################################################################################################
from allNcFiles_wrf import av_ncfiles_T_al_P
av_all_T_al_P = readAllNcfilesAsDataset(av_ncfiles_T_al_P)

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

av_lh_df_T_al_P_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_lh_df_T_al_P_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_lh_df_T_al_P_lsh.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_al_P_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_lh_df_T_al_P_lsp.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_al_P_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_lh_df_T_al_P_ssc.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_al_P_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_lh_df_T_al_P_ssh.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_al_P_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_lh_df_T_al_P_ssp.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_al_P_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_lh_df_T_al_P_ljc.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_al_P_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_lh_df_T_al_P_ljp.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_al_P_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_lh_df_T_al_P_lth.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_al_P_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_lh_df_T_al_P_sjh.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_al_P_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_lh_df_T_al_P_sjp.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_al_P_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarLatHeatGround',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_lh_df_T_al_P_stp.drop(['counter'], axis = 1, inplace = True)
av_lh_df_T_al_P = pd.concat([av_lh_df_T_al_P_lsc,av_lh_df_T_al_P_lsh,av_lh_df_T_al_P_lsp,av_lh_df_T_al_P_ssc,av_lh_df_T_al_P_ssh,av_lh_df_T_al_P_ssp,av_lh_df_T_al_P_ljc,av_lh_df_T_al_P_ljp,av_lh_df_T_al_P_lth,av_lh_df_T_al_P_sjh,av_lh_df_T_al_P_sjp,av_lh_df_T_al_P_stp], axis = 1)

av_gne_df_T_al_P_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_gne_df_T_al_P_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_gne_df_T_al_P_lsh.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_al_P_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_gne_df_T_al_P_lsp.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_al_P_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_gne_df_T_al_P_ssc.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_al_P_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_gne_df_T_al_P_ssh.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_al_P_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_gne_df_T_al_P_ssp.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_al_P_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_gne_df_T_al_P_ljc.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_al_P_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_gne_df_T_al_P_ljp.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_al_P_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_gne_df_T_al_P_lth.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_al_P_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_gne_df_T_al_P_sjh.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_al_P_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_gne_df_T_al_P_sjp.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_al_P_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarGroundNetNrgFlux',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_gne_df_T_al_P_stp.drop(['counter'], axis = 1, inplace = True)
av_gne_df_T_al_P = pd.concat([av_gne_df_T_al_P_lsc,av_gne_df_T_al_P_lsh,av_gne_df_T_al_P_lsp,av_gne_df_T_al_P_ssc,av_gne_df_T_al_P_ssh,av_gne_df_T_al_P_ssp,av_gne_df_T_al_P_ljc,av_gne_df_T_al_P_ljp,av_gne_df_T_al_P_lth,av_gne_df_T_al_P_sjh,av_gne_df_T_al_P_sjp,av_gne_df_T_al_P_stp], axis = 1)

av_nlwr_df_T_al_P_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_nlwr_df_T_al_P_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_nlwr_df_T_al_P_lsh.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_al_P_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_nlwr_df_T_al_P_lsp.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_al_P_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_nlwr_df_T_al_P_ssc.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_al_P_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_nlwr_df_T_al_P_ssh.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_al_P_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_nlwr_df_T_al_P_ssp.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_al_P_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_nlwr_df_T_al_P_ljc.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_al_P_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_nlwr_df_T_al_P_ljp.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_al_P_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_nlwr_df_T_al_P_lth.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_al_P_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_nlwr_df_T_al_P_sjh.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_al_P_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_nlwr_df_T_al_P_sjp.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_al_P_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarLWNetGround',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_nlwr_df_T_al_P_stp.drop(['counter'], axis = 1, inplace = True)
av_nlwr_df_T_al_P = pd.concat([av_nlwr_df_T_al_P_lsc,av_nlwr_df_T_al_P_lsh,av_nlwr_df_T_al_P_lsp,av_nlwr_df_T_al_P_ssc,av_nlwr_df_T_al_P_ssh,av_nlwr_df_T_al_P_ssp,av_nlwr_df_T_al_P_ljc,av_nlwr_df_T_al_P_ljp,av_nlwr_df_T_al_P_lth,av_nlwr_df_T_al_P_sjh,av_nlwr_df_T_al_P_sjp,av_nlwr_df_T_al_P_stp], axis = 1)

av_sh_df_T_al_P_lsc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'],0)
av_sh_df_T_al_P_lsh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'],1)
av_sh_df_T_al_P_lsh.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_al_P_lsp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['lsp'][0]),hruid_swe_dic['lsp'],['lsp'],2)
av_sh_df_T_al_P_lsp.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_al_P_ssc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['ssc'][0]),hruid_swe_dic['ssc'],['ssc'],3)
av_sh_df_T_al_P_ssc.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_al_P_ssh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['ssh'][0]),hruid_swe_dic['ssh'],['ssh'],4)
av_sh_df_T_al_P_ssh.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_al_P_ssp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['ssp'][0]),hruid_swe_dic['ssp'],['ssp'],5)
av_sh_df_T_al_P_ssp.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_al_P_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],6)
av_sh_df_T_al_P_ljc.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_al_P_ljp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['ljp'][0]),hruid_swe_dic['ljp'],['ljp'],7)
av_sh_df_T_al_P_ljp.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_al_P_lth = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['lth'][0]),hruid_swe_dic['lth'],['lth'],8)
av_sh_df_T_al_P_lth.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_al_P_sjh = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['sjh'][0]),hruid_swe_dic['sjh'],['sjh'],9)
av_sh_df_T_al_P_sjh.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_al_P_sjp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['sjp'][0]),hruid_swe_dic['sjp'],['sjp'],10)
av_sh_df_T_al_P_sjp.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_al_P_stp = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSenHeatGround',pd.DataFrame(hru_names_dic_swe['stp'][0]),hruid_swe_dic['stp'],['stp'],11)
av_sh_df_T_al_P_stp.drop(['counter'], axis = 1, inplace = True)
av_sh_df_T_al_P = pd.concat([av_sh_df_T_al_P_lsc,av_sh_df_T_al_P_lsh,av_sh_df_T_al_P_lsp,av_sh_df_T_al_P_ssc,av_sh_df_T_al_P_ssh,av_sh_df_T_al_P_ssp,av_sh_df_T_al_P_ljc,av_sh_df_T_al_P_ljp,av_sh_df_T_al_P_lth,av_sh_df_T_al_P_sjh,av_sh_df_T_al_P_sjp,av_sh_df_T_al_P_stp], axis = 1)

#%% Dec 1st to June  30th 10 percentile of water input as starting of melting season
rPm_season_T_al_P_dry = pd.concat([av_rPm_df_T_al_P[z][1464:6552][(av_rPm_df_T_al_P[z][1464:6552]>0) & (av_swe_df_T_al_P[z][1464:6552]>0) ] for z in av_swe_df_T_al_P.columns[1:]],axis = 1) 
start_melt_T_al_P_dry = rPm_season_T_al_P_dry.quantile(0.1)
start_meltSeason_T_al_P_dry = []
for j in rPm_season_T_al_P_dry.columns[:]:
    time50p = 61+((rPm_season_T_al_P_dry[j]-start_melt_T_al_P_dry[j]).abs().sort_values().index[0])/24
    start_meltSeason_T_al_P_dry.append(time50p)

rPm_season_wet = pd.concat([av_rPm_df_T_al_P[z][10223:15336][(av_rPm_df_T_al_P[z][10223:15336]>0) & (av_swe_df_T_al_P[z][10223:15336]>0) ] for z in av_swe_df_T_al_P.columns[1:]],axis = 1) 
start_melt_wet = rPm_season_wet.quantile(0.1)
start_meltSeason_T_al_P_wet = []
for j in rPm_season_wet.columns[:]:
    time50p = 61+(((rPm_season_wet[j]-start_melt_wet[j]).abs().sort_values().index[0])-8760)/24
    start_meltSeason_T_al_P_wet.append(time50p)

sms_timeStep_T_al_P_dry = pd.DataFrame(start_meltSeason_T_al_P_dry)*24.; sms_timeStep_T_al_P_dry.index = av_swe_df_T_al_P.columns[1:]
sms_timeStep_T_al_P_wet = pd.DataFrame(start_meltSeason_T_al_P_wet)*24 + 8760.; sms_timeStep_T_al_P_dry.index = av_swe_df_T_al_P.columns[1:]

#%% Dec 1st to June 30th, sublimation
lh_T_al_P_dry = pd.concat([av_lh_df_T_al_P[z][0:7000][(av_lh_df_T_al_P[z][0:7000]<0) & (av_swe_df_T_al_P[z][0:7000]>0) ] for z in av_lh_df_T_al_P.columns[1:]],axis = 1) 
sublimation_T_al_P_dry = lh_T_al_P_dry/-25000. #mm/hr
sublimation_yr_T_al_P_dry = sublimation_T_al_P_dry.sum(axis = 0)
total_precip_T_al_P_dry = 868.2649452
sublim_wntrPrecip_T_al_P_dry = sublimation_yr_T_al_P_dry/total_precip_T_al_P_dry*100

lh_T_al_P_wet = pd.concat([av_lh_df_T_al_P[z][8760:][(av_lh_df_T_al_P[z][8760:]<0) & (av_swe_df_T_al_P[z][8760:]>0) ] for z in av_lh_df_T_al_P.columns[1:]],axis = 1) 
sublimation_T_al_P_wet = lh_T_al_P_wet/-25000. #mm/hr
sublimation_yr_T_al_P_wet = sublimation_T_al_P_wet.sum(axis = 0)
total_precip_T_al_P_wet = 1207.938989
sublim_wntrPrecip_T_al_P_wet = sublimation_yr_T_al_P_dry/total_precip_T_al_P_dry*100

#%% energy fluxes criteria for snow season
av_nswr_df_T_al_P = av_gne_df_T_al_P - (av_nlwr_df_T_al_P + av_lh_df_T_al_P + av_sh_df_T_al_P)
av_nr_df_T_al_P = av_nswr_df_T_al_P + av_nlwr_df_T_al_P

nr_ss_df_T_al_P_dry = pd.concat([av_nr_df_T_al_P[z][0:7000][(av_swe_df_T_al_P[z][0:7000]>0)] for z in av_swe_df_T_al_P.columns[1:]],axis = 1) 
nerRad_yr_T_al_P_dry = (nr_ss_df_T_al_P_dry.sum(axis = 0))/1000. #Kj s-1 m-2
nr_ss_df_T_al_P_wet = pd.concat([av_nr_df_T_al_P[z][8760:15760][(av_swe_df_T_al_P[z][8760:15760]>0)] for z in av_swe_df_T_al_P.columns[1:]],axis = 1) 
nerRad_yr_T_al_P_wet = (nr_ss_df_T_al_P_wet.sum(axis = 0))/1000. #Kj s-1 m-2

nswr_ss_df_T_al_P_dry = pd.concat([av_nswr_df_T_al_P[z][0:7000][(av_swe_df_T_al_P[z][0:7000]>0)] for z in av_swe_df_T_al_P.columns[1:]],axis = 1) 
nswr_yr_T_al_P_dry = (nswr_ss_df_T_al_P_dry.sum(axis = 0))/1000. #Kj s-1 m-2
nswr_ss_df_T_al_P_wet = pd.concat([av_nswr_df_T_al_P[z][8760:15760][(av_swe_df_T_al_P[z][8760:15760]>0)] for z in av_swe_df_T_al_P.columns[1:]],axis = 1) 
nswr_yr_T_al_P_wet = (nswr_ss_df_T_al_P_wet.sum(axis = 0))/1000. #Kj s-1 m-2

nlwr_ss_df_T_al_P_dry = pd.concat([av_nlwr_df_T_al_P[z][0:7000][(av_swe_df_T_al_P[z][0:7000]>0)] for z in av_swe_df_T_al_P.columns[1:]],axis = 1) 
nlwr_yr_T_al_P_dry = (nlwr_ss_df_T_al_P_dry.sum(axis = 0))/1000. #Kj s-1 m-2
nlwr_ss_df_T_al_P_wet = pd.concat([av_nlwr_df_T_al_P[z][8760:15760][(av_swe_df_T_al_P[z][8760:15760]>0)] for z in av_swe_df_T_al_P.columns[1:]],axis = 1) 
nlwr_yr_T_al_P_wet = (nlwr_ss_df_T_al_P_wet.sum(axis = 0))/1000. #Kj s-1 m-2

#bowen ratio
lh_ss_df_T_al_P_dry = pd.concat([av_lh_df_T_al_P[z][0:7000][(av_swe_df_T_al_P[z][0:7000]>0)] for z in av_swe_df_T_al_P.columns[1:]],axis = 1) 
lh_yr_T_al_P_dry = (lh_ss_df_T_al_P_dry.sum(axis = 0))/1000. #Kj s-1 m-2
lh_ss_df_T_al_P_wet = pd.concat([av_lh_df_T_al_P[z][8760:15760][(av_swe_df_T_al_P[z][8760:15760]>0)] for z in av_swe_df_T_al_P.columns[1:]],axis = 1) 
lh_yr_T_al_P_wet = (lh_ss_df_T_al_P_wet.sum(axis = 0))/1000. #Kj s-1 m-2

sh_ss_df_T_al_P_dry = pd.concat([av_sh_df_T_al_P[z][0:7000][(av_swe_df_T_al_P[z][0:7000]>0)] for z in av_swe_df_T_al_P.columns[1:]],axis = 1) 
sh_yr_T_al_P_dry = (sh_ss_df_T_al_P_dry.sum(axis = 0))/1000. #Kj s-1 m-2
sh_ss_df_T_al_P_wet = pd.concat([av_sh_df_T_al_P[z][8760:15760][(av_swe_df_T_al_P[z][8760:15760]>0)] for z in av_swe_df_T_al_P.columns[1:]],axis = 1) 
sh_yr_T_al_P_wet = (sh_ss_df_T_al_P_wet.sum(axis = 0))/1000. #Kj s-1 m-2

BowenRatio_T_al_P_dry = sh_yr_T_al_P_dry / lh_yr_T_al_P_dry
BowenRatio_T_al_P_wet = sh_yr_T_al_P_wet / lh_yr_T_al_P_wet

#%% energy fluxes criteria for meting season

nr_ms_df_T_al_P_dry = pd.concat([av_nr_df_T_al_P[z][0:7000][(av_rPm_df_T_al_P[z][0:7000]>0) & (av_swe_df_T_al_P[z][0:7000]>0)] for z in av_swe_df_T_al_P.columns[1:]],axis = 1) 
nerRad_accumms_T_al_P_dry = (nr_ms_df_T_al_P_dry.sum(axis = 0))/1000. #Kj s-1 m-2
nr_ms_df_T_al_P_wet = pd.concat([av_nr_df_T_al_P[z][8760:15760][(av_rPm_df_T_al_P[z][8760:15760]>0) & (av_swe_df_T_al_P[z][8760:15760]>0)] for z in av_swe_df_T_al_P.columns[1:]],axis = 1) 
nerRad_accumms_T_al_P_wet = (nr_ms_df_T_al_P_wet.sum(axis = 0))/1000. #Kj s-1 m-2

nswr_ms_df_T_al_P_dry = pd.concat([av_nswr_df_T_al_P[z][0:7000][(av_rPm_df_T_al_P[z][0:7000]>0) & (av_swe_df_T_al_P[z][0:7000]>0)] for z in av_swe_df_T_al_P.columns[1:]],axis = 1) 
nswr_accumms_T_al_P_dry = (nswr_ms_df_T_al_P_dry.sum(axis = 0))/1000. #Kj s-1 m-2
nswr_ms_df_T_al_P_wet = pd.concat([av_nswr_df_T_al_P[z][8760:15760][(av_rPm_df_T_al_P[z][8760:15760]>0) & (av_swe_df_T_al_P[z][8760:15760]>0)] for z in av_swe_df_T_al_P.columns[1:]],axis = 1) 
nswr_accumms_T_al_P_wet = (nswr_ms_df_T_al_P_wet.sum(axis = 0))/1000. #Kj s-1 m-2

nlwr_ms_df_T_al_P_dry = pd.concat([av_nlwr_df_T_al_P[z][0:7000][(av_rPm_df_T_al_P[z][0:7000]>0) & (av_swe_df_T_al_P[z][0:7000]>0)] for z in av_swe_df_T_al_P.columns[1:]],axis = 1) 
nlwr_accumms_T_al_P_dry = (nlwr_ms_df_T_al_P_dry.sum(axis = 0))/1000. #Kj s-1 m-2
nlwr_ms_df_T_al_P_wet = pd.concat([av_nlwr_df_T_al_P[z][8760:15760][(av_rPm_df_T_al_P[z][8760:15760]>0) & (av_swe_df_T_al_P[z][8760:15760]>0)] for z in av_swe_df_T_al_P.columns[1:]],axis = 1) 
nlwr_accumms_T_al_P_wet = (nlwr_ms_df_T_al_P_wet.sum(axis = 0))/1000. #Kj s-1 m-2

#radiation_ratio
radiation_ratio_sw_df_T_al_P_wet = pd.DataFrame(nswr_accumms_T_al_P_wet/nerRad_accumms_T_al_P_wet);radiation_ratio_sw_df_T_al_P_wet.index = hru_names_df_swe[0]
radiation_ratio_lw_df_T_al_P_wet = pd.DataFrame(nlwr_accumms_T_al_P_wet/nerRad_accumms_T_al_P_wet);radiation_ratio_lw_df_T_al_P_wet.index = hru_names_df_swe[0]
radiation_ratio_sw_df_T_al_P_dry = pd.DataFrame(nswr_accumms_T_al_P_dry/nerRad_accumms_T_al_P_dry);radiation_ratio_sw_df_T_al_P_dry.index = hru_names_df_swe[0]
radiation_ratio_lw_df_T_al_P_dry = pd.DataFrame(nlwr_accumms_T_al_P_dry/nerRad_accumms_T_al_P_dry);radiation_ratio_lw_df_T_al_P_dry.index = hru_names_df_swe[0]

radiation_ratio_sw_df_T_al_P_dry_mr = radiation_ratio_sw_df_T_al_P_dry[0][hru_names_df_swe_mr[0]]
radiation_ratio_sw_df_T_al_P_dry_mr_cc = radiation_ratio_sw_df_T_al_P_dry[0][hru_names_df_swe_mr_sst_cc[0]]
radiation_ratio_sw_df_T_al_P_wet_mr = radiation_ratio_sw_df_T_al_P_wet[0][hru_names_df_swe_mr[0]]
radiation_ratio_sw_df_T_al_P_wet_mr_cc = radiation_ratio_sw_df_T_al_P_wet[0][hru_names_df_swe_mr_sst_cc[0]]
radiation_ratio_lw_df_T_al_P_dry_mr = radiation_ratio_lw_df_T_al_P_dry[0][hru_names_df_swe_mr[0]]
radiation_ratio_lw_df_T_al_P_dry_mr_cc = radiation_ratio_lw_df_T_al_P_dry[0][hru_names_df_swe_mr_sst_cc[0]]
radiation_ratio_lw_df_T_al_P_wet_mr = radiation_ratio_lw_df_T_al_P_wet[0][hru_names_df_swe_mr[0]]
radiation_ratio_lw_df_T_al_P_wet_mr_cc = radiation_ratio_lw_df_T_al_P_wet[0][hru_names_df_swe_mr_sst_cc[0]]



#bowen ratio
lh_ms_df_T_al_P_dry = pd.concat([av_lh_df_T_al_P[z][0:7000][(av_rPm_df_T_al_P[z][0:7000]>0) & (av_swe_df_T_al_P[z][0:7000]>0)] for z in av_swe_df_T_al_P.columns[1:]],axis = 1) 
lh_accumms_T_al_P_dry = (lh_ms_df_T_al_P_dry.sum(axis = 0))/1000. #Kj s-1 m-2
lh_ms_df_T_al_P_wet = pd.concat([av_lh_df_T_al_P[z][8760:15760][(av_rPm_df_T_al_P[z][8760:15760]>0) & (av_swe_df_T_al_P[z][8760:15760]>0)] for z in av_swe_df_T_al_P.columns[1:]],axis = 1) 
lh_accumms_T_al_P_wet = (lh_ms_df_T_al_P_wet.sum(axis = 0))/1000. #Kj s-1 m-2

sh_ms_df_T_al_P_dry = pd.concat([av_sh_df_T_al_P[z][0:7000][(av_rPm_df_T_al_P[z][0:7000]>0) & (av_swe_df_T_al_P[z][0:7000]>0)] for z in av_swe_df_T_al_P.columns[1:]],axis = 1) 
sh_accumms_T_al_P_dry = (sh_ms_df_T_al_P_dry.sum(axis = 0))/1000. #Kj s-1 m-2
sh_ms_df_T_al_P_wet = pd.concat([av_sh_df_T_al_P[z][8760:15760][(av_rPm_df_T_al_P[z][8760:157600]>0) & (av_swe_df_T_al_P[z][8760:15760]>0)] for z in av_swe_df_T_al_P.columns[1:]],axis = 1) 
sh_accumms_T_al_P_wet = (sh_ms_df_T_al_P_wet.sum(axis = 0))/1000. #Kj s-1 m-2

BowenRatio_ms_T_al_P_dry = sh_accumms_T_al_P_dry / lh_accumms_T_al_P_dry
BowenRatio_ms_T_al_P_wet = sh_accumms_T_al_P_wet / lh_accumms_T_al_P_wet

#%%###################################################################################################
# applying different objective functions
start_meltSeason_df_T_al_P_dry = pd.DataFrame(start_meltSeason_T_al_P_dry); start_meltSeason_df_T_al_P_dry.index = hru_names_df_swe[0]
start_meltSeason_df_T_al_P_dry_mr = start_meltSeason_df_T_al_P_dry[0][hru_names_df_swe_mr[0]]
start_meltSeason_df_T_al_P_dry_mr_cc = start_meltSeason_df_T_al_P_dry[0][hru_names_df_swe_mr_sst_cc[0]]
start_meltSeason_df_T_al_P_wet = pd.DataFrame(start_meltSeason_T_al_P_wet); start_meltSeason_df_T_al_P_wet.index = hru_names_df_swe[0]
start_meltSeason_df_T_al_P_wet_mr = start_meltSeason_df_T_al_P_wet[0][hru_names_df_swe_mr[0]]
start_meltSeason_df_T_al_P_wet_mr_cc = start_meltSeason_df_T_al_P_wet[0][hru_names_df_swe_mr_sst_cc[0]]

sublim_wntrPrecip_df_T_al_P_dry = pd.DataFrame(sublim_wntrPrecip_T_al_P_dry); sublim_wntrPrecip_df_T_al_P_dry.index = hru_names_df_swe[0]
sublim_wntrPrecip_df_T_al_P_dry_mr = sublim_wntrPrecip_df_T_al_P_dry[0][hru_names_df_swe_mr[0]]
sublim_wntrPrecip_df_T_al_P_dry_mr_cc = sublim_wntrPrecip_df_T_al_P_dry[0][hru_names_df_swe_mr_sst_cc[0]]
sublim_wntrPrecip_df_T_al_P_wet = pd.DataFrame(sublim_wntrPrecip_T_al_P_wet); sublim_wntrPrecip_df_T_al_P_wet.index = hru_names_df_swe[0]
sublim_wntrPrecip_df_T_al_P_wet_mr = sublim_wntrPrecip_df_T_al_P_wet[0][hru_names_df_swe_mr[0]]
sublim_wntrPrecip_df_T_al_P_wet_mr_cc = sublim_wntrPrecip_df_T_al_P_wet[0][hru_names_df_swe_mr_sst_cc[0]]

BowenRatio_df_T_al_P_dry = pd.DataFrame(BowenRatio_T_al_P_dry); BowenRatio_df_T_al_P_dry.index = hru_names_df_swe[0]
BowenRatio_df_T_al_P_dry_mr = BowenRatio_df_T_al_P_dry[0][hru_names_df_swe_mr[0]]
BowenRatio_df_T_al_P_dry_mr_cc = BowenRatio_df_T_al_P_dry[0][hru_names_df_swe_mr_sst_cc[0]]
BowenRatio_df_T_al_P_wet = pd.DataFrame(BowenRatio_T_al_P_wet); BowenRatio_df_T_al_P_wet.index = hru_names_df_swe[0]
BowenRatio_df_T_al_P_wet_mr = BowenRatio_df_T_al_P_wet[0][hru_names_df_swe_mr[0]]
BowenRatio_df_T_al_P_wet_mr_cc = BowenRatio_df_T_al_P_wet[0][hru_names_df_swe_mr_sst_cc[0]]

BowenRatio_ms_df_T_al_P_dry = pd.DataFrame(BowenRatio_ms_T_al_P_dry); BowenRatio_ms_df_T_al_P_dry.index = hru_names_df_swe[0]
BowenRatio_ms_df_T_al_P_dry_mr = BowenRatio_ms_df_T_al_P_dry[0][hru_names_df_swe_mr[0]]
BowenRatio_ms_df_T_al_P_dry_mr_cc = BowenRatio_ms_df_T_al_P_dry[0][hru_names_df_swe_mr_sst_cc[0]]
BowenRatio_ms_df_T_al_P_wet = pd.DataFrame(BowenRatio_ms_T_al_P_wet); BowenRatio_ms_df_T_al_P_wet.index = hru_names_df_swe[0]
BowenRatio_ms_df_T_al_P_wet_mr = BowenRatio_ms_df_T_al_P_wet[0][hru_names_df_swe_mr[0]]
BowenRatio_ms_df_T_al_P_wet_mr_cc = BowenRatio_ms_df_T_al_P_wet[0][hru_names_df_swe_mr_sst_cc[0]]
#%%
def plotBoxPlotin2panels(d11_dry,d11_wet,savepath,color,ylabel,text_position,text_color,bottom1,top1,roundn,ysize): #
    
    x11 = ['historical','Future_Temp','Future_Temp&Energy','Future_Temp&Energy&precip',
           'historical','Future_Temp','Future_Temp&Energy','Future_Temp&Energy&precip',
           'historical','Future_Temp','Future_Temp&Energy','Future_Temp&Energy&precip',
           ]

    safig, saax = plt.subplots(2,1, figsize=(60,50))#
    saax[0].text(0.9, text_position[0], 'OF: max SWE', fontsize=55, verticalalignment='top')
    saax[0].text(6.8, text_position[0], 'OF: max SWE & melt rate', fontsize=55, verticalalignment='top')
    saax[0].text(11.6, text_position[0], 'OF: max SWE&melt rate$cold content', fontsize=55, verticalalignment='top')
    saax[0].text(7.7, text_position[1], 'Dry Year', color = text_color, fontsize=80, verticalalignment='top')
    saax[1].text(7.7, text_position[2], 'Wet Year', color = text_color, fontsize=80, verticalalignment='top')
    
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
    saax[0].set_ylabel(ylabel, fontsize=ysize)
    saax[0].set_yticklabels(y_tick0,fontsize=60)
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
    saax[1].set_ylabel(ylabel, fontsize=ysize)
    saax[1].set_yticklabels(y_tick1,fontsize=60)
    saax[1].set_xticklabels(x11, fontsize=50, rotation = '25')#position,
    
    plt.savefig(savepath)
#%% plot swr radiation_ratio

d115_dry = [radiation_ratio_sw_df_h_dry[0],radiation_ratio_sw_df_T_dry[0], 
           radiation_ratio_sw_df_T_al_dry[0],radiation_ratio_sw_df_T_al_P_dry[0],
       
           radiation_ratio_sw_df_h_dry_mr,radiation_ratio_sw_df_T_dry_mr, 
           radiation_ratio_sw_df_T_al_dry_mr,radiation_ratio_sw_df_T_al_P_dry_mr,
           
           radiation_ratio_sw_df_h_dry_mr_cc,radiation_ratio_sw_df_T_dry_mr_cc, 
           radiation_ratio_sw_df_T_al_dry_mr_cc,radiation_ratio_sw_df_T_al_P_dry_mr_cc,
           ]

d115_wet = [radiation_ratio_sw_df_h_wet[0],radiation_ratio_sw_df_T_wet[0], 
           radiation_ratio_sw_df_T_al_wet[0],radiation_ratio_sw_df_T_al_P_wet[0],
       
           radiation_ratio_sw_df_h_wet_mr,radiation_ratio_sw_df_T_wet_mr, 
           radiation_ratio_sw_df_T_al_wet_mr,radiation_ratio_sw_df_T_al_P_wet_mr,
           
           radiation_ratio_sw_df_h_wet_mr_cc,radiation_ratio_sw_df_T_wet_mr_cc, 
           radiation_ratio_sw_df_T_al_wet_mr_cc,radiation_ratio_sw_df_T_al_P_wet_mr_cc,
           ]

text_position5 = [2.45, 2.55, 2.5]
savepath5 = 'C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/cc_wrf_swe/swr_ratio1.png'
ylabel5 = 'Ratio of SWR to total radiation in melting season'
color5 = ['rosybrown','darksalmon','peachpuff']
plotBoxPlotin2panels(d115_dry,d115_wet,savepath5,color5,ylabel5,text_position5,'green',1.3,2.4,1,55)

#%% plot lwr radiation_ratio

d116_dry = [radiation_ratio_lw_df_h_dry[0],radiation_ratio_lw_df_T_dry[0], 
           radiation_ratio_lw_df_T_al_dry[0],radiation_ratio_lw_df_T_al_P_dry[0],
       
           radiation_ratio_lw_df_h_dry_mr,radiation_ratio_lw_df_T_dry_mr, 
           radiation_ratio_lw_df_T_al_dry_mr,radiation_ratio_lw_df_T_al_P_dry_mr,
           
           radiation_ratio_lw_df_h_dry_mr_cc,radiation_ratio_lw_df_T_dry_mr_cc, 
           radiation_ratio_lw_df_T_al_dry_mr_cc,radiation_ratio_lw_df_T_al_P_dry_mr_cc,
           ]

d116_wet = [radiation_ratio_lw_df_h_wet[0],radiation_ratio_lw_df_T_wet[0], 
           radiation_ratio_lw_df_T_al_wet[0],radiation_ratio_lw_df_T_al_P_wet[0],
       
           radiation_ratio_lw_df_h_wet_mr,radiation_ratio_lw_df_T_wet_mr, 
           radiation_ratio_lw_df_T_al_wet_mr,radiation_ratio_lw_df_T_al_P_wet_mr,
           
           radiation_ratio_lw_df_h_wet_mr_cc,radiation_ratio_lw_df_T_wet_mr_cc, 
           radiation_ratio_lw_df_T_al_wet_mr_cc,radiation_ratio_lw_df_T_al_P_wet_mr_cc,
           ]

text_position6 = [-0.25, -0.17, -0.23]
savepath6 = 'C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/cc_wrf_swe/lwr_ratio1.png'
ylabel6 = 'Ratio of LWR to total radiation in melting season'
color6 = ['teal','turquoise','azure']
plotBoxPlotin2panels(d116_dry,d116_wet,savepath6,color6,ylabel6,text_position6,'firebrick',-1.3,-0.3,1,55)

#%%
d110_dry = [start_meltSeason_df_h_dry[0],start_meltSeason_df_T_dry[0], 
           start_meltSeason_df_T_al_dry[0],start_meltSeason_df_T_al_P_dry[0],
       
           start_meltSeason_df_h_dry_mr,start_meltSeason_df_T_dry_mr, 
           start_meltSeason_df_T_al_dry_mr,start_meltSeason_df_T_al_P_dry_mr,
           
           start_meltSeason_df_h_dry_mr_cc,start_meltSeason_df_T_dry_mr_cc, 
           start_meltSeason_df_T_al_dry_mr_cc,start_meltSeason_df_T_al_P_dry_mr_cc,
           ]

d110_wet = [start_meltSeason_df_h_wet[0],start_meltSeason_df_T_wet[0], 
           start_meltSeason_df_T_al_wet[0],start_meltSeason_df_T_al_P_wet[0],
       
           start_meltSeason_df_h_wet_mr,start_meltSeason_df_T_wet_mr, 
           start_meltSeason_df_T_al_wet_mr,start_meltSeason_df_T_al_P_wet_mr,
           
           start_meltSeason_df_h_wet_mr_cc,start_meltSeason_df_T_wet_mr_cc, 
           start_meltSeason_df_T_al_wet_mr_cc,start_meltSeason_df_T_al_P_wet_mr_cc,
           ]

text_position1 = [352, 371, 360]
savepath1 = 'C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/cc_wrf_swe/start_meltSeason4.png'
ylabel1 = 'Start of melting season (day of water year)'
color1 = ['navy','deepskyblue','lightcyan']
plotBoxPlotin2panels(d110_dry,d110_wet,savepath1,color1,ylabel1,text_position1,'red',110,340,0,56)

#%%

d111_dry = [sublim_wntrPrecip_df_h_dry[0],sublim_wntrPrecip_df_T_dry[0], 
            sublim_wntrPrecip_df_T_al_dry[0],sublim_wntrPrecip_df_T_al_P_dry[0],
       
            sublim_wntrPrecip_df_h_dry_mr,sublim_wntrPrecip_df_T_dry_mr, 
            sublim_wntrPrecip_df_T_al_dry_mr,sublim_wntrPrecip_df_T_al_P_dry_mr,
           
            sublim_wntrPrecip_df_h_dry_mr_cc,sublim_wntrPrecip_df_T_dry_mr_cc, 
            sublim_wntrPrecip_df_T_al_dry_mr_cc,sublim_wntrPrecip_df_T_al_P_dry_mr_cc,
           ]

d111_wet = [sublim_wntrPrecip_df_h_wet[0],sublim_wntrPrecip_df_T_wet[0], 
            sublim_wntrPrecip_df_T_al_wet[0],sublim_wntrPrecip_df_T_al_P_wet[0],
       
            sublim_wntrPrecip_df_h_wet_mr,sublim_wntrPrecip_df_T_wet_mr, 
            sublim_wntrPrecip_df_T_al_wet_mr,sublim_wntrPrecip_df_T_al_P_wet_mr,
           
            sublim_wntrPrecip_df_h_wet_mr_cc,sublim_wntrPrecip_df_T_wet_mr_cc, 
            sublim_wntrPrecip_df_T_al_wet_mr_cc,sublim_wntrPrecip_df_T_al_P_wet_mr_cc,
           ]

text_position2 = [0.53, 0.57, 0.53]
savepath2 = 'C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/cc_wrf_swe/sublimationRatio4.png'
ylabel2 = 'Ratio of annual sublimation to winter precipitation'
color2 = ['crimson','lightcoral','mistyrose']
plotBoxPlotin2panels(d111_dry,d111_wet,savepath2,color2,ylabel2,text_position2,'green',0.03,0.5,2,50)
#%%

d112_dry = [BowenRatio_ms_df_h_dry[0],BowenRatio_ms_df_T_dry[0], 
            BowenRatio_ms_df_T_al_dry[0],BowenRatio_ms_df_T_al_P_dry[0],
       
            BowenRatio_ms_df_h_dry_mr,BowenRatio_ms_df_T_dry_mr, 
            BowenRatio_ms_df_T_al_dry_mr,BowenRatio_ms_df_T_al_P_dry_mr,
           
            BowenRatio_ms_df_h_dry_mr_cc,BowenRatio_ms_df_T_dry_mr_cc, 
            BowenRatio_ms_df_T_al_dry_mr_cc,BowenRatio_ms_df_T_al_P_dry_mr_cc,
           ]

d112_wet = [BowenRatio_ms_df_h_wet[0],BowenRatio_ms_df_T_wet[0], 
            BowenRatio_ms_df_T_al_wet[0],BowenRatio_ms_df_T_al_P_wet[0],
       
            BowenRatio_ms_df_h_wet_mr,BowenRatio_ms_df_T_wet_mr, 
            BowenRatio_ms_df_T_al_wet_mr,BowenRatio_ms_df_T_al_P_wet_mr,
           
            BowenRatio_ms_df_h_wet_mr_cc,BowenRatio_ms_df_T_wet_mr_cc, 
            BowenRatio_ms_df_T_al_wet_mr_cc,BowenRatio_ms_df_T_al_P_wet_mr_cc,
           ]
savepath3 = 'C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/cc_wrf_swe/bowenRatioMs4.png'
ylabel3 = 'Bowen ratio in melt season'
color3 = ['purple','plum','lavenderblush']
text_position3 = [2.4, 3, 2.6]
plotBoxPlotin2panels(d112_dry,d112_wet,savepath3,color3,ylabel3,text_position3,'green',-5.5,2,1,60)

#%%

d113_dry = [BowenRatio_df_h_dry[0],BowenRatio_df_T_dry[0], 
            BowenRatio_df_T_al_dry[0],BowenRatio_df_T_al_P_dry[0],
       
            BowenRatio_df_h_dry_mr,BowenRatio_df_T_dry_mr, 
            BowenRatio_df_T_al_dry_mr,BowenRatio_df_T_al_P_dry_mr,
           
            BowenRatio_df_h_dry_mr_cc,BowenRatio_df_T_dry_mr_cc, 
            BowenRatio_df_T_al_dry_mr_cc,BowenRatio_df_T_al_P_dry_mr_cc,
           ]

d113_wet = [BowenRatio_df_h_wet[0],BowenRatio_df_T_wet[0], 
            BowenRatio_df_T_al_wet[0],BowenRatio_df_T_al_P_wet[0],
       
            BowenRatio_df_h_wet_mr,BowenRatio_df_T_wet_mr, 
            BowenRatio_df_T_al_wet_mr,BowenRatio_df_T_al_P_wet_mr,
           
            BowenRatio_df_h_wet_mr_cc,BowenRatio_df_T_wet_mr_cc, 
            BowenRatio_df_T_al_wet_mr_cc,BowenRatio_df_T_al_P_wet_mr_cc,
           ]
savepath4 = 'C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/cc_wrf_swe/bowenRatio4.png'
ylabel4 = 'Bowen ratio in snow season'
color4 = ['deeppink','hotpink','lightpink']
text_position4 = [1.2, 1.4, 1.3]
plotBoxPlotin2panels(d113_dry,d113_wet,savepath4,color4,ylabel4,text_position4,'green',-1.1,1.1,2,60)



#%% plot timing of snow season melt

#d11_dry = [start_meltSeason_df_h_dry[0],start_meltSeason_df_T_dry[0], 
#           start_meltSeason_df_T_al_dry[0],start_meltSeason_df_T_al_P_dry[0],
#       
#           start_meltSeason_df_h_dry_mr,start_meltSeason_df_T_dry_mr, 
#           start_meltSeason_df_T_al_dry_mr,start_meltSeason_df_T_al_P_dry_mr,
#           
#           start_meltSeason_df_h_dry_mr_cc,start_meltSeason_df_T_dry_mr_cc, 
#           start_meltSeason_df_T_al_dry_mr_cc,start_meltSeason_df_T_al_P_dry_mr_cc,
#           ]
#
#d11_wet = [start_meltSeason_df_h_wet[0],start_meltSeason_df_T_wet[0], 
#           start_meltSeason_df_T_al_wet[0],start_meltSeason_df_T_al_P_wet[0],
#       
#           start_meltSeason_df_h_wet_mr,start_meltSeason_df_T_wet_mr, 
#           start_meltSeason_df_T_al_wet_mr,start_meltSeason_df_T_al_P_wet_mr,
#           
#           start_meltSeason_df_h_wet_mr_cc,start_meltSeason_df_T_wet_mr_cc, 
#           start_meltSeason_df_T_al_wet_mr_cc,start_meltSeason_df_T_al_P_wet_mr_cc,
#           ]
#
#safig, saax = plt.subplots(2,1, figsize=(60,40))#
#saax[0].text(1.7, 342, 'OF: max SWE', fontsize=55, verticalalignment='top')
#saax[0].text(6.9, 342, 'OF: max SWE & melt rate', fontsize=55, verticalalignment='top')
#saax[0].text(11.6, 342, 'OF: max SWE&melt rate$cold content', fontsize=55, verticalalignment='top')
#saax[0].text(7.7, 367, 'Dry Year', color = 'red', fontsize=75, verticalalignment='top')
#saax[1].text(7.7, 362.5, 'Wet Year', color = 'red', fontsize=75, verticalalignment='top')
#
#position = [1,2,3,4, 7,8,9,10, 13,14,15,16]#, 32,33,34,35,36,37,38,39
#bp0 = saax[0].boxplot(d11_dry, patch_artist=True, positions = position, capprops = {'linewidth':5.0},
#                   flierprops = dict(marker='o', markersize=16, linestyle='none', markeredgecolor='k'),
#                   whiskerprops = {'linewidth':5.0})
#bp0['boxes'][0].set(linewidth=8, facecolor = 'navy', hatch = '\\')
#bp0['boxes'][1].set(linewidth=8, facecolor = 'navy', hatch = '\\')
#bp0['boxes'][2].set(linewidth=8, facecolor = 'navy', hatch = '\\')
#bp0['boxes'][3].set(linewidth=8, facecolor = 'navy', hatch = '\\')
#bp0['boxes'][4].set(linewidth=8, facecolor = 'deepskyblue', hatch = '*')
#bp0['boxes'][5].set(linewidth=8, facecolor = 'deepskyblue', hatch = '*')
#bp0['boxes'][6].set(linewidth=8, facecolor = 'deepskyblue', hatch = '*')
#bp0['boxes'][7].set(linewidth=8, facecolor = 'deepskyblue', hatch = '*')
#bp0['boxes'][8].set(linewidth=8, facecolor = 'lightcyan', hatch = '/')
#bp0['boxes'][9].set(linewidth=8, facecolor = 'lightcyan', hatch = '/')
#bp0['boxes'][10].set(linewidth=8, facecolor = 'lightcyan', hatch = '/')
#bp0['boxes'][11].set(linewidth=8, facecolor = 'lightcyan', hatch = '/')
#y_tick0=(saax[0].get_yticks()).astype(int)
#saax[0].set_ylabel('Start of melting season (day of water year)', fontsize=40)
#saax[0].set_yticklabels(y_tick0,fontsize=40)
#saax[0].set_xticklabels([], fontsize=50, rotation = '25')#position,
##saax[0].set_ylim(100,350)
#
#bp1 = saax[1].boxplot(d11_wet, patch_artist=True, positions = position, capprops = {'linewidth':5.0},
#                   flierprops = dict(marker='o', markersize=16, linestyle='none', markeredgecolor='k'),
#                   whiskerprops = {'linewidth':5.0})
#bp1['boxes'][0].set(linewidth=8, facecolor = 'navy', hatch = '\\')
#bp1['boxes'][1].set(linewidth=8, facecolor = 'navy', hatch = '\\')
#bp1['boxes'][2].set(linewidth=8, facecolor = 'navy', hatch = '\\')
#bp1['boxes'][3].set(linewidth=8, facecolor = 'navy', hatch = '\\')
#bp1['boxes'][4].set(linewidth=8, facecolor = 'deepskyblue', hatch = '*')
#bp1['boxes'][5].set(linewidth=8, facecolor = 'deepskyblue', hatch = '*')
#bp1['boxes'][6].set(linewidth=8, facecolor = 'deepskyblue', hatch = '*')
#bp1['boxes'][7].set(linewidth=8, facecolor = 'deepskyblue', hatch = '*')
#bp1['boxes'][8].set(linewidth=8, facecolor = 'lightcyan', hatch = '/')
#bp1['boxes'][9].set(linewidth=8, facecolor = 'lightcyan', hatch = '/')
#bp1['boxes'][10].set(linewidth=8, facecolor = 'lightcyan', hatch = '/')
#bp1['boxes'][11].set(linewidth=8, facecolor = 'lightcyan', hatch = '/')
#
#y_tick1=(saax[1].get_yticks()).astype(int)
#saax[1].set_ylabel('Start of melting season (day of water year)', fontsize=40)
#saax[1].set_yticklabels(y_tick1,fontsize=40)
#saax[1].set_xticklabels(x11, fontsize=50, rotation = '25')#position,
##saax[1].set_ylim(100,350)
#
##plt.ylabel('start of melting season (day of water year)', fontsize=60)
##plt.yticks(fontsize=50)
##plt.xticks(position,x11, fontsize=50, rotation = '25')
#saax[0].text(2, 0.65, 'OF: maxSWE', fontsize=140, verticalalignment='top')
#
#plt.savefig('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/cc_wrf_swe/start_meltSeason1.png')

#%% plot timing of snow season melt
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
#    saax[0].set_ylabel(ylabel, fontsize=40)
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
#    saax[1].set_ylabel(ylabel, fontsize=40)
#    saax[1].set_yticklabels(y_tick1,fontsize=40)
#    saax[1].set_xticklabels(x11, fontsize=50, rotation = '25')#position,
#    
#    plt.savefig(savepath)
















