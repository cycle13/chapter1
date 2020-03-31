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

def rainSnowSeperationDic4SWEOF(param_dic,av_rf_df_h):
    rain_h_dic = {'lsc': [], 'lsh': [], 'lsp': [], 'ssc': [], 'ssh': [], 'ssp': [], 
                  'ljc': [], 'ljp': [], 'lth': [], 'sjh': [], 'sjp': [], 'stp': []}
    for keeyy in param_dic.keys():
        rain = []
        param_intr = param_dic[keeyy][0][20]
        for temps in param_intr:
            if np.round(temps) == 272. :
                rain.append(av_rf_df_h[0])
            elif np.round(temps) == 273. :
                rain.append(av_rf_df_h[1])
            elif np.round(temps) == 274. :
                rain.append(av_rf_df_h[2])
            elif np.round(temps) == 275. :
                rain.append(av_rf_df_h[3])
        rain_h_dic[keeyy] = rain
    return rain_h_dic

def calculatingSnowFractionfromSnowfallAndRainfall(param_dic,snow_h_swe_dic,rain_h_swe_dic):
    snowFraction_h_swe = np.zeros((len(param_dic.keys()[:]),1))#{keysss : np.array([0]) for keysss in param_dic.keys()[:]} 
     
    for pdm in param_dic.keys():
        snowFractionTimeSeries = np.zeros((len(snow_h_swe_dic[pdm]),1))
        for ts in range (len(snow_h_swe_dic[pdm])):
            snowFractionParam = np.zeros((len(snow_h_swe_dic[pdm][ts]),1))
            for precip in range (len(snow_h_swe_dic[pdm][ts])):
                if (snow_h_swe_dic[pdm][ts][precip]+rain_h_swe_dic[pdm][ts][precip])>0:
                    snwfrac = snow_h_swe_dic[pdm][ts][precip]/(snow_h_swe_dic[pdm][ts][precip]+rain_h_swe_dic[pdm][ts][precip])
                    snowFractionParam[precip] = snwfrac
            snowFractionParam_mean = np.mean(snowFractionParam)
            snowFractionTimeSeries[ts] = snowFractionParam_mean
        snowFraction_h_swe[pdm] = snowFractionTimeSeries
    return snowFraction_h_swe

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

    
#%% reading index of best parameters for each decision model combination for swe and melt rate
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
   
#%% reading index of best parameters for each decision combination for swe, melt rate, SST
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

index_swe_mr_sst_dic = {'lsc': [], 'lsh': [], 'lsp': [], 'ssc': [], 'ssh': [], 'ssp': [],  
                        'ljc': [], 'ljp': [], 'sjh': []}
for indx3 in params_index_swe_mr_sst_df.columns:
    index_array3 = np.array(hruid_swe_mr_sst_dic[indx3])-10000
    index_swe_mr_sst_dic[indx3].append(index_array3)

param_swe_mr_sst_dic = {'lsc': [], 'lsh': [], 'lsp': [], 'ssc': [], 'ssh': [], 'ssp': [],  
                        'ljc': [], 'ljp': [], 'sjh': []}
for prms3 in params_index_swe_mr_sst_df.columns:
    if prms3[1]=='s':
        params0000 = params_sa_df13p.iloc[index_swe_mr_sst_dic[prms3][0]]
        param_swe_mr_sst_dic[prms3].append(params0000)
    else:
        params0000 = params_sa_df12p.iloc[index_swe_mr_sst_dic[prms3][0]]
        param_swe_mr_sst_dic[prms3].append(params0000)
    
#%% reading index of best parameters for each decision combination for swe, melt rate, SST, CC
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

#%%  reading output files for historical for rainfall snowfall seperation
from allNcFiles_wrf_snowRain import av_ncfiles_h_rs
av_all_h_rs = readAllNcfilesAsDataset(av_ncfiles_h_rs)

DateSa21 = date(av_all_h_rs[0],"%Y-%m-%d") #"%Y-%m-%d %H:%M"
DateSa22 = date(av_all_h_rs[1],"%Y-%m-%d")
date_sa = np.append(DateSa21,DateSa22)
sax = np.arange(0,np.size(date_sa))
sa_xticks = date_sa

# calculating historical snowfall rainfall
av_rf_df_h_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h_rs,'scalarRainfall',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],0)
av_sf_df_h_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h_rs,'scalarSnowfall',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],0)
av_rf_df_h_272 = av_rf_df_h_ljc['ljc10760'][:]
av_rf_df_h_273 = av_rf_df_h_ljc['ljc10761'][:]
av_rf_df_h_274 = av_rf_df_h_ljc['ljc10762'][:]
av_rf_df_h_275 = av_rf_df_h_ljc['ljc10763'][:]
av_rf_df_h = [av_rf_df_h_272, av_rf_df_h_273, av_rf_df_h_274, av_rf_df_h_275]
av_sf_df_h = [av_sf_df_h_ljc['ljc10760'][:], av_sf_df_h_ljc['ljc10761'][:], av_sf_df_h_ljc['ljc10762'][:], av_sf_df_h_ljc['ljc10763'][:]]

rain_h_swe_dic = rainSnowSeperationDic4SWEOF(param_dic,av_rf_df_h)
snow_h_swe_dic = rainSnowSeperationDic4SWEOF(param_dic,av_sf_df_h)
rain_h_swe_mr_dic = rainSnowSeperationDic4SWEOF(param_swe_mr_dic,av_rf_df_h)
snow_h_swe_mr_dic = rainSnowSeperationDic4SWEOF(param_swe_mr_dic,av_sf_df_h)
rain_h_swe_mr_sst_dic = rainSnowSeperationDic4SWEOF(param_swe_mr_sst_dic,av_rf_df_h)
snow_h_swe_mr_sst_dic = rainSnowSeperationDic4SWEOF(param_swe_mr_sst_dic,av_sf_df_h)
rain_h_swe_mr_sst_cc_dic = rainSnowSeperationDic4SWEOF(param_swe_mr_sst_cc_dic,av_rf_df_h)
snow_h_swe_mr_sst_cc_dic = rainSnowSeperationDic4SWEOF(param_swe_mr_sst_cc_dic,av_sf_df_h)
    
    
from allNcFiles_wrf_snowRain import av_ncfiles_T_rs
av_all_T_rs = readAllNcfilesAsDataset(av_ncfiles_T_rs)
av_rf_df_T_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_rs,'scalarRainfall',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],0)
av_sf_df_T_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_rs,'scalarSnowfall',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],0)
av_rf_df_T = [av_rf_df_T_ljc['ljc10760'][:],av_rf_df_T_ljc['ljc10761'][:],
              av_rf_df_T_ljc['ljc10762'][:],av_rf_df_T_ljc['ljc10763'][:]]
av_sf_df_T = [av_sf_df_T_ljc['ljc10760'][:],av_sf_df_T_ljc['ljc10761'][:],
              av_sf_df_T_ljc['ljc10762'][:],av_sf_df_T_ljc['ljc10763'][:]]
rain_T_swe_dic = rainSnowSeperationDic4SWEOF(param_dic,av_rf_df_T)
snow_T_swe_dic = rainSnowSeperationDic4SWEOF(param_dic,av_sf_df_T)
rain_T_swe_mr_dic = rainSnowSeperationDic4SWEOF(param_swe_mr_dic,av_rf_df_T)
snow_T_swe_mr_dic = rainSnowSeperationDic4SWEOF(param_swe_mr_dic,av_sf_df_T)
rain_T_swe_mr_sst_dic = rainSnowSeperationDic4SWEOF(param_swe_mr_sst_dic,av_rf_df_T)
snow_T_swe_mr_sst_dic = rainSnowSeperationDic4SWEOF(param_swe_mr_sst_dic,av_sf_df_T)
rain_T_swe_mr_sst_cc_dic = rainSnowSeperationDic4SWEOF(param_swe_mr_sst_cc_dic,av_rf_df_T)
snow_T_swe_mr_sst_cc_dic = rainSnowSeperationDic4SWEOF(param_swe_mr_sst_cc_dic,av_sf_df_T)


from allNcFiles_wrf_snowRain import av_ncfiles_T_al_rs
av_all_T_al_rs = readAllNcfilesAsDataset(av_ncfiles_T_al_rs)
av_rf_df_T_al_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_rs,'scalarRainfall',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],0)
av_sf_df_T_al_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_rs,'scalarSnowfall',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],0)
av_rf_df_T_al = [av_rf_df_T_al_ljc['ljc10760'][:],av_rf_df_T_al_ljc['ljc10761'][:],
                 av_rf_df_T_al_ljc['ljc10762'][:],av_rf_df_T_al_ljc['ljc10763'][:]]
av_sf_df_T_al = [av_sf_df_T_al_ljc['ljc10760'][:],av_sf_df_T_al_ljc['ljc10761'][:],
                 av_sf_df_T_al_ljc['ljc10762'][:],av_sf_df_T_al_ljc['ljc10763'][:]]
rain_T_al_swe_dic = rainSnowSeperationDic4SWEOF(param_dic,av_rf_df_T_al)
snow_T_al_swe_dic = rainSnowSeperationDic4SWEOF(param_dic,av_sf_df_T_al)
rain_T_al_swe_mr_dic = rainSnowSeperationDic4SWEOF(param_swe_mr_dic,av_rf_df_T_al)
snow_T_al_swe_mr_dic = rainSnowSeperationDic4SWEOF(param_swe_mr_dic,av_sf_df_T_al)
rain_T_al_swe_mr_sst_dic = rainSnowSeperationDic4SWEOF(param_swe_mr_sst_dic,av_rf_df_T_al)
snow_T_al_swe_mr_sst_dic = rainSnowSeperationDic4SWEOF(param_swe_mr_sst_dic,av_sf_df_T_al)
rain_T_al_swe_mr_sst_cc_dic = rainSnowSeperationDic4SWEOF(param_swe_mr_sst_cc_dic,av_rf_df_T_al)
snow_T_al_swe_mr_sst_cc_dic = rainSnowSeperationDic4SWEOF(param_swe_mr_sst_cc_dic,av_sf_df_T_al)


from allNcFiles_wrf_snowRain import av_ncfiles_T_al_P_rs
av_all_T_al_P_rs = readAllNcfilesAsDataset(av_ncfiles_T_al_P_rs)
av_rf_df_T_al_P_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P_rs,'scalarRainfall',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],0)
av_sf_df_T_al_P_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P_rs,'scalarSnowfall',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'],0)
av_rf_df_T_al_P = [av_rf_df_T_al_P_ljc['ljc10760'][:],av_rf_df_T_al_P_ljc['ljc10761'][:],
                   av_rf_df_T_al_P_ljc['ljc10762'][:],av_rf_df_T_al_P_ljc['ljc10763'][:]]
av_sf_df_T_al_P = [av_sf_df_T_al_P_ljc['ljc10760'][:],av_sf_df_T_al_P_ljc['ljc10761'][:],
                   av_sf_df_T_al_P_ljc['ljc10762'][:],av_sf_df_T_al_P_ljc['ljc10763'][:]]
rain_T_al_P_swe_dic = rainSnowSeperationDic4SWEOF(param_dic,av_rf_df_T_al_P)
snow_T_al_P_swe_dic = rainSnowSeperationDic4SWEOF(param_dic,av_sf_df_T_al_P)
rain_T_al_P_swe_mr_dic = rainSnowSeperationDic4SWEOF(param_swe_mr_dic,av_rf_df_T_al_P)
snow_T_al_P_swe_mr_dic = rainSnowSeperationDic4SWEOF(param_swe_mr_dic,av_sf_df_T_al_P)
rain_T_al_P_swe_mr_sst_dic = rainSnowSeperationDic4SWEOF(param_swe_mr_sst_dic,av_rf_df_T_al_P)
snow_T_al_P_swe_mr_sst_dic = rainSnowSeperationDic4SWEOF(param_swe_mr_sst_dic,av_sf_df_T_al_P)
rain_T_al_P_swe_mr_sst_cc_dic = rainSnowSeperationDic4SWEOF(param_swe_mr_sst_cc_dic,av_rf_df_T_al_P)
snow_T_al_P_swe_mr_sst_cc_dic = rainSnowSeperationDic4SWEOF(param_swe_mr_sst_cc_dic,av_sf_df_T_al_P)

#%% calculation fraction of snow
snowFraction_h_swe_dry = np.zeros((len(param_dic.keys()[:]),1))#{keysss : np.array([0]) for keysss in param_dic.keys()[:]} 
 
for pdm in range(len(param_dic.keys()[0:2])):
    snowFractionTimeSeries = np.zeros((len(snow_h_swe_dic[param_dic.keys()[pdm]]),1))
    for ts in range (1):#len(snow_h_swe_dicparam_dic.keys()[param_dic.keys()[pdm]])
        snowFractionParam = np.zeros((len(snow_h_swe_dic[param_dic.keys()[pdm]][ts]),1))
        for precip in range (len(snow_h_swe_dic[param_dic.keys()[pdm]][ts][0:8760])):
            if (snow_h_swe_dic[param_dic.keys()[pdm]][ts][precip]+rain_h_swe_dic[param_dic.keys()[pdm]][ts][precip])>0:
                snwfrac = snow_h_swe_dic[param_dic.keys()[pdm]][ts][precip]/(snow_h_swe_dic[param_dic.keys()[pdm]][ts][precip]+rain_h_swe_dic[param_dic.keys()[pdm]][ts][precip])
                snowFractionParam[precip] = snwfrac
        snowFractionParam_mean = np.mean(snowFractionParam)
        snowFractionTimeSeries[ts] = snowFractionParam_mean
    snowFraction_h_swe_dry[param_dic.keys()[pdm]] = snowFractionTimeSeries




snowFraction_h_swe2 = calculatingSnowFractionfromSnowfallAndRainfall(param_dic,snow_h_swe_dic,rain_h_swe_dic)    
snowFraction_h_swe_mr = calculatingSnowFractionfromSnowfallAndRainfall(param_swe_mr_dic,snow_h_swe_mr_dic,rain_h_swe_mr_dic)    
snowFraction_h_swe_mr_sst = calculatingSnowFractionfromSnowfallAndRainfall(param_swe_mr_sst_dic,snow_h_swe_mr_sst_dic,rain_h_swe_mr_sst_dic)    
snowFraction_h_swe_mr_sst_cc = calculatingSnowFractionfromSnowfallAndRainfall(param_swe_mr_sst_cc_dic,snow_h_swe_mr_sst_cc_dic,rain_h_swe_mr_sst_cc_dic)    

snowFraction_T_swe2 = calculatingSnowFractionfromSnowfallAndRainfall(param_dic,snow_T_swe_dic,rain_T_swe_dic)    
snowFraction_T_swe_mr = calculatingSnowFractionfromSnowfallAndRainfall(param_swe_mr_dic,snow_T_swe_mr_dic,rain_T_swe_mr_dic)    
snowFraction_T_swe_mr_sst = calculatingSnowFractionfromSnowfallAndRainfall(param_swe_mr_sst_dic,snow_T_swe_mr_sst_dic,rain_T_swe_mr_sst_dic)    
snowFraction_T_swe_mr_sst_cc = calculatingSnowFractionfromSnowfallAndRainfall(param_swe_mr_sst_cc_dic,snow_T_swe_mr_sst_cc_dic,rain_T_swe_mr_sst_cc_dic)    

snowFraction_T_al_swe2 = calculatingSnowFractionfromSnowfallAndRainfall(param_dic,snow_T_al_swe_dic,rain_T_al_swe_dic)    
snowFraction_T_al_swe_mr = calculatingSnowFractionfromSnowfallAndRainfall(param_swe_mr_dic,snow_T_al_swe_mr_dic,rain_T_al_swe_mr_dic)    
snowFraction_T_al_swe_mr_sst = calculatingSnowFractionfromSnowfallAndRainfall(param_swe_mr_sst_dic,snow_T_al_swe_mr_sst_dic,rain_T_al_swe_mr_sst_dic)    
snowFraction_T_al_swe_mr_sst_cc = calculatingSnowFractionfromSnowfallAndRainfall(param_swe_mr_sst_cc_dic,snow_T_al_swe_mr_sst_cc_dic,rain_T_al_swe_mr_sst_cc_dic)    

snowFraction_T_al_P_swe2 = calculatingSnowFractionfromSnowfallAndRainfall(param_dic,snow_T_al_P_swe_dic,rain_T_al_P_swe_dic)    
snowFraction_T_al_P_swe_mr = calculatingSnowFractionfromSnowfallAndRainfall(param_swe_mr_dic,snow_T_al_P_swe_mr_dic,rain_T_al_P_swe_mr_dic)    
snowFraction_T_al_P_swe_mr_sst = calculatingSnowFractionfromSnowfallAndRainfall(param_swe_mr_sst_dic,snow_T_al_P_swe_mr_sst_dic,rain_T_al_P_swe_mr_sst_dic)    
snowFraction_T_al_P_swe_mr_sst_cc = calculatingSnowFractionfromSnowfallAndRainfall(param_swe_mr_sst_cc_dic,snow_T_al_P_swe_mr_sst_cc_dic,rain_T_al_P_swe_mr_sst_cc_dic)    

#np.save('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/snow_rain_sep/snowFraction_h_swe',snowFraction_h_swe2)
#snowFraction_h_swe = np.array(np.load('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/snow_rain_sep/snowFraction_h_swe.npy'))

#%% ploting 
n_groups = 32
x11 = ['his_wet','T_wet','T_al_wet','T_al_P_wet',
       'his_dry','T_dry','T_al_dry','T_al_P_dry',
       
       'his_wet','T_wet','T_al_wet','T_al_P_wet',
       'his_dry','T_dry','T_al_dry','T_al_P_dry',
       
       'his_wet','T_wet','T_al_wet','T_al_P_wet',
       'his_dry','T_dry','T_al_dry','T_al_P_dry',
       
       'his_wet','T_wet','T_al_wet','T_al_P_wet',
       'his_dry','T_dry','T_al_dry','T_al_P_dry',
       ]

d11 = [maxSWE_df_h_wet[0],maxSWE_df_T_wet[0], 
       maxSWE_df_T_al_wet[0],maxSWE_df_T_al_P_wet[0],
       maxSWE_df_h_dry[0],maxSWE_df_T_dry[0], 
       maxSWE_df_T_al_dry[0],maxSWE_df_T_al_P_dry[0],
       
       maxSWE_h_wet_df_mr,maxSWE_T_wet_df_mr,
       maxSWE_T_al_wet_df_mr,maxSWE_T_al_P_wet_df_mr,
       maxSWE_h_dry_df_mr,maxSWE_T_dry_df_mr,
       maxSWE_T_al_dry_df_mr,maxSWE_T_al_P_dry_df_mr,
       
       maxSWE_h_wet_df_mr_sst,maxSWE_T_wet_df_mr_sst,
       maxSWE_T_al_wet_df_mr_sst,maxSWE_T_al_P_wet_df_mr_sst,
       maxSWE_h_dry_df_mr_sst,maxSWE_T_dry_df_mr_sst,
       maxSWE_T_al_dry_df_mr_sst,maxSWE_T_al_P_dry_df_mr_sst,
       
       maxSWE_h_wet_df_mr_sst_cc,maxSWE_T_wet_df_mr_sst_cc,
       maxSWE_T_al_wet_df_mr_sst_cc,maxSWE_T_al_P_wet_df_mr_sst_cc,
       maxSWE_h_dry_df_mr_sst_cc,maxSWE_T_dry_df_mr_sst_cc,
       maxSWE_T_al_dry_df_mr_sst_cc,maxSWE_T_al_P_dry_df_mr_sst_cc,
       ]

index = np.arange(n_groups)
bar_width = 0.5

safig, saax = plt.subplots(1,1, figsize=(160,40))#
position = [1,2,3,4,5,6,7,8, 11,12,13,14,15,16,17,18, 21,22,23,24,25,26,27,28, 32,33,34,35,36,37,38,39]
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

bp3['boxes'][24].set(linewidth=8, facecolor = 'deepskyblue', hatch = '\\')
bp3['boxes'][25].set(linewidth=8, facecolor = 'deepskyblue', hatch = '\\')
bp3['boxes'][26].set(linewidth=8, facecolor = 'deepskyblue', hatch = '\\')
bp3['boxes'][27].set(linewidth=8, facecolor = 'deepskyblue', hatch = '\\')
bp3['boxes'][28].set(linewidth=8, facecolor = 'lightcyan', hatch = '\\')
bp3['boxes'][29].set(linewidth=8, facecolor = 'lightcyan', hatch = '\\')
bp3['boxes'][30].set(linewidth=8, facecolor = 'lightcyan', hatch = '\\')
bp3['boxes'][31].set(linewidth=8, facecolor = 'lightcyan', hatch = '\\')

plt.ylabel('max SWE (mm)', fontsize=80)
plt.yticks(fontsize=80)
plt.xticks(position,x11, fontsize=60, rotation = '25')
plt.title('hhhh')
plt.savefig('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/cc_wrf_swe/sa_cc_swe_swe_mr_sst_cc_all.png')



















































