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

    return meltingrateAvg_h_dry, meltingrateAvg_h_wet

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

# reading index of best parameters for each decision model combination 
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
                              pd.DataFrame(hru_names_dic_swe['ljc'][0])],ignore_index = True)

#%%  reading output files for historical
from allNcFiles_wrf import av_ncfiles_h
av_all_h = readAllNcfilesAsDataset(av_ncfiles_h)

DateSa21 = date(av_all_h[0],"%Y-%m-%d") #"%Y-%m-%d %H:%M"
DateSa22 = date(av_all_h[1],"%Y-%m-%d")
date_sa = np.append(DateSa21,DateSa22)
sax = np.arange(0,np.size(date_sa))
sa_xticks = date_sa

# calculating historical swe, sd, and 50%input
av_sd_df_h_lsc = readVariablefromMultipleNcfilesDatasetasDF(av_all_h,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'])
av_sd_df_h_lsh = readVariablefromMultipleNcfilesDatasetasDF(av_all_h,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'])
av_sd_df_h_lsh.drop(['counter'], axis = 1)
av_sd_df_h_ljc = readVariablefromMultipleNcfilesDatasetasDF(av_all_h,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'])
av_sd_df_h_ljc.drop(['counter'], axis = 1)
av_sd_df_h = pd.concat([av_sd_df_h_lsc,av_sd_df_h_lsh,av_sd_df_h_ljc], axis = 1)

av_swe_df_h_lsc = readVariablefromMultipleNcfilesDatasetasDF(av_all_h,'scalarSWE',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'])
av_swe_df_h_lsh = readVariablefromMultipleNcfilesDatasetasDF(av_all_h,'scalarSWE',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'])
av_swe_df_h_lsh.drop(['counter'], axis = 1)
av_swe_df_h_ljc = readVariablefromMultipleNcfilesDatasetasDF(av_all_h,'scalarSWE',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'])
av_swe_df_h_ljc.drop(['counter'], axis = 1)
av_swe_df_h = pd.concat([av_swe_df_h_lsc,av_swe_df_h_lsh,av_swe_df_h_ljc], axis = 1)

av_rPm_df_h_lsc = readVariablefromMultipleNcfilesDatasetasDF(av_all_h,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'])
av_rPm_df_h_lsh = readVariablefromMultipleNcfilesDatasetasDF(av_all_h,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['lsh'][0]),hruid_swe_dic['lsh'],['lsh'])
av_rPm_df_h_lsh.drop(['counter'], axis = 1)
av_rPm_df_h_ljc = readVariablefromMultipleNcfilesDatasetasDF(av_all_h,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_swe_dic['ljc'],['ljc'])
av_rPm_df_h_ljc.drop(['counter'], axis = 1)
av_rPm_df_h = pd.concat([av_rPm_df_h_lsc,av_rPm_df_h_lsh,av_rPm_df_h_ljc], axis = 1)

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

time0f50input_h_wet = []
av_rPm_df_h_wet_cum50 = (av_rPm_df_h_wet_cum.iloc[-1])[1:]/2.
for hrun in range(len(hru_names_df_swe)):
    inpt = av_rPm_df_h_wet_cum50[hrun]
    time0f50input = ((av_rPm_df_h_wet_cum[hru_names_df_swe[0][hrun]]-inpt).abs()).argmin()
    time0f50input_h_wet.append(time0f50input-8760)
    
# historical day of snow disappearance (based on snowdepth)-final output
dosd_df_h_dry, dosd_residual_df_h_dry = calculatingSDD(av_sd_df_h[:][5000:8737],hru_names_df_swe,5976,'2007',3737,5000)
dosd_df_h_wet, dosd_residual_df_h_wet = calculatingSDD(av_sd_df_h[:][14000:],hru_names_df_swe,14976,'2008',3521,14000)

# ***********************historical  finding max corespondance swe for 'h_dry and h_wet'***********************
maxSWE_df_h_dry, maxSWE_df_h_wet = averageMaxSWE4dryAndWetYears (av_swe_df_h,hru_names_df_swe)
   
## ***********historical: calculating snowmelt rate based on SWE #cm/day *************************************
meltingrateAvg_h_dry, meltingrateAvg_h_wet = meltRateBased0nSWE (hru_names_df_swe,av_swe_df_h,dosd_df_h_dry,dosd_df_h_wet)

#################################################################################################
#%% calculating CC WRF T swe, sd, and 50%input
#################################################################################################
from allNcFiles_wrf import av_ncfiles_T
from allNcFiles_wrf import av_ncfiles_T_al
av_all_T = readAllNcfilesAsDataset(av_ncfiles_T)
av_all_T_al = readAllNcfilesAsDataset(av_ncfiles_T_al)

av_sd_df_T = readVariablefromMultipleNcfilesDatasetasDF(av_all_T,'scalarSnowDepth',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'])
av_swe_df_T = readVariablefromMultipleNcfilesDatasetasDF(av_all_T,'scalarSWE',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'])
av_rPm_df_T = readVariablefromMultipleNcfilesDatasetasDF(av_all_T,'scalarRainPlusMelt',pd.DataFrame(hru_names_dic_swe['lsc'][0]),hruid_swe_dic['lsc'],['lsc'])

#av_sd_df = pd.concat([av_sd_df12p,av_sd_df13p], axis = 1)
#av_swe_df = pd.concat([av_swe_df12p,av_swe_df13p], axis = 1)
#av_rPm_df = pd.concat([av_rPm_df12p,av_rPm_df13p], axis = 1)
av_rPm_df_T_dry = av_rPm_df_T.iloc[0:8760]
av_rPm_df_T_wet = av_rPm_df_T.iloc[8760:]

av_rPm_df_T_dry_cum = av_rPm_df_T_dry.cumsum(axis=0)#; av_rPm_df_T_dry_cum.index = date_sa [0:8760]
av_rPm_df_T_wet_cum = av_rPm_df_T_wet.cumsum(axis=0)#; av_rPm_df_T_wet_cum.index = date_sa [8760:]

time0f50input_T_dry = []
av_rPm_df_T_dry_cum50 = (av_rPm_df_T_dry_cum.iloc[-1])[1:]/2.
for hrun in range(len(pd.DataFrame(hru_names_dic_swe['lsc'][0]))):
    inpt = av_rPm_df_T_dry_cum50[hrun]
    time0f50input = [((av_rPm_df_T_dry_cum[pd.DataFrame(hru_names_dic_swe['lsc'][0])[0][hrun]]-inpt).abs()).argmin()]
    time0f50input_T_dry.append(time0f50input)

time0f50input_T_wet = []
av_rPm_df_T_wet_cum50 = (av_rPm_df_T_wet_cum.iloc[-1])[1:]/2.
for hrun in range(len(pd.DataFrame(hru_names_dic_swe['lsc'][0]))):
    inpt = av_rPm_df_T_wet_cum50[hrun]
    time0f50input = [((av_rPm_df_T_wet_cum[pd.DataFrame(hru_names_dic_swe['lsc'][0])[0][hrun]]-inpt).abs()).argmin()]
    time0f50input_T_wet.append(time0f50input)
    
    
#%% CC WRF T day of snow disappearance (based on snowdepth)-final output
dosd_df_T_dry, dosd_residual_df_T_dry = calculatingSDD(av_sd_df_T[:][5000:8737],pd.DataFrame(hru_names_dic_swe['lsc'][0]),5976,'2007',3737,5000)
dosd_df_T_wet, dosd_residual_df_T_wet = calculatingSDD(av_sd_df_T[:][14000:],pd.DataFrame(hru_names_dic_swe['lsc'][0]),14976,'2008',3521,14000)

#%%**************************************************************************************************
# ***********************CC WRF T  finding max corespondance swe for '_T_dry and _T_wet'***********************
# 2220000777
max1SWE_T_dry = readSpecificDatafromAllHRUs(av_swe_df_T,pd.DataFrame(hru_names_dic_swe['lsc'][0])[0],5289)
max2SWE_T_dry = readSpecificDatafromAllHRUs(av_swe_df_T,pd.DataFrame(hru_names_dic_swe['lsc'][0])[0],5125)
max3SWE_T_dry = readSpecificDatafromAllHRUs(av_swe_df_T,pd.DataFrame(hru_names_dic_swe['lsc'][0])[0],4886)
max4SWE_T_dry = readSpecificDatafromAllHRUs(av_swe_df_T,pd.DataFrame(hru_names_dic_swe['lsc'][0])[0],4766)

maxSWE_T_dry = (np.array(max1SWE_T_dry)+np.array(max2SWE_T_dry)+np.array(max3SWE_T_dry)+np.array(max4SWE_T_dry))/4.
maxSWE_T_dry_pd = pd.DataFrame(maxSWE_T_dry)
maxSWE_T_dry_pd.index = pd.DataFrame(hru_names_dic_swe['lsc'][0])[0]

#2220000888
max1SWE_T_wet = readSpecificDatafromAllHRUs(av_swe_df_T,pd.DataFrame(hru_names_dic_swe['lsc'][0])[0],13488)
max2SWE_T_wet = readSpecificDatafromAllHRUs(av_swe_df_T,pd.DataFrame(hru_names_dic_swe['lsc'][0])[0],13670)
max3SWE_T_wet = readSpecificDatafromAllHRUs(av_swe_df_T,pd.DataFrame(hru_names_dic_swe['lsc'][0])[0],13815)
max4SWE_T_wet = readSpecificDatafromAllHRUs(av_swe_df_T,pd.DataFrame(hru_names_dic_swe['lsc'][0])[0],14010)

maxSWE_T_wet = (np.array(max1SWE_T_wet)+np.array(max2SWE_T_wet)+np.array(max3SWE_T_wet)+np.array(max4SWE_T_wet))/4.
maxSWE_T_wet_pd = pd.DataFrame(maxSWE_T_wet)
maxSWE_T_wet_pd.index = pd.DataFrame(hru_names_dic_swe['lsc'][0])[0]
   
#%%**************************************************************************************************
## ***********CC WRF T: calculating snowmelt rate based on SWE #cm/day *************************************
#maxSWE_T_wet = readSpecificDatafromAllHRUs(av_swe_df_T,hru_names_df[0],13670)
#2007
sweM1_T,SWE1date_T = SWEandSWEDateforSpecificDate(pd.DataFrame(hru_names_dic_swe['lsc'][0])[0],5289,av_swe_df_T,dosd_df_T_dry)
sweM2_T,SWE2date_T = SWEandSWEDateforSpecificDate(pd.DataFrame(hru_names_dic_swe['lsc'][0])[0],5457,av_swe_df_T,dosd_df_T_dry)
sweM3_T,SWE3date_T = SWEandSWEDateforSpecificDate(pd.DataFrame(hru_names_dic_swe['lsc'][0])[0],5799,av_swe_df_T,dosd_df_T_dry)
sweM4_T,SWE4date_T = SWEandSWEDateforSpecificDate(pd.DataFrame(hru_names_dic_swe['lsc'][0])[0],5965,av_swe_df_T,dosd_df_T_dry)

#2008
sweM5_T,SWE5date_T = SWEandSWEDateforSpecificDate(pd.DataFrame(hru_names_dic_swe['lsc'][0])[0],13488,av_swe_df_T,dosd_df_T_wet)
sweM6_T,SWE6date_T = SWEandSWEDateforSpecificDate(pd.DataFrame(hru_names_dic_swe['lsc'][0])[0],14320,av_swe_df_T,dosd_df_T_wet)
sweM7_T,SWE7date_T = SWEandSWEDateforSpecificDate(pd.DataFrame(hru_names_dic_swe['lsc'][0])[0],14485,av_swe_df_T,dosd_df_T_wet)
sweM8_T,SWE8date_T = SWEandSWEDateforSpecificDate(pd.DataFrame(hru_names_dic_swe['lsc'][0])[0],14800,av_swe_df_T,dosd_df_T_wet)

#cm/day
meltingrate_T_dry1 = meltingRateBetween2days(sweM1_T,sweM2_T,SWE1date_T,SWE2date_T)
meltingrate_T_dry2 = meltingRateBetween2days(sweM2_T,sweM3_T,SWE2date_T,SWE3date_T)
meltingrate_T_dry3 = meltingRateBetween2days(sweM3_T,sweM4_T,SWE3date_T,SWE4date_T)
meltingrate_T_dry4 = np.array(0.1*24)*sweM4_T / abs(SWE4date_T-dosd_df_T_dry.values)

meltingrateAvg_T_dry = []
for countermr in range (np.size(meltingrate_T_dry1)):
    meltingrateAvg_T_dry.append((meltingrate_T_dry1[countermr]+meltingrate_T_dry2[countermr]+
                                 meltingrate_T_dry3[countermr]+meltingrate_T_dry2[countermr])/4.)

meltingrate_T_wet1 = meltingRateBetween2days(sweM5_T,sweM6_T,SWE5date_T,SWE6date_T)
meltingrate_T_wet2 = meltingRateBetween2days(sweM6_T,sweM7_T,SWE6date_T,SWE7date_T)
meltingrate_T_wet3 = meltingRateBetween2days(sweM7_T,sweM8_T,SWE7date_T,SWE8date_T)
meltingrate_T_wet4 = np.array(0.1*24)*sweM8_T / abs(SWE8date_T-dosd_df_T_wet.values)

meltingrateAvg_T_wet = []
for countermr1 in range (np.size(meltingrate_T_wet1)):
    meltingrateAvg_T_wet.append((meltingrate_T_wet1[countermr1]+meltingrate_T_wet2[countermr1]+
                                 meltingrate_T_wet3[countermr1]+meltingrate_T_dry2[countermr])/4.)

#%% plot swe
[dosd_df_h_dry,maxSWE_T_wet_pd, meltingrateAvg_T_wet, time0f50input_T_wet]
n_groups = 4
x11 = ['maxSWE_his_dry','maxSWE_his_wet','maxSWE_T_dry','maxSWE_T_wet']
d11 = [maxSWE_h_dry_pd[0],maxSWE_h_wet_pd[0],maxSWE_T_dry_pd[0],maxSWE_T_wet_pd[0]]

index = np.arange(n_groups)
bar_width = 0.4
opacity = 0.8

safig, saax = plt.subplots(1,1, figsize=(15,10))#

bp3 = saax.boxplot(d11, patch_artist=True)
bp3['boxes'][0].set(color='wheat', linewidth=2, facecolor = 'skyblue', hatch = '/')
bp3['boxes'][1].set(color='darkgreen', linewidth=2, facecolor = 'olive', hatch = '/')
bp3['boxes'][2].set(color='wheat', linewidth=2, facecolor = 'pink', hatch = '/')
bp3['boxes'][3].set(color='darkgreen', linewidth=2, facecolor = 'pink', hatch = '/')

plt.ylabel('maxSWE (mm)', fontsize=30)
plt.yticks(fontsize=40)
plt.xticks(index + bar_width/2.,('maxSWE_his_dry','maxSWE_his_wet','maxSWE_T_dry','maxSWE_T_wet'), fontsize=40)

#plt.title('',fontsize=20)
plt.legend(fontsize=20)

#saax.set_ylabel('SWE(mm)', fontsize=40)

#plt.savefig('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa2_bestSweSD/sa_sa2_swe_taSweMrSDDSST.png')
  
#%%  #cm/day melting rate calcullation based on observation data
sweMR = [711, 550, 309, 84]
mrDate = ['2007-05-09 08:50 5289','2007-05-16 09:00 5457','2007-05-30 09:00 5793','2007-06-06 08:15 5960']  
meltingrate1_obs2007 = np.array([0.1*24*(711-550.)/(5457.-5289)])
meltingrate2_obs2007 = np.array([0.1*24*(550.-309)/(5799.-5457)])
meltingrate3_obs2007 = np.array([0.1*24*(309-84.)/(5965-5799.)])
meltingrate4_obs2007 = np.array([0.1*24*(84.)/(5976-5965)])

dosd_obs2007 = 5976
dosd_obs2008 = 14976-8760.

meltingrate1_obs2008 = np.array([0.1*24*(977-851.)/(14320.-13488)])
meltingrate2_obs2008 = np.array([0.1*24*(851.-739)/(14485.-14320)])
meltingrate3_obs2008 = np.array([0.1*24*(739-381.)/(14800-14485.)])
meltingrate4_obs2008 = np.array([0.1*24*(381.)/(14976-14800.)])
  