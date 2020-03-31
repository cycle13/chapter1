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

#%% defining hrus_name
years = ['2007','2008']#
out_names_swe = ['lsc','lsh','lsp','ssc','ssh','ssp','ljc','ljp','lth','sjh','sjp','stp']

with open("C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/snow_rain_sep/STAR_out_P12_233_L.csv") as safd:
    reader = csv.reader(safd)
    params0 = [r for r in reader]
params1 = params0[1:]
sa_fd_column = []
for csv_counter1 in range (len (params1)):
    for csv_counter2 in range (21):
        sa_fd_column.append(float(params1[csv_counter1][csv_counter2]))
params_sa0=np.reshape(sa_fd_column,(len (params1),21))
params_sa_df12p = pd.DataFrame(params_sa0)#,columns = params0[0]


# reading index of best parameters for each decision model combination 
with open("C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/snow_rain_sep/p1213_sweBestParam_index.csv") as sapr:
    reader1 = csv.reader(sapr)
    params2 = [r1 for r1 in reader1]
params_index = np.array(params2[1:])
params_index_df = pd.DataFrame(params_index, columns = ['lsc','lsh','lsp','ssc','ssh','ssp',
                                                        'ljc','ljp','lth','sjh','sjp','stp'])

#lsh453  lsp441	lsc457	ssh405	ssp412	ssc401	ljp190	ljc290	lth121	sjh162	sjp182	stp116
hruid_dic = {'lsc': [], 'lsh': [], 'lsp': [],  'ssc': [], 'ssh': [], 'ssp': [], 
             'ljc': [], 'ljp': [],  'lth': [], 'sjh': [], 'sjp': [], 'stp': []}
for keys in params_index_df.columns:
    for counter in range (len(params_index_df)):
        if int(params_index_df[keys][counter])>0:
            hruid_dic[keys].append(int(params_index_df[keys][counter]))

index_dic = {'lsc': [], 'lsh': [], 'lsp': [],  'ssc': [], 'ssh': [], 'ssp': [], 
             'ljc': [], 'ljp': [],  'lth': [], 'sjh': [], 'sjp': [], 'stp': []}
for indx in params_index_df.columns:
    index_array = np.array(hruid_dic[indx])-10000
    index_dic[indx].append(index_array)

param_df_ljc = params_sa_df12p.iloc[index_dic['ljc'][0]]
hruidxID = hruid_dic['ljc']
hru_num = np.size(hruidxID)


hru_names_dic_swe = {'lsc': [], 'lsh': [], 'lsp': [], 'ssc': [], 'ssh': [], 'ssp': [],  
                     'ljc': [], 'ljp': [], 'lth': [], 'sjh': [], 'sjp': [], 'stp': []}
for i in out_names_swe:
    hru_names_dic_swe[i].append(['{}{}'.format(i, j) for j in hruid_dic[i]])

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

#%%  reading output files for historical for rainfall snowfall seperation
from allNcFiles_wrf_snowRain import av_ncfiles_h_rs
av_all_h = readAllNcfilesAsDataset(av_ncfiles_h_rs)

DateSa21 = date(av_all_h[0],"%Y-%m-%d") #"%Y-%m-%d %H:%M"
DateSa22 = date(av_all_h[1],"%Y-%m-%d")
date_sa = np.append(DateSa21,DateSa22)
sax = np.arange(0,np.size(date_sa))
sa_xticks = date_sa

# calculating historical snowfall rainfall
av_rf_df_h_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarRainfall',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_dic['ljc'],['ljc'],0)
av_sf_df_h_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_h,'scalarSnowfall',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_dic['ljc'],['ljc'],0)
av_rf_df_h_272 = av_rf_df_h_ljc['ljc10302'][:]
av_rf_df_h_273 = av_rf_df_h_ljc['ljc10307'][:]
av_rf_df_h_274 = av_rf_df_h_ljc['ljc10313'][:]
av_rf_df_h_275 = av_rf_df_h_ljc['ljc10317'][:]
av_sf_df_h_272 = av_sf_df_h_ljc['ljc10302'][:]
av_sf_df_h_273 = av_sf_df_h_ljc['ljc10307'][:]
av_sf_df_h_274 = av_sf_df_h_ljc['ljc10313'][:]
av_sf_df_h_275 = av_sf_df_h_ljc['ljc10317'][:]

    
from allNcFiles_wrf_snowRain import av_ncfiles_T_rs
av_all_T = readAllNcfilesAsDataset(av_ncfiles_T_rs)
av_rf_df_T_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarRainfall',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_dic['ljc'],['ljc'],0)
av_sf_df_T_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T,'scalarSnowfall',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_dic['ljc'],['ljc'],0)
av_rf_df_T_272 = av_rf_df_T_ljc['ljc10302'][:]
av_rf_df_T_273 = av_rf_df_T_ljc['ljc10307'][:]
av_rf_df_T_274 = av_rf_df_T_ljc['ljc10313'][:]
av_rf_df_T_275 = av_rf_df_T_ljc['ljc10317'][:]
av_sf_df_T_272 = av_sf_df_T_ljc['ljc10302'][:]
av_sf_df_T_273 = av_sf_df_T_ljc['ljc10307'][:]
av_sf_df_T_274 = av_sf_df_T_ljc['ljc10313'][:]
av_sf_df_T_275 = av_sf_df_T_ljc['ljc10317'][:]


from allNcFiles_wrf_snowRain import av_ncfiles_T_al_rs
av_all_T_al = readAllNcfilesAsDataset(av_ncfiles_T_al_rs)
av_rf_df_T_al_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarRainfall',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_dic['ljc'],['ljc'],0)
av_sf_df_T_al_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al,'scalarSnowfall',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_dic['ljc'],['ljc'],0)
av_rf_df_T_al_272 = av_rf_df_T_al_ljc['ljc10302'][:]
av_rf_df_T_al_273 = av_rf_df_T_al_ljc['ljc10307'][:]
av_rf_df_T_al_274 = av_rf_df_T_al_ljc['ljc10313'][:]
av_rf_df_T_al_275 = av_rf_df_T_al_ljc['ljc10317'][:]
av_sf_df_T_al_272 = av_sf_df_T_al_ljc['ljc10302'][:]
av_sf_df_T_al_273 = av_sf_df_T_al_ljc['ljc10307'][:]
av_sf_df_T_al_274 = av_sf_df_T_al_ljc['ljc10313'][:]
av_sf_df_T_al_275 = av_sf_df_T_al_ljc['ljc10317'][:]


from allNcFiles_wrf_snowRain import av_ncfiles_T_al_P_rs
av_all_T_al_P = readAllNcfilesAsDataset(av_ncfiles_T_al_P_rs)
av_rf_df_T_al_P_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarRainfall',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_dic['ljc'],['ljc'],0)
av_sf_df_T_al_P_ljc = readVariablefromMultipleNcfilesDatasetasDF2(av_all_T_al_P,'scalarSnowfall',pd.DataFrame(hru_names_dic_swe['ljc'][0]),hruid_dic['ljc'],['ljc'],0)
av_rf_df_T_al_P_272 = av_rf_df_T_al_P_ljc['ljc10302'][:]
av_rf_df_T_al_P_273 = av_rf_df_T_al_P_ljc['ljc10307'][:]
av_rf_df_T_al_P_274 = av_rf_df_T_al_P_ljc['ljc10313'][:]
av_rf_df_T_al_P_275 = av_rf_df_T_al_P_ljc['ljc10317'][:]
av_sf_df_T_al_P_272 = av_sf_df_T_al_P_ljc['ljc10302'][:]
av_sf_df_T_al_P_273 = av_sf_df_T_al_P_ljc['ljc10307'][:]
av_sf_df_T_al_P_274 = av_sf_df_T_al_P_ljc['ljc10313'][:]
av_sf_df_T_al_P_275 = av_sf_df_T_al_P_ljc['ljc10317'][:]


#%% plot max swe
