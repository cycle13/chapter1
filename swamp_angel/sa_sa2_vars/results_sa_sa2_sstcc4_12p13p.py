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
sst_obs_df2= pd.DataFrame(sst_obs_df.values, columns = ['observed_snowSurfaceTemp']) 

sst_obs_morn_df = pd.DataFrame((sst_obs_df['observed_snowSurfaceTemp']['2006-10-01 03:00':'2008-06-13 03:00':24].values+
                                sst_obs_df['observed_snowSurfaceTemp']['2006-10-01 04:00':'2008-06-13 04:00':24].values+
                                sst_obs_df['observed_snowSurfaceTemp']['2006-10-01 05:00':'2008-06-13 05:00':24].values+
                                sst_obs_df['observed_snowSurfaceTemp']['2006-10-01 06:00':'2008-06-13 06:00':24].values)/4.)
sst_obs_morn_df.index = snowdepth_obs_df.index[0:len(sst_obs_morn_df)]

sst_obs_morn_df2 = pd.DataFrame(sst_obs_morn_df.values, columns = ['observed_snowSurfaceTemp_morning']) 
sst_obs_morn_df2.set_index(indx_sd[0:622],inplace=True)

sst_obs_morn2007_ablation_df = sst_obs_morn_df[0]['2007-04-18':'2007-05-23'] #perior_ablationPeriod = ['2007-04-18','2007-04-23','2007-05-02','2007-05-09','2007-05-16','2007-05-23']#,
sst_obs_morn2008_ablation_df = sst_obs_morn_df[0]['2008-04-14':'2008-05-06'] #perior_ablationPeriod = ['2008-04-14 14:45','2008-04-22 12:30','2008-04-28 12:30','2008-05-06 09:15']#,
sst_obs_morn_ablation_df = pd.concat([sst_obs_morn2007_ablation_df,sst_obs_morn2008_ablation_df])

sst_obs_5am2007_ablation_df = sst_obs_df['observed_snowSurfaceTemp']['2007-04-18 05:00':'2007-05-23 06:00':24] #perior_ablationPeriod = ['2007-04-18','2007-04-23','2007-05-02','2007-05-09','2007-05-16','2007-05-23']#,
sst_obs_5am2008_ablation_df = sst_obs_df['observed_snowSurfaceTemp']['2008-04-14 05:00':'2008-05-06 06:00':24] #perior_ablationPeriod = ['2008-04-14 14:45','2008-04-22 12:30','2008-04-28 12:30','2008-05-06 09:15']#,
sst_obs_5am_ablation_df = pd.concat([sst_obs_5am2007_ablation_df,sst_obs_5am2008_ablation_df])

sst_obs_3pm2007_ablation_df = sst_obs_df['observed_snowSurfaceTemp']['2007-04-18 15:00':'2007-05-23 16:00':24] #perior_ablationPeriod = ['2007-04-18','2007-04-23','2007-05-02','2007-05-09','2007-05-16','2007-05-23']#,
sst_obs_3pm2008_ablation_df = sst_obs_df['observed_snowSurfaceTemp']['2008-04-14 15:00':'2008-05-06 16:00':24] #perior_ablationPeriod = ['2008-04-14 14:45','2008-04-22 12:30','2008-04-28 12:30','2008-05-06 09:15']#,
sst_obs_3pm_ablation_df = pd.concat([sst_obs_3pm2007_ablation_df,sst_obs_3pm2008_ablation_df])

#%% #Swamp Angel forcing data
#swampangel_forcing = open('swamp_angel_forcingdata2_corrected.csv', 'rb')
#sa_forcing = csv.reader(swampangel_forcing)#, delimiter=',')
with open("C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa2_bestSweSD/swamp_angel_forcingdata2_corrected_precipCalib_final_L.csv") as safd:
    reader = csv.reader(safd)
    data_forcing = [r for r in reader]
data_forcing2 = data_forcing[1:]
sa_fd_column = []
for csv_counter1 in range (len (data_forcing2)):
    for csv_counter2 in range (11):
        sa_fd_column.append(float(data_forcing2[csv_counter1][csv_counter2]))
sa_forcing=np.reshape(sa_fd_column,(len (data_forcing2),11))

#%%
h = 6.626 * 10 ** (-34.) #m2kg/s 
c = 3. * 10 ** 8 #m/s c is the speed of light (c = 3 × 108m/s),
landa = 11 * 10 ** (-6)        #is the wavelength of the radiation, and 
kB= 1.3806 * 10 ** (-23.)   #kBis the Boltzmann con-stant (m2kg/(s2K))
CI1 = (2 * h * c ** 2)/ (landa ** 5)
CI2 = h * c / (landa *kB)

temp_obj = sst_obs_df['observed_snowSurfaceTemp']+273.15
I_landa_T_obj_tot = CI1/(np.exp(CI2/temp_obj)-1)

temp_air = sa_forcing[:,7]
I_landa_T_air = CI1/(np.exp(CI2/temp_air)-1)

#%% calculated cold content from observation data
coldContent_obs = [0.2659,1.6621,2.3153,3.5530,3.7742,2.8175,1.0309,0.31845,0,0.2134,0.0015,0,0.0464,
                   0,0.0155,0,0,0.9618,3.2154,5.4645,5.0077,4.6845,2.7450,1.9861,0.37033,0.0833,0,0,0,
                   0,0,0]
coldContent_obs_df = pd.DataFrame (coldContent_obs, columns=['cc(Mj/kg/m3)'])
coldContent_obs_df.set_index(obs_swe_ind,inplace=True)
coldContent_obs_df2 = coldContent_obs_df.copy()
coldContent_obs_df2.set_index(pd.DatetimeIndex(date_swe),inplace=True)

#%% defining hrus_name
years = ['2007','2008']#

hruidxID12p = [10317,10327,10336,10369,10375,10524,10526,10726]
hru_num12p = np.size(hruidxID12p)
out_names12p = ["ljc","ljp","sjh","sjh"]#
                       
paramModel12p = (np.size(out_names12p))*(hru_num12p)
hru_names12p =[]
for i in out_names12p:
    hru_names12p.append(['{}{}'.format(j, i) for j in hruidxID12p])
hru_names112p = np.reshape(hru_names12p,(paramModel12p,1))
hru_names_df12p = pd.DataFrame (hru_names112p)


hruidxID13p = [10030,10041,10042,10053,10054,10077,10078,10079,10081,10082,10083,10084,10085,10094,10095,
               10096,10097,10098,10103,10104,10105,10106,10107,10108,10109,10110,10111,10112,10113,10114,
               10115,10116,10117,10118,10119,10120,10122,10126,10156,10223,10224,10225,10234,10235,10336,
               10337,10342,10348,10365,10379,10380,10381,10392,10393,10394,10398,10399,10400,10408,10409,
               10410,10411,10412,10413,10414,10421,10423,10426,10427,10428,10429,10430,10431,10432,10433,
               10436,10437,10438,10439,10440,10441,10442,10443,10444,10453,10454,10460,10461,10462,10617,
               10625,10626,10628,10637,10650,10652,10653,10669,10670,10737,10749,10752,10756,10767,10768,
               10769,10788,10790,10794,10795,10796,10825,10826,10827,10833,10834,10835,10836,10837,10841,
               10850,10865,10866,10867,10879,10880,10881,10882,10883,10884,10885,10886,10887,10888,10904,
               10910,10911,10912,10942,10945,10948,10951,10958,10959,10960,11272,11273,11274]  
hru_num13p = np.size(hruidxID13p)
out_names13p = ["lsc","lsh","lsp","ssc","ssh","ssp",]#
                       
paramModel13p = (np.size(out_names13p))*(hru_num13p)
hru_names13p =[]
for i in out_names13p:
    hru_names13p.append(['{}{}'.format(j, i) for j in hruidxID13p])
hru_names113p = np.reshape(hru_names13p,(paramModel13p,1))
hru_names_df13p = pd.DataFrame (hru_names113p)

out_names = pd.concat([pd.DataFrame(out_names12p),pd.DataFrame(out_names13p)], ignore_index = 'True')
hru_names_df = pd.concat([hru_names_df12p,hru_names_df13p], ignore_index = 'True')

#%%  reading output files
from allNcFiles_sa2_stcc import av_ncfiles
from allNcFiles_sa2 import av_ncfiles13p

av_all = readAllNcfilesAsDataset(av_ncfiles)
av_all13p = readAllNcfilesAsDataset(av_ncfiles13p)

av_sd_df12p = readVariablefromMultipleNcfilesDatasetasDF(av_all,'scalarSnowDepth',hru_names_df12p,hruidxID12p,out_names12p)
av_swe_df12p = readVariablefromMultipleNcfilesDatasetasDF(av_all,'scalarSWE',hru_names_df12p,hruidxID12p,out_names12p)

av_sd_df13p = readVariablefromMultipleNcfilesDatasetasDF(av_all13p,'scalarSnowDepth',hru_names_df13p,hruidxID13p,out_names13p)
av_swe_df13p = readVariablefromMultipleNcfilesDatasetasDF(av_all13p,'scalarSWE',hru_names_df13p,hruidxID13p,out_names13p)
#
av_sd_df = pd.concat([av_sd_df12p,av_sd_df13p], axis = 1)
av_swe_df = pd.concat([av_swe_df12p,av_swe_df13p], axis = 1)


av_st_df12p = readVariablefromMultipleNcfilesDatasetasDF(av_all,'scalarSurfaceTemp',hru_names_df12p,hruidxID12p,out_names12p)
av_st_df13p = readVariablefromMultipleNcfilesDatasetasDF(av_all13p,'scalarSurfaceTemp',hru_names_df13p,hruidxID13p,out_names13p)
av_st_df = pd.concat([av_st_df12p,av_st_df13p], axis = 1)

av_sst_date = np.append(date(av_all[0],"%Y-%m-%d %H:%M"),date(av_all[1],"%Y-%m-%d %H:%M"))
av_st_df.index = av_sst_date
av_st_df2 =pd.DataFrame(av_st_df.values, columns=av_st_df.columns)

# snow surface temp in the morning
av_st_morn_df = pd.DataFrame((av_st_df[:]['2006-10-01 03:00':'2008-06-13 04:00':24].values+
                              av_st_df[:]['2006-10-01 04:00':'2008-06-13 05:00':24].values+
                              av_st_df[:]['2006-10-01 05:00':'2008-06-13 06:00':24].values+
                              av_st_df[:]['2006-10-01 06:00':'2008-06-13 07:00':24].values)/4.)
av_st_morn_df.index = snowdepth_obs_df.index[0:len(sst_obs_morn_df)]
av_st_morn_df2 = pd.DataFrame(av_st_morn_df.values, columns=av_st_df2.columns)
av_st_morn_df2.set_index(sst_obs_morn_df2.index,inplace=True)

# snow surface temp in 5am perior to ablation (p2a)
av_st_5amp2a_df2007 = pd.DataFrame(av_st_df[:]['2007-04-18 05:00':'2007-05-23 06:00':24])
av_st_5amp2a_df2008 = pd.DataFrame(av_st_df[:]['2008-04-14 05:00':'2008-05-06 06:00':24])
av_st_5amp2a_df = pd.concat([av_st_5amp2a_df2007,av_st_5amp2a_df2008])#, ignore_index = 'True'
av_st_5amp2a_df2 = pd.DataFrame(av_st_5amp2a_df.values, columns=av_st_5amp2a_df.columns)
av_st_5amp2a_df2.set_index(av_st_5amp2a_df['counter'],inplace=True)

# snow surface temp in 3pm perior to ablation (p2a)
av_st_3pmp2a_df2007 = pd.DataFrame(av_st_df[:]['2007-04-18 15:00':'2007-05-23 16:00':24])
av_st_3pmp2a_df2008 = pd.DataFrame(av_st_df[:]['2008-04-14 15:00':'2008-05-06 16:00':24])
av_st_3pmp2a_df = pd.concat([av_st_3pmp2a_df2007,av_st_3pmp2a_df2008])#, ignore_index = 'True'
av_st_3pmp2a_df2 = pd.DataFrame(av_st_3pmp2a_df.values, columns=av_st_3pmp2a_df.columns)
av_st_3pmp2a_df2.set_index(av_st_3pmp2a_df['counter'],inplace=True)

#%%calculating modeled cold content for 12 parameters
#number of snow layer for dates of observed swe
av_nsl_df = readVariablefromMultipleNcfilesDatasetasDF(av_all,'nSnow',hru_names_df,hruidxID12p,out_names12p)
nsnow = [av_nsl_df.values[ns,1:] for ns in obs_swe_ind]

# snow layers temperature for dates of observed swe
av_mlt_df = readVariablefromMultipleNcfilesDatasetasDF(av_all,'mLayerTemp',hru_names_df,hruidxID12p,out_names12p)
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

#height of each layer for dates of observed swe
av_mld_df = readVariablefromMultipleNcfilesDatasetasDF(av_all,'mLayerDepth',hru_names_df,hruidxID12p,out_names12p)
layerDepth4dates = readData4multibleLayers4specificDateofAvailableSWE(av_mld_df,obs_swe_ind,hru_names_df,sum0flayers0,sum0flayers1)
snowLayerDepth4dates = readData4multibleSnowLayers4specificDateofAvailableSWE(av_mld_df,obs_swe_ind,hru_names_df,sum0flayers0,sum0flayers1)
snowLayerHeight4dates = []
for dates in range(len(obs_swe_ind)):
    snowLayerHeight_hru = []
    for hrus in range (len(hru_names_df[0])):
        snowLayerDepth4dates_sup1 = pd.concat([pd.DataFrame([0.0]),snowLayerDepth4dates[dates][hrus]])
        snowLayerDepth4dates_sup2 = snowLayerDepth4dates[dates][hrus].append(pd.DataFrame([0.0]))
        snowLayerHeight = abs((np.array(snowLayerDepth4dates_sup2[0])-np.array(snowLayerDepth4dates_sup1[0]))[:-1])
        snowLayerHeight_hru.append(snowLayerHeight)
    snowLayerHeight4dates.append(snowLayerHeight_hru)

#volumetric fraction of ice and liquid for dates of observed swe   
av_vfi_df = readVariablefromMultipleNcfilesDatasetasDF(av_all,'mLayerVolFracIce',hru_names_df,hruidxID12p,out_names12p)
layerVolFracIce4dates = readData4multibleLayers4specificDateofAvailableSWE(av_vfi_df,obs_swe_ind,hru_names_df,sum0flayers0,sum0flayers1)
snowLayerVolFracIce4dates = readData4multibleSnowLayers4specificDateofAvailableSWE(av_vfi_df,obs_swe_ind,hru_names_df,sum0flayers0,sum0flayers1)

av_vfl_df = readVariablefromMultipleNcfilesDatasetasDF(av_all,'mLayerVolFracLiq',hru_names_df,hruidxID12p,out_names12p)
layerVolFracLiq4dates = readData4multibleLayers4specificDateofAvailableSWE(av_vfl_df,obs_swe_ind,hru_names_df,sum0flayers0,sum0flayers1)
snowLayerVolFracLiq4dates = readData4multibleSnowLayers4specificDateofAvailableSWE(av_vfl_df,obs_swe_ind,hru_names_df,sum0flayers0,sum0flayers1)

#volumetric heat capacity of snow layers for dates of observed swe  
av_mlhci_df = readVariablefromMultipleNcfilesDatasetasDF(av_all,'mLayerVolHtCapBulk',hru_names_df,hruidxID12p,out_names12p)
layerHeatCap4dates = readData4multibleLayers4specificDateofAvailableSWE(av_mlhci_df,obs_swe_ind,hru_names_df,sum0flayers0,sum0flayers1)
snowLayerHeatCap4dates = readData4multibleSnowLayers4specificDateofAvailableSWE(av_mlhci_df,obs_swe_ind,hru_names_df,sum0flayers0,sum0flayers1)


#coldContentCalculation
densityofWater = 997. #kg/m³
densityofIce = 917. #kg/m3
coldcontent4Dates = []
for dates in range(len(obs_swe_ind)):
    cct_hru =[]
    for hrus in range (len(hru_names_df[0])):
        swe = (snowLayerVolFracIce4dates[dates][hrus]*densityofWater/1000. + 
               snowLayerVolFracLiq4dates[dates][hrus]*densityofIce/1000.) #m
        temp = snowLayerTemp4dates[dates][hrus] - 273.15
        HCItHS = snowLayerHeatCap4dates[dates][hrus] * snowLayerHeight4dates[dates][hrus]
        cct = sum(list(swe*temp*HCItHS/1000000.)) #MJ m-3 K-1
        cct_hru.append(abs(cct))
    coldcontent4Dates.append(cct_hru)

coldcontent4Dates_df = pd.DataFrame(coldcontent4Dates, columns = hru_names_df[0])
coldcontent4Dates_df2 = coldcontent4Dates_df.copy()
coldcontent4Dates_df2.set_index(pd.DatetimeIndex(date_swe),inplace=True)

#%% day of snow disappearance (based on snowdepth)-final output
dosd_df2007, dosd_residual_df2007 = calculatingSDD(av_sd_df[:][5000:8737],hru_names_df,5976,'2007',3737,5000)
dosd_df2008, dosd_residual_df2008 = calculatingSDD(av_sd_df[:][14000:],hru_names_df,14976,'2008',3521,14000)

dosd_residual_df_sum = abs(dosd_residual_df2007)+abs(dosd_residual_df2008)

dosd_obs2007 = 5976
dosd_obs2008 = 14976-8760.
dosd_normal = (dosd_obs2008 + dosd_obs2007)/24.

#%%**************************************************************************************************
# *********************** finding max corespondance swe for '2007 and 2008'***********************
#'2007-04-18' 4776: 4800, '2007-04-23' 4896:4920, '2007-05-02' 5112:5136
#Group1: '2007-03-12 14:00' (3902),'2007-03-19 12:30 (4068)','2007-03-26 12:30 (4236)','2007-04-02 12:30'(4404),
#Group2: '2007-04-18 08:35' (4784),'2007-04-23 10:30 (4907)','2007-05-02 08:40'(5121), 
# 2220000777
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

maxSWE_residual2007_perc = ((abs(maxSWE1_residual2007)/max1SWE_obs2007[0]+
                            abs(maxSWE2_residual2007)/max2SWE_obs2007[0]+
                            abs(maxSWE3_residual2007)/max3SWE_obs2007[0]+
                            abs(maxSWE4_residual2007)/max4SWE_obs2007[0])/4.).T#+
maxSWE_residual2007_perc.index = hru_names_df[0]


#2220000888
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

maxSWE_residual2008_perc = ((abs(max1SWE_residual2008)/max1SWE_obs2008[0]+
                            abs(max2SWE_residual2008)/max2SWE_obs2008[0]+
                            abs(max3SWE_residual2008)/max3SWE_obs2008[0]+
                            abs(max4SWE_residual2008)/max4SWE_obs2008[0])/4.).T
maxSWE_residual2008_perc.index = hru_names_df[0]

# max delta swe (modeled - observed)
maxSWE_residual_df_mean = abs(maxSWE_residual2008_perc)/2+abs(maxSWE_residual2007_perc)/2

#max swe objective function
maxSWE_residual_df_perc_thresh = maxSWE_residual_df_mean[0][maxSWE_residual_df_mean[0]<=0.10]
inx_maxSWE_residual_df_perc_thresh = maxSWE_residual_df_perc_thresh.index
   
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

#cm/day
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

#%% melting rate objective function
meltRate_residual_mean_perc = (abs(meltingrate20071-meltingrate1_obs2007)/meltingrate1_obs2007 +
                               abs(meltingrate20072-meltingrate2_obs2007)/meltingrate2_obs2007 +
                               abs(meltingrate20073-meltingrate3_obs2007)/meltingrate3_obs2007 +
                               abs(meltingrate20081-meltingrate1_obs2008)/meltingrate1_obs2008 +
                               abs(meltingrate20083-meltingrate3_obs2008)/meltingrate3_obs2008 +
                               abs(meltingrate20082-meltingrate2_obs2008)/meltingrate2_obs2008)/6.
meltRate_residual_df_mean_perc = pd.DataFrame(meltRate_residual_mean_perc)

meltRate_residual_df_mean_perc.index = hru_names_df[0]
meltRate_residual_df_mean_perc_bestSWE = meltRate_residual_df_mean_perc[0][inx_maxSWE_residual_df_perc_thresh[:]]
ind_df_mean_perc_bestSWE_bestMR = (meltRate_residual_df_mean_perc_bestSWE[meltRate_residual_df_mean_perc_bestSWE<=0.2]).index

#%% snow disappearance date objective function
dosd_residual_df_sum_r_bestSWE_bestMR = dosd_residual_df_sum[0][ind_df_mean_perc_bestSWE_bestMR]
ind_df_mean_perc_bestSWE_bestMR_bestSDD = (dosd_residual_df_sum_r_bestSWE_bestMR[dosd_residual_df_sum_r_bestSWE_bestMR<=12]).index

#%% snow surface temp morning objective function
sst_morn2007_ablation_df = av_st_morn_df['2007-04-18':'2007-05-23'] #perior_ablationPeriod = ['2007-04-18','2007-04-23','2007-05-02','2007-05-09','2007-05-16','2007-05-23']#,
sst_morn2008_ablation_df = av_st_morn_df[:]['2008-04-14':'2008-05-06'] #perior_ablationPeriod = ['2008-04-14 14:45','2008-04-22 12:30','2008-04-28 12:30','2008-05-06 09:15']#,
sst_morn_ablation_df = pd.concat([sst_morn2007_ablation_df,sst_morn2008_ablation_df])
sst_morn_ablation_df.drop(0, axis = 'columns', inplace = True)
sst_morn_ablation_df2 = sst_morn_ablation_df.copy()
sst_morn_ablation_df2.columns=hru_names_df[0]

sst_morn_ablation_residual = abs(sst_morn_ablation_df.subtract((sst_obs_morn_ablation_df+273.15), axis = 0))
sst_morn_ablation_residual_mean_df = pd.DataFrame(np.mean(sst_morn_ablation_residual, axis = 0))
sst_morn_ablation_residual_mean_df.index = hru_names_df[0]

sst_obs_morn_ablation_mean = abs(np.mean(sst_obs_morn_ablation_df, axis =0))

sst_morn_ablation_delta_mean_df_perc = sst_morn_ablation_residual_mean_df/sst_obs_morn_ablation_mean

delta_sst_df_mean_r_bestSWE_bestMR_bestSDD = sst_morn_ablation_delta_mean_df_perc[0][ind_df_mean_perc_bestSWE_bestMR_bestSDD]
ind_df_mean_perc_bestSWE_bestMR_bestSDD_bestSSTM0r = (delta_sst_df_mean_r_bestSWE_bestMR_bestSDD[delta_sst_df_mean_r_bestSWE_bestMR_bestSDD<=0.25]).index

#snow surface temp second objective function (rate between 5 am and 3 pm)
sst_obs_ratep2a_df = pd.DataFrame((sst_obs_5am_ablation_df.values-sst_obs_3pm_ablation_df.values)/10.)
sst_obs_ratep2a_df.set_index(sst_obs_morn_ablation_df.index, inplace = True)
sst_obs_ratep2a_mean = np.mean(sst_obs_ratep2a_df, axis =0).values

sst_ratep2a_df = pd.DataFrame((av_st_5amp2a_df2.values-av_st_3pmp2a_df2.values)/10.)
sst_ratep2a_df.set_index(sst_obs_morn_ablation_df.index, inplace = True)
sst_ratep2a_df.drop(0, axis = 'columns', inplace = True)
sst_ratep2a_mean = np.mean(sst_ratep2a_df, axis =0)
sst_ratep2a_df2 = sst_ratep2a_df.copy()
sst_ratep2a_df2.columns = hru_names_df[0]

sst_ratep2a_delta_perc = (sst_ratep2a_mean-sst_obs_ratep2a_mean)/sst_obs_ratep2a_mean
sst_ratep2a_delta_perc.index = hru_names_df[0]

delta_sstRate_df_mean_r_bestSWE_bestMR_bestSDD = sst_ratep2a_delta_perc[ind_df_mean_perc_bestSWE_bestMR_bestSDD_bestSSTM0r]
ind_df_mean_perc_bestSWE_bestMR_bestSDD_bestSST_bestSstRate = (delta_sstRate_df_mean_r_bestSWE_bestMR_bestSDD[delta_sstRate_df_mean_r_bestSWE_bestMR_bestSDD<=0.25]).index

#%% cold content objective function
coldcontent_p2a2007_df = coldcontent4Dates_df2['2007-03-12 14:00':'2007-03-26 12:30']
coldcontent_p2a2008_df = coldcontent4Dates_df2['2008-03-24 09:10':'2008-04-22 12:30']
coldcontent_p2a_df = pd.concat([coldcontent_p2a2007_df,coldcontent_p2a2008_df])
coldcontent_p2a_df2 = coldcontent_p2a_df.copy()
coldcontent_p2a_df2.columns = np.arange(1,len(coldcontent_p2a_df.columns)+1)

coldcontent_obs_p2a2007_df = coldContent_obs_df2['2007-03-12 14:00':'2007-03-26 12:30']
coldcontent_obs_p2a2008_df = coldContent_obs_df2['2008-03-24 09:10':'2008-04-22 12:30']
coldcontent_obs_p2a_df = pd.concat([coldcontent_obs_p2a2007_df,coldcontent_obs_p2a2008_df])
coldcontent_obs_p2a_df2 = coldcontent_obs_p2a_df.copy()
coldcontent_obs_p2a_df2.columns = np.arange(1,len(coldcontent_obs_p2a_df.columns)+1)

cc_ablation_deta = abs(coldcontent_p2a_df2.values - coldcontent_obs_p2a_df2.values)
cc_ablation_perc = abs(coldcontent_p2a_df2.values - coldcontent_obs_p2a_df2.values)/coldcontent_obs_p2a_df2.values
cc_ablation_mean_perc = np.mean(cc_ablation_perc, axis = 0)
cc_ablation_mean_perc_df = pd.DataFrame(cc_ablation_mean_perc)
cc_ablation_mean_perc_df.index = hru_names_df[0]

delta_cc_df_mean_r_bestSWE_bestMR_bestSDD_bestSST = cc_ablation_mean_perc_df[0][ind_df_mean_perc_bestSWE_bestMR_bestSDD_bestSST_bestSstRate]
ind_df_mean_perc_bestSWE_bestMR_bestSDD_bestSST_bestSstRate_bestCC = (delta_cc_df_mean_r_bestSWE_bestMR_bestSDD_bestSST[delta_cc_df_mean_r_bestSWE_bestMR_bestSDD_bestSST<=0.8]).index

#%%  plot best cold content
DateSa21 = date(av_all[0],"%Y-%m-%d") #"%Y-%m-%d %H:%M"
DateSa22 = date(av_all[1],"%Y-%m-%d")
date_sa = np.append(DateSa21,DateSa22)
sax = np.arange(0,np.size(date_sa))
sa_xticks = date_sa

setXtickCc = ['2007-03-12','2007-03-19','2007-03-26','2008-03-24','2008-04-01','2008-04-14','2008-04-22']
safig4, saax4 = plt.subplots(1,1, figsize=(30,20))#
for hru4 in ind_df_mean_perc_bestSWE_bestMR_bestSDD_bestSST_bestSstRate_bestCC:#
    plt.scatter(np.arange(1,len(coldcontent_p2a_df)+1),coldcontent_p2a_df[hru4], s =2500) ##, color  = 'green', linewidth=4, 

plt.scatter(np.arange(1,len(coldcontent_obs_p2a_df)+1),coldcontent_obs_p2a_df, s =3000, color  = 'k') #sst_obs_morn_ablation_df.index, linewidth=4[0:16], markersize=15
plt.yticks(fontsize=40)
plt.xlabel('Time 2006-2008', fontsize=40)
saax4.set_ylabel('cold content perior to the ablation period', fontsize=40)
plt.xticks(np.arange(1,len(coldcontent_p2a_df)+1),setXtickCc,rotation=25, fontsize=40)# sax[::2000], sa_xticks[::2000], 
safig4.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa2_bestSweSD/sa_sa2_cc_4c_lastP_bestSWEMrSSTM0rCc.png')

#%% plot swe
safig, saax = plt.subplots(1,1, figsize=(30,20))#
color = []

for hru in ind_df_mean_perc_bestSWE_bestMR_bestSDD_bestSST_bestSstRate:
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
#%%  plot best hourly snow surface temperature
safig4, saax4 = plt.subplots(1,1, figsize=(30,20))#
for hru4 in ind_df_mean_perc_bestSWE_bestMR_bestSDD_bestSST_bestSstRate:#
    plt.plot(av_st_df2[hru4]-273.15, linewidth=4)

plt.plot(sst_obs_df2.index, sst_obs_df2['observed_snowSurfaceTemp'], 'k', linewidth=4)#[0:16], markersize=15
plt.yticks(fontsize=40)
plt.xlabel('Time 2006-2008', fontsize=40)
saax4.set_ylabel('Hourly snow surface temperature(C)', fontsize=40)
plt.xticks(sax[::2000], sa_xticks[::2000], rotation=25, fontsize=40)# 
safig4.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa2_bestSweSD/sa_sa2_sst_4c_lastP_bestSWEMrSST.png')
#%%  plot best morning snow surface temperature
setXtick = ['2007-04-18','2007-04-19','2007-04-20','2007-04-21','2007-04-22','2007-04-23','2007-04-24', '2007-04-25',
            '2007-04-26','2007-04-27','2007-04-28','2007-04-29','2007-04-30','2007-05-01','2007-05-02','2007-05-03',
            '2007-05-04','2007-05-05','2007-05-06','2007-05-07','2007-05-08','2007-05-09','2007-05-10','2007-05-11',
            '2007-05-12','2007-05-13','2007-05-14','2007-05-15','2007-05-16','2007-05-17','2007-05-18','2007-05-19',
            '2007-05-20','2007-05-21','2007-05-22','2007-05-23','2008-04-14','2008-04-15','2008-04-16','2008-04-17',
            '2008-04-18','2008-04-19','2008-04-20','2008-04-21','2008-04-22','2008-04-23','2008-04-24','2008-04-25',
            '2008-04-26','2008-04-27','2008-04-28','2008-04-29','2008-04-30','2008-05-01','2008-05-02','2008-05-03',
            '2008-05-04','2008-05-05','2008-05-06']

safig4, saax4 = plt.subplots(1,1, figsize=(30,20))#
for hru4 in ind_df_mean_perc_bestSWE_bestMR_bestSDD_bestSST_bestSstRate:
    #print (sst_morn_ablation_df2[hru4])#
    plt.scatter(np.arange(1,len(sst_morn_ablation_df2)+1),sst_morn_ablation_df2[hru4]-273.15, s =1500, color  = 'hotpink') ##, linewidth=4

plt.scatter(np.arange(1,len(sst_morn_ablation_df2)+1),sst_obs_morn_ablation_df, s =2000, color  = 'k') #sst_obs_morn_ablation_df.index, linewidth=4[0:16], markersize=15
plt.yticks(fontsize=40)
plt.xlabel('Prior to ablation period in 2007 and 2008', fontsize=40)
saax4.set_ylabel('snow surface temperature before sunrise(C)', fontsize=40)
plt.xticks(np.arange(1,len(sst_morn_ablation_df2)+1)[::5],setXtick[::5],rotation=25, fontsize=40)# sax[::2000], sa_xticks[::2000], 
#saax4.set_xticks(setXtick)
safig4.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa2_bestSweSD/sa_sa2_sstM0r_4c_lastP_bestSWEMrSSTRate.png')

#%%  plot best rate of snow surface temperature
safig4, saax4 = plt.subplots(1,1, figsize=(30,20))#
for hru4 in ind_df_mean_perc_bestSWE_bestMR_bestSDD_bestSST_bestSstRate:#
    plt.scatter(np.arange(1,len(sst_ratep2a_df2)+1),sst_ratep2a_df2[hru4], s =1500, color  = 'green') ##, linewidth=4, 

plt.scatter(np.arange(1,len(sst_obs_ratep2a_df)+1),sst_obs_ratep2a_df, s =2000, color  = 'k') #sst_obs_morn_ablation_df.index, linewidth=4[0:16], markersize=15
plt.yticks(fontsize=40)
plt.xlabel('Time 2006-2008', fontsize=40)
saax4.set_ylabel('Hourly rate of change in SST from 5am to 3pm (C/hr)', fontsize=40)
plt.xticks(np.arange(1,len(sst_ratep2a_df2)+1)[::5],setXtick[::5],rotation=25, fontsize=40)# sax[::2000], sa_xticks[::2000], 
safig4.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa2_bestSweSD/sa_sa2_sstRate_4c_lastP_bestSWEMrSSTM0r.png')

#%% plot snow depth
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
#%% ploting snow surface temperature 

safig3, saax3 = plt.subplots(1,1, figsize=(30,20))#
for hru3 in range (len(hru_names_df[0])):#
    plt.plot(av_st_df2[hru_names_df[0][hru3]]-273.15, linewidth=4)

plt.plot(sst_obs_df2['observed_snowSurfaceTemp'], 'k')#, linewidth=6[0:16], markersize=15
plt.yticks(fontsize=40)
plt.xlabel('Time 2006-2008', fontsize=40)
saax3.set_ylabel('snow surface temperature (C)', fontsize=40)
plt.xticks(sax[::3000], sa_xticks[::3000], rotation=25, fontsize=40)# 
safig3.tight_layout()  
plt.savefig('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa2_bestSweSD/sa_sa2_sst_4c_lastP_sst1.png')


safig4, saax4 = plt.subplots(1,1, figsize=(30,20))#
for hru4 in range (len(hru_names_df[0])):#
    plt.scatter(av_st_morn_df2.index,av_st_morn_df2[hru_names_df[0][hru4]]-273.15, s = 500)#, linewidth=4

plt.scatter(sst_obs_morn_df2.index, sst_obs_morn_df2['observed_snowSurfaceTemp_morning'], c = 'k', s =500)#, linewidth=4[0:16], markersize=15
plt.yticks(fontsize=40)
plt.xlabel('Time 2006-2008', fontsize=40)
saax4.set_ylabel('snow surface temperature at before sunrise (C)', fontsize=40)
plt.xticks(sax[::2000], sa_xticks[::2000], rotation=25, fontsize=40)# 
safig4.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa2_bestSweSD/sa_sa2_sst5am_4c_lastP_sst1_scatter.png')

#%% ploting annual swe curves 
DateSa21 = date(av_all[0],"%Y-%m-%d") #"%Y-%m-%d %H:%M"
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


  
  