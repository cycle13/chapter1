###       /bin/bash runTestCases_docker.sh

# anaconda prompt
# conda install netcdf4 
#import pandas as pd
#import datetime as dt 
from scipy.io import netcdf
from netCDF4 import Dataset
import numpy as np
from netCDF4 import netcdftime
from netCDF4 import num2date
from netCDF4 import date2num
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#%% Getting Data
### Senator Beck
#finS = netcdf.netcdf_file('senatorBeck_SASP_1hr.nc', 'r')
finS = Dataset("forcing_sheltered.nc","r",format="NETCDF4")
for v in finS.variables:
    print(v)

#a1 = finS.variables['zs_sheltered'][:]
#a2 = finS.variables['ustar'][:]
#a3 = finS.variables['turbFlux'][:]

#print a1
#%%
# ***** How to get time value into Python DateTIme Objects *****

#tname = "time"
#
TimeS = finS.variables['time'][:] # get values
#t_unitS = finS.variables['time'].units # get unit  "days since 1950-01-01T00:00:00Z"
#
#
#try :
#
#    t_cal = finS.variables[tname].calendar
#
#except AttributeError : # Attribute doesn't exist
#
#    t_cal = u"gregorian" # or standard
#
#tvalueS = num2date(TimeS, units=t_unitS, calendar=t_cal)
#DateS = [i.strftime("%Y-%m-%d %H:%M") for i in tvalueS] # to display dates as string #i.strftime("%Y-%m-%d %H:%M")
##print DateS [69000]
#
#SD = finS.variables['snowDepth'][:] # get values

#%% Plotting figures Humidity Senator Beck

#Sx = np.arange(0,np.size(DateS))
Sy = np.array(a1)
plt.plot(TimeS, Sy, color='aqua')
#plt.xlim(0, 300000000)
#plt.ylim(0, 200)

Sy = np.array(a3)
plt.plot(TimeS, Sy, color='red')
plt.xlim(0, 300000000)
plt.ylim(0, 200)
#%%
#Smy_xticks = DateS
#Sfig, Sax = plt.subplots(1,1)
#plt.xticks(Sx, Smy_xticks[::9000], rotation=25)
#Sax.xaxis.set_major_locator(ticker.AutoLocator())
#
#plt.plot(Sx, Sy, label='Specific Humidity (g/m3)', color='aqua')
#plt.title('Hourly Average Specific Humidity for Senator Beck')
#plt.xlabel('Date 2003 to 2008')
#plt.ylabel('Specific Humidity (g/m3)')
#plt.show()
#

#%%    
#try:
#    import netcdftime.utime.date2num as date2num
#except ImportError:
#    import netcdftime.date2num as date2num

#datevar = []
#
#datevar.append(netcdftime.num2date(nctime, units=t_unit, calendar = t_cal))

# ***** End *****
##%%
## FIX TIME UNITS
#from netCDF4 import num2date
#nctime = fin.variables['time'][:]    # get values
#t_unit = fin.variables['time'].units # get unit  "seconds since 1990-01-01 00:00:00Z"


#%%
#HRUS = finS.variables['hruId'][:]
#print HRUS
#HumidityS=1292*finS.variables['spechum'][:] #g/m3
#TempS=finS.variables['airtemp'][:] - 273
#swrs = finS.variables['SWRadAtm'][:]
#swrs2=[]
#for e in range (np.size(swrs)):
#    if swrs[e]==-9999:
#        hhs=0
#        swrs2.append(hhs)
#    else:
#        hhs=swrs[e]
#        swrs2.append(hhs)
#        
##print swrs2[39167:41000]
#
#lwrs = finS.variables['LWRadAtm'][:]
#lwrs2=[]
#for f in range (np.size(lwrs)):
#    if lwrs[f]==-9999:
#        hhs2=0
#        lwrs2.append(hhs2)
#    else:
#        hhs2=lwrs[f]
#        lwrs2.append(hhs2)
#        
##print lwrs2[39167:41000]
#
#precips = 3600*finS.variables['pptrate'][:]
#winds = finS.variables['windspd'][:]
#TimeS = finS.variables['time'][:]
#
#### Renolds Creek
#finR = netcdf.netcdf_file('forcing_above_aspen.nc', 'r')
##for w in finR.variables:
##    print(w)
#HRUR = finR.variables['hruId'][:]
#HumidityR=1292*finR.variables['spechum'][:] #g/m3
#TempR=finR.variables['airtemp'][:] - 273    # C
#swrr = finR.variables['SWRadAtm'][:]
#lwrr = finR.variables['LWRadAtm'][:]
#precipr = 3600*finR.variables['pptrate'][:]
#windr = finR.variables['windspd'][:]
#TimeR = finR.variables['time'][:]
#print finR.variables['windspd'].units
#plt.plot(TimeR, HumidityR)
#%% Number to Date Senator Beck
#t_unitS = finS.variables['time'].units # get unit  
#t_calS = finS.variables['time'].calendar
#tvalueS = num2date(TimeS, units=t_unitS, calendar=t_calS)
#DateS = [i.strftime("%Y-%m-%d %H:%M") for i in tvalueS] # to display dates as string #i.strftime("%Y-%m-%d %H:%M")
#
## Number to Date Reynolds Creek
#t_unitR = finR.variables['time'].units # get unit  
#t_calR = finR.variables['time'].calendar
#tvalueR = num2date(TimeR, units=t_unitR, calendar=t_calR)
#DateR = [i.strftime("%Y-%m-%d %H:%M") for i in tvalueR] # to display dates as string
#%% Plotting figures Humidity Senator Beck
#Sx = np.arange(0,np.size(DateS))
#Sy = np.array(HumidityS)
#Smy_xticks = DateS
#Sfig, Sax = plt.subplots(1,1)
#plt.xticks(Sx, Smy_xticks[::9000], rotation=25)
#Sax.xaxis.set_major_locator(ticker.AutoLocator())
#
#plt.plot(Sx, Sy, label='Specific Humidity (g/m3)', color='aqua')
#plt.title('Hourly Average Specific Humidity for Senator Beck')
#plt.xlabel('Date 2003 to 2008')
#plt.ylabel('Specific Humidity (g/m3)')
#plt.show()
#
## Plotting figures Humidity Reynolds Creek
#Rx = np.arange(0,np.size(DateR))
#Ry = np.array(HumidityR)
#Rmy_xticks = DateR
#Rfig, Rax = plt.subplots(1,1)
#plt.xticks(Rx, Rmy_xticks[::9000], rotation=25)
#Rax.xaxis.set_major_locator(ticker.AutoLocator())
#
#plt.plot(Rx, Ry, label='Specific Humidity (g/m3)', color='orange')
#plt.title('Hourly Average Specific Humidity for Renolds')
#plt.xlabel('Date 2003 to 2008')
#plt.ylabel('Specific Humidity (g/m3)')
#plt.show()
#%%  Plotting figures Humidity
#Sx = np.arange(0,np.size(DateS))
#Sx = np.array(TimeS)
#Sy = np.array(HumidityS)
#
#Rx = np.array(TimeR)
#Ry = np.array(HumidityR)
#
#DateT = list(set(DateS + DateR)) #Combine lists and remove duplicates python
#DateT.sort()
##resultList= List(set(first_list)|set(second_list))
#x = np.arange(0,np.size(DateT))
#my_xticks = DateT
#
#fig, ax = plt.subplots(1,1)
#plt.xticks(x, my_xticks[::20000], rotation=25)
#ax.xaxis.set_major_locator(ticker.AutoLocator())
#
#plt.plot(Rx, Ry, label='Reynolds Creek', color='orange')
#plt.plot(Sx, Sy, label='Senator Beck', color='navy')
#ax.legend()
#plt.title('Hourly Average Specific Humidity (g/m3)')
#plt.xlabel('Date: 1998 to 2008')
#plt.ylabel('Specific Humidity (g/m3)')
#plt.savefig('Humidity.png')
#plt.show()
#%%  Plotting figures Temperature
#Syt = np.array(TempS)
#Ryt = np.array(TempR)
#
#figt, axt = plt.subplots(1,1)
#plt.xticks(x, my_xticks[::20000], rotation=25)
#axt.xaxis.set_major_locator(ticker.AutoLocator())
#
#plt.plot(Rx, Ryt, label='Reynolds Creek', color='red')
#plt.plot(Sx, Syt, label='Senator Beck', color='aqua')
#axt.legend()
#plt.title('Hourly Average Temperature (C))')
#plt.xlabel('Date: 1998 to 2008')
#plt.ylabel('Temperature (C)')
#plt.savefig('Temperature.png')
#plt.show()
#%%
#Sys = np.array(swrs2)
#Rys = np.array(swrr)
#
#figs, axs = plt.subplots(1,1)
#plt.xticks(x, my_xticks[::20000], rotation=25)
#axs.xaxis.set_major_locator(ticker.AutoLocator())
#
#plt.plot(Rx, Rys, label='Reynolds Creek', color='green')
#plt.plot(Sx, Sys, label='Senator Beck', color='purple')
#axs.legend()
#plt.title('Hourly Average Short Wave Radiation)')
#plt.xlabel('Date: 1998 to 2008')
#plt.ylabel('Short Wave Radiation')
#plt.savefig('Short Wave Radiation.png')
#plt.show()
#%%
#Syl = np.array(lwrs2)
#Ryl = np.array(lwrr)
#
#figl, axl = plt.subplots(1,1)
#plt.xticks(x, my_xticks[::20000], rotation=25)
#axl.xaxis.set_major_locator(ticker.AutoLocator())
#
#plt.plot(Rx, Ryl, label='Reynolds Creek', color='gray')
#plt.plot(Sx, Syl, label='Senator Beck', color='lavender')
#axl.legend()
#plt.title('Hourly Average Long Wave Radiation)')
#plt.xlabel('Date: 1998 to 2008')
#plt.ylabel('Long Wave Radiation')
#plt.savefig('Long Wave Radiation.png')
#plt.show()
#%%
#Syp = np.array(precips)
#Ryp = np.array(precipr)
#
#figp, axp = plt.subplots(1,1)
#plt.xticks(x, my_xticks[::20000], rotation=25)
#axp.xaxis.set_major_locator(ticker.AutoLocator())
#
#plt.plot(Rx, Ryp, label='Reynolds Creek', color='skyblue')
#plt.plot(Sx, Syp, label='Senator Beck', color='cyan')
#axp.legend()
#plt.title('Hourly Average Precipitation (mm)')
#plt.xlabel('Date: 1998 to 2008')
#plt.ylabel('Precipitation (mm)')
#plt.savefig('Precipitation.png')
#plt.show()
#%%
#Syw = np.array(winds)
#Ryw = np.array(windr)
#
#figw, axw = plt.subplots(1,1)
#plt.xticks(x, my_xticks[::20000], rotation=25)
#axw.xaxis.set_major_locator(ticker.AutoLocator())
#
#plt.plot(Rx, Ryw, label='Reynolds Creek', color='olive')
#plt.plot(Sx, Syw, label='Senator Beck', color='pink')
#axw.legend()
#plt.title('Hourly Average Wind Speed (m/s)')
#plt.xlabel('Date: 1998 to 2008')
#plt.ylabel('Wind Speed (m/s)')
#plt.savefig('WindSpeed.png')
#plt.show()
#%%
#finR2 = netcdf.netcdf_file('forcing_above_aspenPP.nc', 'r')
#for i in finR2.variables:
#    print(i)
#    
#finR3 = netcdf.netcdf_file('forcing_reynolds_distributed.nc', 'r')
#for j in finR3.variables:
#    print(j)
    
#finR4 = netcdf.netcdf_file('forcing_sheltered.nc', 'r')
#for k in finR4.variables:
#    print(k)

#finR5 = netcdf.netcdf_file('ReynoldsCreek_eddyFlux.nc', 'r')
#for a in finR5.variables:
#    print(a)
#
#finR6 = netcdf.netcdf_file('ReynoldsCreek_valData.nc', 'r')
#for b in finR6.variables:
#    print(b)

#zssheltered = finR6.variables['zs_sheltered']
#print zssheltered
#finS2 = netcdf.netcdf_file('senatorBeck_SASP_1hr.nc', 'r')
#for c in finS2.variables:
#    print(c)
#%%








































