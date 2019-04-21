# anaconda prompt
# conda install netcdf4 
from scipy.io import netcdf
import numpy as np
#import pandas as pd
import datetime
from netCDF4 import Dataset
from netCDF4 import netcdftime
from netCDF4 import num2date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#%%
#import cdms 
#f = cdms.open('forcing_above_aspenPP.nc')
#import netCDF4 
#%%
fin = netcdf.netcdf_file('SenatorBeck_forcing.nc', 'r')
for v in fin.variables:
    print(v)

print fin.variables['pptrate'].units
SpecHumidity=1292*fin.variables['spechum'][:]
LWRad=fin.variables['LWRadAtm'][:]
SWRad=fin.variables['SWRadAtm'][:]
Temp=272-fin.variables['airtemp'][:]
Precip=3600*fin.variables['pptrate'][:]
Windspd=fin.variables['windspd'][:]

Time = fin.variables['time'][:]
plt.plot(Time, SpecHumidity)
#%% Number to Date
#file_in = Dataset("forcing_above_aspen.nc","r",format="NETCDF4")
#tname = "time"
#nctime = file_in.variables[tname][:] # get values
#t_unit = file_in.variables[tname].units # get unit  "days since 1950-01-01T00:00:00Z"
#t_cal = file_in.variables[tname].calendar
#tvalue = num2date(nctime,units = t_unit,calendar = t_cal)
#str_time = [i.strftime("%Y-%m-%d %H:%M") for i in tvalue] # to display dates as string
#
#
#
## Plotting figures
#
#hfmt = mdates.DateFormatter('%Y-%m-%d')
#fig1 = plt.figure()
#
##ax1 = fig1.add_subplot(111)
##ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=10))
##ax1.xaxis.set_major_formatter(hfmt)
##plt.plot(str_time, Time, label='Humidity', color='cyan')
#width = np.diff(str_time).min()
#fig1.autofmt_xdate()
#plt.title('Specific Humidity')
#plt.xticks(str_time, width = width, labels='Date', rotation='vertical')
##plt.xlabel('Date')
#plt.ylabel('Humidity')
#plt.savefig('Humidity.png')
##plt.close()
#plt.show()


























