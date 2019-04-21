## Reading Data
from netCDF4 import Dataset   # http://unidata.github.io/netcdf4-python/
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from netCDF4 import num2date  #, date2num
#%%
finrc = Dataset ('met_insitu_swa_2005_2015.nc', format="NETCDF4")
#print finrc.file_format
#print finrc.dimensions.keys()
print finrc.variables.keys()
times = finrc.variables['time'] # hour
swrdown = finrc.variables['SWdown'] #Surface downward shortwave radiation = W/m2
lvrdown = finrc.variables['LWdown'] #Surface downward longwave radiation = W/m2
rain = finrc.variables['Rainf'] #Rainfall rate = kg/m2/s
snow = finrc.variables['Snowf'] #Snowfall rate = kg/m2/s
sp = finrc.variables['Psurf']  #Surface Pressure = Pa
tair = finrc.variables['Tair'] #Near-Surface air temp = K
wind = finrc.variables['Wind'] #near-surface wind speed = m/s
nssh = finrc.variables['Qair'] #near-surface specific humidity = kg/kg
#%% Defining Variables
time = finrc.variables['time'][:]
swrdown = finrc.variables['SWdown'][:]
lvrdown = finrc.variables['LWdown'][:]
rain = finrc.variables['Rainf'][:]
snow = finrc.variables['Snowf'][:]
sp = finrc.variables['Psurf'][:]
tair = finrc.variables['Tair'][:]
wind = finrc.variables['Wind'][:]
nssh = finrc.variables['Qair'][:]
#%% Plotting figures Shortwave Radiation
## Number to Date 
#Reynolds Creek
t_unit = finrc.variables['time'].units # get unit  "days since 1950-01-01T00:00:00Z"
try :
    t_cal = finrc.variables['time'].calendar

except AttributeError : # Attribute doesn't exist
    t_cal = u"gregorian" # or standard

tvalue = num2date(time, units=t_unit, calendar=t_cal)
date = [i.strftime("%Y-%m-%d %H:%M") for i in tvalue] # to display dates as string

#x = np.arange(0,np.size(time))
#y = np.array(swrdown)
#my_xticks = date
#fig, ax = plt.subplots(1,1)
#plt.xticks(x, my_xticks[::12000], rotation=20)
#ax.xaxis.set_major_locator(ticker.AutoLocator())
#ax.legend()
#plt.plot(x, y, label='Shortwave radiation [W/m2]', color='salmon')
#plt.title('Hourly Average Shortwave Radiation for Renolds Creek')
#plt.xlabel('Date, hourly, 1988 to 2008')
#plt.ylabel('Shortwave Radiation [W/m2]')
#plt.show()
#%%  Senator Beck
finsb = Dataset ('met_insitu_snb_2005_2015.nc', format="NETCDF4")
timesb = finsb.variables['time'][:]
swrdown_sb = finsb.variables['SWdown'] #Surface downward shortwave radiation = W/m2

t_unitsb = finsb.variables['time'].units # get unit  "days since 1950-01-01T00:00:00Z"
try :
    t_calsb = finsb.variables['time'].calendar

except AttributeError : # Attribute doesn't exist
    t_calsb = u"gregorian" # or standard

tvaluesb = num2date(timesb, units=t_unitsb, calendar=t_calsb)
datesb = [i.strftime("%Y-%m-%d %H:%M") for i in tvaluesb] # to display dates as string
#%%  Swamp Angle
finsa = Dataset ('met_insitu_swa_2005_2015.nc', format="NETCDF4")
timesa = finsa.variables['time'][:]
swrdown_sa = finsa.variables['SWdown'] #Surface downward shortwave radiation = W/m2

t_unitsa = finsa.variables['time'].units # get unit  "days since 1950-01-01T00:00:00Z"
try :
    t_calsa = finsa.variables['time'].calendar

except AttributeError : # Attribute doesn't exist
    t_calsa = u"gregorian" # or standard

tvaluesa = num2date(timesa, units=t_unitsa, calendar=t_calsa)
datesa = [i.strftime("%Y-%m-%d %H:%M") for i in tvaluesa] # to display dates as string
#%%  Plotting 3 series in one figure
xrc = np.array(time)
yrc = np.array(swrdown)

xsb = np.array(timesb)
ysb = np.array(swrdown_sb)

xsa = np.array(timesa)
ysa = np.array(swrdown_sa)

datet = list(set(date + datesb + datesa)) #Combine lists and remove duplicates python
datet.sort()
x = np.arange(0,np.size(datet))
my_xticks = datet
fig, ax = plt.subplots(1,1)
plt.xticks(x, my_xticks[::40000], rotation=25)
ax.xaxis.set_major_locator(ticker.AutoLocator())

plt.plot(xrc, yrc, label='Reynolds Creek', color='orchid')
plt.plot(xsb, ysb, label='Senator Beck', color='teal')
plt.plot(xsa, ysa, label='Swamp Angle', color='yellowgreen')
ax.legend()
plt.title('Hourly Average Shortwave Radiation [w/m2]')
plt.xlabel('Date: 1988 to 2015')
plt.ylabel('SWR [w/m2]')
plt.savefig('Shortwave.png')
plt.show()






























