# Before we open anything to edit we want to get all of our forcing data in one place
#Two different sources for Saghen Data:
#1.the main met tower at the first site in Sagehen
#2.NLDAS forcing data for the grid cell containing Sagehen
#This is good example-wise as it covers two different input types.
#I've included a link to where I got the NLDAS data in the google doc as well as a quick description of how I downloaded it.
from netCDF4 import Dataset,netcdftime,num2date
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#%%
#NLDAS data
# I got precip, longwave, shortwave, and specific humidity from NLDAS
# the NLDAS data come as netCDFs
n_p = Dataset('SH_forcing/NLDAS_FORA0125_H_APCPsfc.nc')
n_l = Dataset('SH_forcing/NLDAS_FORA0125_H_DLWRFsfc.nc')
n_s = Dataset('SH_forcing/NLDAS_FORA0125_H_DSWRFsfc.nc')
n_h = Dataset('SH_forcing/NLDAS_FORA0125_H_SPFH2m.nc')

# output info in various ways, comment out all but one to see each one
n_p.variables.keys()
n_p.variables
n_p.variables['APCPsfc']
#%%
# NLDAS time is in seconds since 1970, you want your time reference to be consistent across all input data
# because test case data is in days since 1990, I did some squirrelly code to get that here
# the other (probably more efficient) workaround to this would be to edit the units of the time variable 
# the important part is that the unit specified and all of the data match

# go to datetime and then back to time since reference
time_n = n_l.variables['time'][:]
t_unit= n_l.variables['time'].units # get unit  "seconds since 1970-01-01 00:00:00Z"

try:
    t_cal = n_l.variables['time'].calendar
except AttributeError: # Attribute doesn't exist
    t_cal = u"gregorian" # or standard

tvalue = num2date(time_n,units = t_unit,calendar = t_cal)
str_time = [i.strftime("%Y-%m-%d %H:%M") for i in tvalue] 

date_time = [datetime.strptime(i,'%Y-%m-%d %H:%M') for i in str_time]

nldas_index = pd.Index(date_time)
# time since 1990-01-01 in seconds
ix_n = (nldas_index-datetime(1990,1,1)).total_seconds()
# convert to days
nldas_ix =ix_n/86400
#%%
# shape of the input array is also important
# NLDAS inputs are 3-dimensional (time, lon, lat)
# SUMMA is expecting 2 dimensions (time, hru)

# get values for all variables
n_pre = n_p.variables['APCPsfc'][:]
n_lw = n_l.variables['DLWRFsfc'][:]
n_sw = n_s.variables['DSWRFsfc'][:]
n_hum = n_h.variables['SPFH2m'][:]

# reshape arrays to 2 dimensions 
# this works fine for our 1-d models but would be more complicated for 2-d (matching lats and lons to HRU)
nl_ppt = np.reshape(n_pre,(87696,1))
nl_lwr = np.reshape(n_lw,(87696,1))
nl_swr = np.reshape(n_sw,(87696,1))
nl_sph = np.reshape(n_hum,(87696,1))

# this is optional but I prefer plotting and slicing with dataframes
nldas_ppt = pd.DataFrame(nl_ppt, index=nldas_index, columns=['p'])
nldas_lwr = pd.DataFrame(nl_lwr, index=nldas_index, columns=['lw'])
nldas_swr = pd.DataFrame(nl_swr, index=nldas_index, columns=['sw'])
nldas_sph = pd.DataFrame(nl_sph, index=nldas_index, columns=['sp_hm'])

import pytz
index = nldas_index
pac = pytz.timezone('US/Pacific')
nldas_ppt.index = index.tz_localize(pytz.utc).tz_convert(pac)
nldas_lwr.index = index.tz_localize(pytz.utc).tz_convert(pac)
nldas_swr.index = index.tz_localize(pytz.utc).tz_convert(pac)
nldas_sph.index = index.tz_localize(pytz.utc).tz_convert(pac)

pac_ix = nldas_sph.index

# time since 1990-01-01 in seconds
ix_n = (pac_ix-pac.localize(datetime(1990,1,1))).total_seconds()
# convert to days
nldas_ix =ix_n/86400

# this is optional but I prefer plotting and slicing with dataframes
nldas_ppt = pd.DataFrame(nl_ppt, index=nldas_ix, columns=['p'])
nldas_lwr = pd.DataFrame(nl_lwr, index=nldas_ix, columns=['lw'])
nldas_swr = pd.DataFrame(nl_swr, index=nldas_ix, columns=['sw'])
nldas_sph = pd.DataFrame(nl_sph, index=nldas_ix, columns=['sp_hm'])
#%%
# I got air pressure, temperature, and wind speed from Sagehen
# I call the data in from a csv to a pandas dataframe
sh_dat = pd.read_csv("SH_forcing/sage_2009_present.csv", delimiter=',',header=None,index_col=0, skiprows=1,na_values = "NAN",\
                  keep_default_na= False, parse_dates=[0])
# trim the data to times where all sensors have data
sh_dat = sh_dat["2010-10-29 00:00:00":"2017-09-15 00:00:00"]

# resample to hourly 
hr = sh_dat.resample('H').mean()

# get time in days since 1990-01-01 to match NLDAS data
t = hr.index
ix = (t-datetime(1990,1,1)).total_seconds()
# convert to days
ix =ix/86400
# array of values to replace those in the forcing file
new_ix = ix.values

# other data from the sagehen dataframe
# for whatever reason windspeed doesn't want to cooperate so I do it separately
ws_pd = pd.to_numeric(sh_dat[6])
sh_ws = ws_pd.resample('H').mean()
swr = pd.to_numeric(sh_dat[4])
sh_sw = swr.resample('H').mean()
sh_sw = sh_sw*1000
t = hr[1]+273.15
print(len(t))
# I use the average air pressure to make a list of constant air pressre (next cell)
ap = hr[2]*1000
ap.mean()
# make an array the same size and shape as the rest of the data and fill with air pressure
airp = np.full((60313, 1), 826560.34542)

# make those arrays and reshape to model input shape
t_a = t.values
airt = np.reshape(t_a,(60313,1))
ws_a = sh_ws.values
wndsp = np.reshape(ws_a,(60313,1))
sw_a = sh_sw.values
swr = np.reshape(sw_a,(60313,1))
#%%
# I downladed my NLDAS data before seeing what was available from Sagehen 
# so I want all the data to be for the same time period
# to figure out the range of values of the sagehen data, I print the index I made for it
print(new_ix)
# it is easy to slice the NLDAS dataframes to match the sagehen data
nldas_ppt = nldas_ppt[7606:10119]
nldas_lwr = nldas_lwr[7606:10119]
nldas_swr = nldas_swr[7606:10119]
nldas_sph = nldas_sph[7606:10119]

# these also need to be arrays
ppt_for_n = nldas_ppt.values
lwr_for_n = nldas_lwr.values
swr_for_n = nldas_swr.values
sph_for_n = nldas_sph.values

# and precip needs to be converted to a rate
# it is currently kg/m^2 and needs to be kg/(m^2s)
# so, divide by the seconds in an hour
ppt_for_n = ppt_for_n/3600

# I had to trim some data, this only applies if you have to do the same
# likely better to have gap-filled data input into this code
t[t < 250] = np.NaN
t_a = t.values
sh_t = np.reshape(t_a,(60313,1))
plt.plot(new_ix,sh_t)
plt.show()
#%%

























