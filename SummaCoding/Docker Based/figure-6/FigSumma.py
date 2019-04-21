# anaconda prompt
# conda install netcdf4 
from scipy.io import netcdf
import numpy as np
from netCDF4 import num2date
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#%% Getting Data
## Renold Creek
finrc_cr = netcdf.netcdf_file('albedoTest_2005-2006_reynoldsConstantDecayRate_1.nc', 'r')
#print finrc.variables.keys()
snowdepth_rci_cr = finrc_cr.variables['scalarSWE'][:] # kg/m2 = 0.001 m.H2O
snowdepth_rc_cr = 0.005*snowdepth_rci_cr              # rho_snow = 0.2 ; 0.005 m.snow

timerc_cr = finrc_cr.variables['time'][:] # Second
t_unitrc_cr = finrc_cr.variables['time'].units # get unit  
try :
    t_calrc_cr = finrc_cr.variables['time'].calendar

except AttributeError : # Attribute doesn't exist
    t_calrc_cr = u"gregorian" # or standard

tvaluerc_cr = num2date(timerc_cr, units=t_unitrc_cr, calendar=t_calrc_cr)
daterc_cr = [i.strftime("%Y-%m-%d %H:%M") for i in tvaluerc_cr] # to display dates as string
#%% 
finrc_vr = netcdf.netcdf_file('albedoTest_2005-2006_reynoldsVariableDecayRate_1.nc', 'r')
#print finrc.variables.keys()
snowdepth_rci_vr = finrc_vr.variables['scalarSWE'][:] # kg/m2 = 0.001 m.H2O
snowdepth_rc_vr = 0.005*snowdepth_rci_vr              # rho_snow = 0.2 ; 0.005 m.snow

timerc_vr = finrc_vr.variables['time'][:] # Second
t_unitrc_vr = finrc_vr.variables['time'].units # get unit  
try :
    t_calrc_vr = finrc_vr.variables['time'].calendar

except AttributeError : # Attribute doesn't exist
    t_calrc_vr = u"gregorian" # or standard

tvaluerc_vr = num2date(timerc_vr, units=t_unitrc_vr, calendar=t_calrc_vr)
daterc_vr = [i.strftime("%Y-%m-%d %H:%M") for i in tvaluerc_vr] # to display dates as string
#%%
finrc_ob = netcdf.netcdf_file('ReynoldsCreek_valData.nc', 'r')
print finrc_ob.variables.keys()
print finrc_ob.dimensions.keys()

snowdepth_rc_ob = finrc_ob.variables['zs_sheltered'][:]
snowdepth_rc_obm = snowdepth_rc_ob[52608: 61345]

timerc_ob = finrc_ob.variables['time'][:] # Second
timerc_obm = timerc_ob[52608: 61345]

t_unitrc_ob = finrc_ob.variables['time'].units
try :
    t_calrc_ob = finrc_ob.variables['time'].calendar

except AttributeError : # Attribute doesn't exist
    t_calrc_ob = u"gregorian" # or standard

tvaluerc_ob = num2date(timerc_ob, units=t_unitrc_ob, calendar=t_calrc_ob)
daterc_ob = [i.strftime("%Y-%m-%d %H:%M") for i in tvaluerc_ob] # to display dates as string
daterc_obm = daterc_ob[52608: 61345]
#%%
xrc_cr = np.array(timerc_cr)
yrc_cr = np.array(snowdepth_rc_cr)

xrc_vr = np.array(timerc_vr)
yrc_vr = np.array(snowdepth_rc_vr)

xrc_ob = np.array(timerc_obm)
yrc_ob = np.array(snowdepth_rc_obm)*0.02

datet_rc = list(set(daterc_cr + daterc_vr)) #Combine lists and remove duplicates python
datet_rc.sort()
xrc = np.arange(0,np.size(datet_rc))
my_xticksrc = datet_rc
figrc, axrc = plt.subplots(1,1)
plt.xticks(xrc, my_xticksrc[::1000], rotation=25)
axrc.xaxis.set_major_locator(ticker.AutoLocator())

plt.plot(xrc_vr, yrc_vr, label='Variable Abledo Decay', color='navy')
plt.plot(xrc_cr, yrc_cr, label='Constant Abledo Decay', color='red')
axrc.fill(xrc_cr, yrc_ob, label='Observation Data', color='lightgray')

axrc.legend(loc=2)
plt.title('SWE-Renolds Mountain East')
plt.xlabel('Date: 2005 to 2006')
plt.ylabel('Snow Depth (m)')
plt.savefig('SnowDepth_RC.png')
plt.show()
#%%***********************************Senator Beck************************************************************


finsb_cr = netcdf.netcdf_file('albedoTest_2010-2011_senatorConstantDecayRate_1.nc', 'r')
#print finrc.variables.keys()
snowdepth_sbi_cr = finsb_cr.variables['scalarSWE'][:] # kg/m2 = 0.001 m.H2O
snowdepth_sb_cr = 0.005*snowdepth_sbi_cr              # rho_snow = 0.2 ; 0.005 m.snow

timesb_cr = finsb_cr.variables['time'][:] # Second
t_unitsb_cr = finsb_cr.variables['time'].units # get unit  
try :
    t_calsb_cr = finsb_cr.variables['time'].calendar

except AttributeError : # Attribute doesn't exist
    t_calsb_cr = u"gregorian" # or standard

tvaluesb_cr = num2date(timesb_cr, units=t_unitsb_cr, calendar=t_calsb_cr)
datesb_cr = [i.strftime("%Y-%m-%d %H:%M") for i in tvaluesb_cr] # to display dates as string
#%% 
finsb_vr = netcdf.netcdf_file('albedoTest_2010-2011_senatorVariableDecayRate_1.nc', 'r')
#print finrc.variables.keys()
snowdepth_sbi_vr = finsb_vr.variables['scalarSWE'][:] # kg/m2 = 0.001 m.H2O
snowdepth_sb_vr = 0.005*snowdepth_sbi_vr              # rho_snow = 0.2 ; 0.005 m.snow

timesb_vr = finsb_vr.variables['time'][:] # Second
t_unitsb_vr = finsb_vr.variables['time'].units # get unit  
try :
    t_calsb_vr = finsb_vr.variables['time'].calendar

except AttributeError : # Attribute doesn't exist
    t_calsb_vr = u"gregorian" # or standard

tvaluesb_vr = num2date(timesb_vr, units=t_unitsb_vr, calendar=t_calsb_vr)
datesb_vr = [i.strftime("%Y-%m-%d %H:%M") for i in tvaluesb_vr] # to display dates as string
#%%
finsb_ob = netcdf.netcdf_file('senatorBeck_SASP_1hr.nc', 'r')
print finsb_ob.variables.keys()
print finsb_ob.dimensions.keys()

snowdepth_sb_ob = finsb_ob.variables['snowDepth'][:]
snowdepth_sb_ob2=[]
for k in range (np.size(snowdepth_sb_ob)):
    if snowdepth_sb_ob[k]==-9999:
        hhs=0
        snowdepth_sb_ob2.append(hhs)
    else:
        hhs=snowdepth_sb_ob[k]
        snowdepth_sb_ob2.append(hhs)
snowdepth_sb_obm = snowdepth_sb_ob2[60442: 69179]

timesb_ob = finsb_ob.variables['time'][:] # Second
timesb_obm = timesb_ob[60442: 69179]

t_unitsb_ob = finsb_ob.variables['time'].units
try :
    t_calsb_ob = finsb_ob.variables['time'].calendar

except AttributeError : # Attribute doesn't exist
    t_calsb_ob = u"gregorian" # or standard

tvaluesb_ob = num2date(timesb_ob, units=t_unitsb_ob, calendar=t_calsb_ob)
datesb_ob = [i.strftime("%Y-%m-%d %H:%M") for i in tvaluesb_ob] # to display dates as string
datesb_obm = datesb_ob[60442: 69179]
#%%
xsb_cr = np.array(timesb_cr)
ysb_cr = np.array(snowdepth_sb_cr)

xsb_vr = np.array(timesb_vr)
ysb_vr = np.array(snowdepth_sb_vr)

xsb_ob = np.array(timesb_obm)
ysb_ob = np.array(snowdepth_sb_obm)

datet_sb = list(set(datesb_cr + datesb_vr)) #Combine lists and remove duplicates python
datet_sb.sort()
xsb = np.arange(0,np.size(datet_sb))
my_xtickssb = datet_sb
figsb, axsb = plt.subplots(1,1)
plt.xticks(xsb, my_xtickssb[::1000], rotation=25)
axsb.xaxis.set_major_locator(ticker.AutoLocator())

plt.plot(xsb_vr, ysb_vr, label='Variable Abledo Decay', color='teal')
plt.plot(xsb_cr, ysb_cr, label='Constant Abledo Decay', color='violet')
axsb.fill(xsb_cr, ysb_ob, label='Observation Data', color='lightgray')

axsb.legend(loc=2)
plt.title('SWE-Senator Creek')
plt.xlabel('Date: 2010-2011')
plt.ylabel('Snow Depth (m)')
plt.savefig('SnowDepth_SB.png')
plt.show()
#%%
from netCDF4 import Dataset
fin = Dataset('summa_zParamTrial_variableDecayRate.nc')
print fin.variables.keys()
albedo1 = fin.variables['albedoDecayRate'][:] 
#plt.plot(snowdepth1)

























