###       /bin/bash runTestCases_docker.sh
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
finS = Dataset("albedoTest_2010-2011_senatorVariableDecayRate_1.nc","r",format="NETCDF4")
for v in finS.variables:
    print(v)

a1 = finS.variables['scalarSWE'][:]
TimeS = finS.variables['time'][:]
Sy = np.array(a1)
plt.plot(TimeS, Sy, color='red')

finS1 = Dataset("albedoTest_2009-2010_senatorVariableDecayRate_1.nc","r",format="NETCDF4")
a2 = finS1.variables['scalarSWE'][:]
TimeS1 = finS1.variables['time'][:]
Sy1 = np.array(a2)
plt.plot(TimeS1, Sy1, color='blue')
#%%

finS2 = Dataset("summa_zParamTrial_variableDecayRate.nc","r",format="NETCDF4")
for m in finS2.variables:
    print(m)
#a3 = finS2.variables['gruId'][:]
#print a3
#%%
finS2 = Dataset("t1paramtrial_6p.nc","r",format="NETCDF4")
for i in finS2.variables:
    print(i)
a2 = finS2.variables['hruIndex'][:]
print a2   































