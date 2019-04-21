###       /bin/bash runTestCases_docker.sh
# %matplotlib inline
import numpy as np
from netCDF4 import Dataset
import itertools
#%%
#albedoMinWinter           |       0.5500 |       0.6000 |       1.0000 $$$$$$ 2: 0.65 | 0.90
#albedoMinSpring           |       0.5500 |       0.3000 |       1.0000 $$$$$$ 2: 0.40 | 0.90
#albedoMaxVisible          |       0.9500 |       0.7000 |       0.9500 ****** 1: 0.95
#albedoMinVisible          |       0.7500 |       0.5000 |       0.7500 ****** 1: 0.75
#albedoMaxNearIR           |       0.6500 |       0.5000 |       0.7500 $$$$$$ 2: 0.55 | 0.70
#albedoMinNearIR           |       0.3000 |       0.1500 |       0.4500 $$$$$$ 2: 0.20 | 0.43

#p1 = [200000, 360000, 1000000] #albedoDecayRate_ConsD  |       1.0d+6 |       0.1d+6 |       5.0d+6 $$$$$$ 3: 0.8d+6 | 2d+6 | 4.0d+6
#p1 = [500000, 1000000, 4000000] #albedoDecayRate_VarD  |       1.0d+6 |       0.1d+6 |       5.0d+6 $$$$$$ 3: 0.8d+6 | 2d+6 | 4.0d+6
p1 = [0.75, 0.84, 0.91] #albedoMax_ConsDR_VarDR      |       0.8400 |       0.7000 |       0.9500 $$$$$$ 2: 0.75 | 0.92
p2 = [0.45, 0.55, 0.8] #albedoMinWinter   |       0.5500 |       0.6000 |       1.0000 $$$$$$ 2: 0.65 | 0.90
#p3 = [0.4, 0.7] #albedoMinSpring    |       0.5500 |       0.3000 |       1.0000 $$$$$$ 2: 0.40 | 0.90
#p4 = [0.55, 0.71] #albedoMaxNearIR   |       0.6500 |       0.5000 |       0.7500 $$$$$$ 2: 0.55 | 0.70
#p5 = [0.25, 0.4] #albedoMinNearIR  |       0.3000 |       0.1500 |       0.4500 $$$$$$ 2: 0.20 | 0.43

hruidxID = [1001, 1002, 1003, 1004, 1005, 1006]
hru_num = 6

# %store hruidxID
#%%
param_nam_list = ['albedoMax', 'albedoMinWinter'] 

#%% #create new paramtrail.nc file for constantDecayRate model
paramfile_c = Dataset("summa_zParamTrial_constantDecayRate_SA.nc",'w',format='NETCDF3_CLASSIC') #create new paramtrail.nc file
hru = paramfile_c.createDimension('hru', None)
hidx = paramfile_c.createVariable('hruIndex', np.float64,('hru',)) # add hruIndex variable

for param in param_nam_list:
    paramfile_c.createVariable(param, np.float64,('hru',))

constant_params = ['albedoDecayRate', 'frozenPrecipMultip', 'rootingDepth', 'rootDistExp', 'theta_sat','theta_res','vGn_alpha','vGn_n','k_soil','critSoilWilting','critSoilTranspire','winterSAI','summerLAI','heightCanopyTop','heightCanopyBottom']
for params in constant_params:
    paramfile_c.createVariable(params, np.float64,('hru',))
    
#%% # add values for the constant variables in HRUs
pt_c = Dataset('summa_zParamTrial_constantDecayRate.nc')

for varname in pt_c.variables.keys():
    var = pt_c.variables[varname][0]
    c = np.full((hru_num,),var)
    try :
        paramfile_c.variables[varname][:]=c
    except IndexError: # size of data array does not conform to slice
        pass
    
#%% # add values for the changing variables in HRUs
for var in param_nam_list:
    paramfile.variables[var][:]=p1

paramfile.variables['hruIndex'][:]=hruidxID

for varname in paramfile.variables.keys():
    var = paramfile.variables[varname]
    print varname, var.dtype, var.dimensions, var.shape
print paramfile.variables['albedoDecayRate'][:]
paramfile.close()

#%% #create new paramtrail.nc file for variableDecayRate model
paramfile_v = Dataset("summa_zParamTrial_variableDecayRate_SA.nc",'w',format='NETCDF3_CLASSIC') #create new paramtrail.nc file
hru = paramfile_v.createDimension('hru', None)
hidx = paramfile_v.createVariable('hruIndex', np.float64,('hru',)) # add hruIndex variable

for param in param_nam_list:
    paramfile_v.createVariable(param, np.float64,('hru',))

constant_params = ['albedoDecayRate', 'frozenPrecipMultip', 'rootingDepth', 'rootDistExp', 'theta_sat','theta_res','vGn_alpha','vGn_n','k_soil','critSoilWilting','critSoilTranspire','winterSAI','summerLAI','heightCanopyTop','heightCanopyBottom']
for params in constant_params:
    paramfile_v.createVariable(params, np.float64,('hru',))
    

#%%
iccheck = Dataset("summa_zParamTrial_constantDecayRate_AM.nc")
print iccheck.variables['albedoMax'][:]
#%%

    
    
    