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

#p2 = [0.45, 0.6] #albedoMinWinter   |       0.5500 |       0.6000 |       1.0000 $$$$$$ 2: 0.65 | 0.90
#p3 = [0.4, 0.7] #albedoMinSpring    |       0.5500 |       0.3000 |       1.0000 $$$$$$ 2: 0.40 | 0.90
#p4 = [0.55, 0.71] #albedoMaxNearIR   |       0.6500 |       0.5000 |       0.7500 $$$$$$ 2: 0.55 | 0.70
#p5 = [0.25, 0.4] #albedoMinNearIR  |       0.3000 |       0.1500 |       0.4500 $$$$$$ 2: 0.20 | 0.43

hruidxID = [1001, 1002, 1003]
hru_num = 3

# %store hruidxID
#%%
param_nam_list = ['albedoMax'] 

#%% #create new paramtrail.nc file and adding vaiables to it
paramfile = Dataset("summa_zParamTrial_constantDecayRate_am.nc",'w',format='NETCDF3_CLASSIC') #create new paramtrail.nc file
hru = paramfile.createDimension('hru', None)
hidx = paramfile.createVariable('hruIndex', np.float64,('hru',)) # add hruIndex variable

for param in param_nam_list:
    paramfile.createVariable(param, np.float64,('hru',))

constant_params = ['albedoDecayRate', 'frozenPrecipMultip', 'rootingDepth', 'rootDistExp', 'theta_sat','theta_res','vGn_alpha','vGn_n','k_soil','critSoilWilting','critSoilTranspire','winterSAI','summerLAI','heightCanopyTop','heightCanopyBottom']
for params in constant_params:
    paramfile.createVariable(params, np.float64,('hru',))

#%% # add values for the constant variables in HRUs
pt = Dataset('summa_zParamTrial_constantDecayRate.nc')

for varname in pt.variables.keys():
    var = pt.variables[varname][0]
    c = np.full((hru_num,),var)
    try :
        paramfile.variables[varname][:]=c
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
#%% #Local attributes and initial conditions
la = Dataset('summa_zLocalAttributes_senatorSheltered.nc') #('settings/wrrPaperTestCases/figure07/summa_zLocalAttributes_riparianAspen.nc')
ic = Dataset('summa_zInitialCond.nc') #('settings/wrrPaperTestCases/figure07/summa_zInitialCond.nc')
#%% # local attributes file
# create a new localAtribute file
local_atrbt = Dataset("summa_zLocalAttributes_senatorSheltered_ADR.nc",'w',format='NETCDF3_CLASSIC')
# define dimensions 
hru = local_atrbt.createDimension('hru', hru_num) 
time = local_atrbt.createDimension('gru', 1)
# define variables
h2gid = local_atrbt.createVariable('hru2gruId', np.int32,('hru',))
dhruindx = local_atrbt.createVariable('downHRUindex', np.int32,('hru',))
slopeindx = local_atrbt.createVariable('slopeTypeIndex', np.int32,('hru',))
soilindx = local_atrbt.createVariable('soilTypeIndex', np.int32,('hru',))
vegindx = local_atrbt.createVariable('vegTypeIndex', np.int32,('hru',))
mh = local_atrbt.createVariable('mHeight', np.float64,('hru',))
cl = local_atrbt.createVariable('contourLength', np.float64,('hru',))
tanslope = local_atrbt.createVariable('tan_slope', np.float64,('hru',))
elev = local_atrbt.createVariable('elevation', np.float64,('hru',))
lon = local_atrbt.createVariable('longitude', np.float64,('hru',))
lat = local_atrbt.createVariable('latitude', np.float64,('hru',))
hruarea = local_atrbt.createVariable('HRUarea', np.float64,('hru',))
hruid = local_atrbt.createVariable('hruId', np.int32,('hru',))
gruid = local_atrbt.createVariable('gruId', np.int32,('gru',))
# give variables units
mh.units = 'm'
cl.units = 'm'
tanslope.units = 'm m-1'
elev.units = 'm'
lat.units = 'decimal degree north'
lon.units = 'decimal degree east'
hruarea.units = 'm^2'
#%% # add values for the constant variables in HRUs
for varname in la.variables.keys():
    var = la.variables[varname][0]
    c2 = np.full((hru_num,),var)
    try :
        local_atrbt.variables[varname][:]=c2
    except IndexError: # size of data array does not conform to slice
        pass
    
#%% # get the hru, gru, and hru2gru in local_atribute file
newgru = np.array([111])
local_atrbt.variables['gruId'][:] = newgru

c3 = np.repeat(newgru[:,np.newaxis], hru_num, axis=1); newlad = c3.reshape(hru_num,)
local_atrbt.variables['hru2gruId'][:] = c3

local_atrbt.variables['hruId'][:] = hruidxID

local_atrbt.close()
#%%
laCheck = Dataset('summa_zLocalAttributes_senatorSheltered_ADR.nc')

print laCheck.variables['longitude'][:]
for j in laCheck.variables:
    print j

laCheck.close()
#%% # initial conditions file. 

in_condi = Dataset("summa_zInitialCond_ADR.nc",'w',format='NETCDF3_CLASSIC')

# define dimensions 
midtoto = in_condi.createDimension('midToto',8)
midsoil = in_condi.createDimension('midSoil',8)
idctoto = in_condi.createDimension('ifcToto',9)
scalarv = in_condi.createDimension('scalarv', 1)
# this is the number you will change to the number of HRU's from your param trial file
hrud = in_condi.createDimension('hru', hru_num)
# define variables
mlvfi = in_condi.createVariable('mLayerVolFracIce', np.float64, ('midToto', 'hru'))
scat = in_condi.createVariable('scalarCanairTemp', np.float64, ('scalarv', 'hru'))
nsnow = in_condi.createVariable('nSnow', np.int32, ('scalarv', 'hru'))
ilh = in_condi.createVariable('iLayerHeight', np.float64, ('ifcToto', 'hru'))
mlmh = in_condi.createVariable('mLayerMatricHead', np.float64, ('midSoil', 'hru'))
ssa = in_condi.createVariable('scalarSnowAlbedo', np.float64, ('scalarv', 'hru'))
dti = in_condi.createVariable('dt_init', np.float64, ('scalarv', 'hru'))
mlt = in_condi.createVariable('mLayerTemp', np.float64, ('midToto', 'hru'))
ssmp = in_condi.createVariable('scalarSfcMeltPond', np.float64, ('scalarv', 'hru'))
sct = in_condi.createVariable('scalarCanopyTemp', np.float64, ('scalarv', 'hru'))
ssd = in_condi.createVariable('scalarSnowDepth', np.float64, ('scalarv', 'hru'))
nsoil = in_condi.createVariable('nSoil', np.int32, ('scalarv', 'hru'))
sswe = in_condi.createVariable('scalarSWE', np.float64, ('scalarv', 'hru'))
scl = in_condi.createVariable('scalarCanopyLiq', np.float64, ('scalarv', 'hru'))
mlvf = in_condi.createVariable('mLayerVolFracLiq', np.float64, ('midToto', 'hru'))
mld = in_condi.createVariable('mLayerDepth', np.float64, ('midToto', 'hru'))
sci = in_condi.createVariable('scalarCanopyIce', np.float64, ('scalarv', 'hru'))
sas = in_condi.createVariable('scalarAquiferStorage', np.float64, ('scalarv', 'hru'))
#%% # add values for the intial condition variables in HRUs
icif = Dataset('summa_zInitialCond.nc')
#print icif.variables['nSoil'][:]

for varname in icif.variables.keys():
    infovar = icif.variables[varname]
    var = icif.variables[varname][:]
    cic = np.repeat(var[:,np.newaxis], hru_num, axis=1); newic = cic.reshape(infovar.shape[0],hru_num)
    in_condi.variables[varname][:]=newic

in_condi.close()
#%%
iccheck = Dataset("summa_zParamTrial_constantDecayRate_am.nc")
print iccheck.variables['albedoMax'][:]
#%%

    
    
    