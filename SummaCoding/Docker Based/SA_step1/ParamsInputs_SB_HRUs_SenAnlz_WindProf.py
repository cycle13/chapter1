###       /bin/bash runTestCases_docker.sh
# %matplotlib inline
import numpy as np
from netCDF4 import Dataset
import itertools
#%%
#! ====================================================================
#! turbulent heat fluxes
#! variable decay rate model
#! ====================================================================
#z0Snow                    |       0.0010 |       0.0010 |      10.0000  
#z0Soil                    |       0.0100 |       0.0010 |      10.0000  
#z0Canopy                  |       0.1000 |       0.0010 |      10.0000  
#zpdFraction               |       0.6500 |       0.5000 |       0.8500  
#critRichNumber            |       0.2000 |       0.1000 |       1.0000  
#Louis79_bparam            |       9.4000 |       9.2000 |       9.6000  A_stability function
#Louis79_cStar             |       5.3000 |       5.1000 |       5.5000  A_stability function
#Mahrt87_eScale            |       1.0000 |       0.5000 |       2.0000  A_stability function
#leafExchangeCoeff         |       0.0100 |       0.0010 |       0.1000  
#windReductionParam        |       0.2800 |       0.0000 |       1.0000  

p1 = [0.001,0.010,0.100,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001]
p2 = [0.010,0.010,0.010,0.001,0.010,0.100,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010]
p3 = [0.100,0.100,0.100,0.100,0.100,0.100,0.020,0.100,1.000,0.100,0.100,0.100,0.100,0.100,0.100,0.100,0.100,0.100,0.100,0.100,0.100,0.100,0.100,0.100,0.100,0.100,0.100,0.100,0.100,0.100]
p4 = [0.650,0.650,0.650,0.650,0.650,0.650,0.650,0.650,0.650,0.550,0.650,0.800,0.650,0.650,0.650,0.650,0.650,0.650,0.650,0.650,0.650,0.650,0.650,0.650,0.650,0.650,0.650,0.650,0.650,0.650] 
p5 = [0.200,0.200,0.200,0.200,0.200,0.200,0.200,0.200,0.200,0.200,0.200,0.200,0.150,0.200,0.500,0.200,0.200,0.200,0.200,0.200,0.200,0.200,0.200,0.200,0.200,0.200,0.200,0.200,0.200,0.200]
p6 = [9.400,9.400,9.400,9.400,9.400,9.400,9.400,9.400,9.400,9.400,9.400,9.400,9.400,9.400,9.400,9.250,9.400,9.550,9.400,9.400,9.400,9.400,9.400,9.400,9.400,9.400,9.400,9.400,9.400,9.400]
p7 = [5.300,5.300,5.300,5.300,5.300,5.300,5.300,5.300,5.300,5.300,5.300,5.300,5.300,5.300,5.300,5.300,5.300,5.300,5.150,5.300,5.450,5.300,5.300,5.300,5.300,5.300,5.300,5.300,5.300,5.300] 
p8 = [1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,0.600,1.000,1.700,1.000,1.000,1.000,1.000,1.000,1.000] 
p9 = [0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.005,0.010,0.050,0.010,0.010,0.010] 
p10 =[0.280,0.280,0.280,0.280,0.280,0.280,0.280,0.280,0.280,0.280,0.280,0.280,0.280,0.280,0.280,0.280,0.280,0.280,0.280,0.280,0.280,0.280,0.280,0.280,0.280,0.280,0.280,0.100,0.280,0.700] 

hru_num = 30
hruidxID =np.arange(1001, 1001+hru_num) 
param_nam_list = ['z0Snow', 'z0Soil', 'z0Canopy', 'zpdFraction', 'critRichNumber', 'Louis79_bparam', 'Louis79_cStar', 'Mahrt87_eScale', 'leafExchangeCoeff', 'windReductionParam'] 

#%% #create new paramtrail.nc file for constantDecayRate model
paramfile = Dataset("summa_zParamTrial_variableDecayRate_WindPSA.nc",'w',format='NETCDF3_CLASSIC') #create new paramtrail.nc file
hru = paramfile.createDimension('hru', None)

hidx = paramfile.createVariable('hruIndex', np.float64,('hru',)) # add hruIndex variable

print paramfile.variables['hruIndex'][:]
#%%
constant_params = ['albedoDecayRate', 'frozenPrecipMultip', 'rootingDepth', 'rootDistExp', 'theta_sat','theta_res','vGn_alpha','vGn_n','k_soil','critSoilWilting','critSoilTranspire','winterSAI','summerLAI','heightCanopyTop','heightCanopyBottom']
for params in constant_params:
    paramfile.createVariable(params, np.float64,('hru',))
    
# add values for the constant variables in HRUs
pt_c = Dataset('summa_zParamTrial_variableDecayRate.nc')

for varname in pt_c.variables.keys():
    var = pt_c.variables[varname][0]
    c = np.full((hru_num,),var)
    try :
        paramfile.variables[varname][:]=c
    except IndexError: # size of data array does not conform to slice
        pass
    
#%% creating changing variables and adding values
paramfile.variables['hruIndex'][:]=hruidxID

for param in param_nam_list:
    paramfile.createVariable(param, np.float64,('hru',))

# add values for the changing variables in HRUs
paramfile.variables['z0Snow'][:]=p1
paramfile.variables['z0Soil'][:]=p2
paramfile.variables['z0Canopy'][:]=p3
paramfile.variables['zpdFraction'][:]=p4
paramfile.variables['critRichNumber'][:]=p5
paramfile.variables['Louis79_bparam'][:]=p6
paramfile.variables['Louis79_cStar'][:]=p7
paramfile.variables['Mahrt87_eScale'][:]=p8
paramfile.variables['leafExchangeCoeff'][:]=p9
paramfile.variables['windReductionParam'][:]=p10

for varname in paramfile.variables.keys():
    var = paramfile.variables[varname]
    #print varname, var.dtype, var.dimensions, var.shape

print paramfile.variables['albedoDecayRate'][:]
paramfile.close()

#%%
ptcheck = Dataset("summa_zParamTrial_variableDecayRate_WindPSA.nc")
print ptcheck.variables['Mahrt87_eScale'][:]

#%%#%% #Local attributes and initial conditions
la = Dataset('summa_zLocalAttributes_senatorSheltered.nc') #('settings/wrrPaperTestCases/figure07/summa_zLocalAttributes_riparianAspen.nc')
ic = Dataset('summa_zInitialCond.nc') #('settings/wrrPaperTestCases/figure07/summa_zInitialCond.nc')
#%% # local attributes file
# create a new localAtribute file
local_atrbt = Dataset("summa_zLocalAttributes_senatorSheltered_WindPSA.nc",'w',format='NETCDF3_CLASSIC')
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
newgru = np.array([1111])
local_atrbt.variables['gruId'][:] = newgru

c3 = np.repeat(newgru[:,np.newaxis], hru_num, axis=1); newlad = c3.reshape(hru_num,)
local_atrbt.variables['hru2gruId'][:] = c3

local_atrbt.variables['hruId'][:] = hruidxID

local_atrbt.close()
#%%
laCheck = Dataset('summa_zLocalAttributes_senatorSheltered_WindPSA.nc')

print laCheck.variables['longitude'][:]
#for j in laCheck.variables:
#    print j

laCheck.close()
#%% # initial conditions file

in_condi = Dataset("summa_zInitialCond_WindPSA.nc",'w',format='NETCDF3_CLASSIC')

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












    
    
    