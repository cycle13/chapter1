###       /bin/bash runTestCases_docker.sh 
# 2007 - 2008 as wet year for sensirivity analysis 1st step
import numpy as np
from netCDF4 import Dataset
import itertools
#! ====================================================================
#! soil properties                 (DID  NOT NOOOOOOOOOOOOOOOO0000000000OOOOOOOTTTTT CHANGED; same as Senator Beck)
#! ====================================================================
#frac_sand                 |       0.1600 |       0.0000 |       1.0000
#frac_silt                 |       0.2800 |       0.0000 |       1.0000
#frac_clay                 |       0.5600 |       0.0000 |       1.0000
#fieldCapacity             |       0.2000 |       0.0000 |       1.0000
#wettingFrontSuction       |       0.3000 |       0.1000 |       1.5000
#theta_mp                  |       0.4010 |       0.3000 |       0.6000
#theta_sat                 |       0.5500 |       0.3000 |       0.6000
#theta_res                 |       0.1390 |       0.0010 |       0.1000
#vGn_alpha                 |      -0.8400 |      -1.0000 |      -0.0100
#vGn_n                     |       1.3000 |       1.0000 |       3.0000
#k_soil                    |      7.5d-06 |       1.d-07 |     100.d-07
#specificYield             |       0.2000 |       0.1000 |       0.3000
#specificStorage           |       1.d-09 |       1.d-05 |       1.d-07
#! ====================================================================
#! radiation transfer within snow (DID  NOT NOOOOOOOOOOOOOOOO0000000000OOOOOOOTTTTT TESTED; same as Senator Beck)
#! ====================================================================
#radExt_snow               |      20.0000 |      20.0000 |      20.0000
#directScale               |       0.0900 |       0.0000 |       0.5000
#Frad_direct               |       0.7000 |       0.0000 |       1.0000
#Frad_vis                  |       0.5000 |       0.0000 |       1.0000
#! ====================================================================
#! vegetation properties
#! ====================================================================
#critSoilWilting           |       0.0750 |       0.0000 |       1.0000 (DID  NOT NOOOOOOOOOOOOOOOO0000000000OOOOOOOTTTTT CHANGED; same as Senator Beck)
#leafDimension             |       0.0200 |       0.0100 |       0.1000 (DID  NOT NOOOOOOOOOOOOOOOO0000000000OOOOOOOTTTTT CHANGED; not incorporated in Senator Beck)
#throughfallScaleRain      |       0.8500 |       0.1000 |       0.9000 (DID  NOT NOOOOOOOOOOOOOOOO0000000000OOOOOOOTTTTT CHANGED; not incorporated in Senator Beck)
#refInterceptCapSnow       |       6.6000 |       1.0000 |      10.0000 (DID  NOT NOOOOOOOOOOOOOOOO0000000000OOOOOOOTTTTT CHANGED; not incorporated in Senator Beck)
#refInterceptCapRain       |       1.0000 |       0.0100 |       1.0000 (DID  NOT NOOOOOOOOOOOOOOOO0000000000OOOOOOOTTTTT CHANGED; not incorporated in Senator Beck)
#snowUnloadingCoeff        |       0.0000 |       0.0000 |       1.5d-6 (DID  NOT NOOOOOOOOOOOOOOOO0000000000OOOOOOOTTTTT CHANGED; not incorporated in Senator Beck)
#rootDistExp               |       1.0000 |       0.0100 |       1.0000 (DID  NOT NOOOOOOOOOOOOOOOO0000000000OOOOOOOTTTTT CHANGED; not incorporated in Senator Beck)

#winterSAI                 |       0.0100 |       0.0100 |       3.0000
#summerLAI                 |       0.0100 |       0.0100 |      10.0000
#rootingDepth              |       0.3000 |       0.0100 |      10.0000
#heightCanopyTop           |       2.1000 |       0.0500 |     100.0000
#heightCanopyBottom        |       2.0000 |       0.0000 |       5.0000
#throughfallScaleSnow      |       0.8500 |       0.1000 |       0.9000

p1 = [0.020, 0.050, 0.100, 0.500, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 
      0.015, 0.015, 0.015, 0.015, 0.015, 0.015] #winterSAI

p2 = [0.020, 0.020, 0.020, 0.020, 0.020, 0.050, 0.100, 0.500, 0.020, 0.020, 0.020, 0.020, 0.020, 0.020, 0.020, 0.020, 0.020, 0.020, 
      0.020, 0.020, 0.020, 0.020, 0.020, 0.020] #summerLAI

p3 = [0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.300, 0.500, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 
      0.100, 0.100, 0.100, 0.100, 0.100, 0.100] #rootingDepth

p4 = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.100, 0.500, 1.000, 0.50, 0.50, 0.50, 0.50, 
      0.50, 0.50, 0.50, 0.50, 0.50, 0.50] #heightCanopyTop

p5 = [0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.020, 0.030, 0.050, 0.010, 
      0.010, 0.010, 0.010, 0.010, 0.010, 0.010] #heightCanopyBottom

p6 = [0.890, 0.890, 0.890, 0.890, 0.890, 0.890, 0.890, 0.890, 0.890, 0.890, 0.890, 0.890, 0.890, 0.890, 0.890, 0.890, 0.890, 0.700, 
      0.800, 0.890, 0.890, 0.890, 0.890, 0.890] #throughfallScaleSnow
#%%
#! ====================================================================
#! snow albedo
#! ====================================================================
#albedoDecayRate           |       1.0d+6 |       0.1d+6 |       5.0d+6

p7 = [1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 
      1000000, 1000000, 1000000, 1000000, 1000000, 300000, 500000, 1000000, 2000000] #albedoDecayRate


#
hruidxID = list(np.arange(101,125))
hru_num = np.size(hruidxID)

#%% #create new paramtrail.nc file and adding vaiables to it --- summa_zParamTrial_variableDecayRate_test
#"settings/SH/t1paramtrial_36h.nc"
paramfile = Dataset("summa_zParamTrial_variableDecayRate_SA_density.nc",'w',format='NETCDF3_CLASSIC') #create new paramtrail.nc file
hru = paramfile.createDimension('hru', None)
hidx = paramfile.createVariable('hruIndex', np.float64,('hru',)) # add hruIndex variable

# add 2 new variables to paramfile that we are going to change (use the param list to make the variables for the netCDF)
param_nam_list = ['winterSAI', 'summerLAI', 'rootingDepth', 'heightCanopyTop', 'heightCanopyBottom', 'throughfallScaleSnow', 'albedoDecayRate']
#                  'albedoDecayRate', 'albedoMaxVisible', 'albedoMinVisible', 'albedoMaxNearIR', 'albedoMinNearIR', 'albedoRefresh',
#                  'fixedThermalCond_snow', 'Fcapil', 'k_snow', 'mw_exp', 'z0Snow', 'critRichNumber', 'Louis79_bparam', 
#                  'Louis79_cStar', 'Mahrt87_eScale', 'newSnowDenMin', 'newSnowDenMult', 'newSnowDenScal', 'constSnowDen', 
#                  'newSnowDenAdd', 'newSnowDenMultTemp', 'newSnowDenMultWind', 'newSnowDenMultAnd'] 
for param in param_nam_list:
    paramfile.createVariable(param, np.float64,('hru',))

# add any variable to paramfile that we are NOT going to change. Any variable that you are not going to change, it should include in this list
# Ava case in constant_parameter : 'windReductionParam'
constant_params = ['frozenPrecipMultip','rootDistExp','theta_sat','theta_res','vGn_alpha','vGn_n','k_soil','critSoilWilting','critSoilTranspire']
for params in constant_params:
    paramfile.createVariable(params, np.float64,('hru',))

#paramfile.close()
#%% parameterTrial, Local attributes and initial conditions for senatore beck
pt = Dataset('summa_zParamTrial_variableDecayRate.nc')
la = Dataset('summa_zLocalAttributes_senatorSheltered.nc') #('settings/wrrPaperTestCases/figure07/summa_zLocalAttributes_riparianAspen.nc')
ic = Dataset('summa_zInitialCond.nc') #('settings/wrrPaperTestCases/figure07/summa_zInitialCond.nc')
#for j in pt.variables:
#    print j
#%% # add values for the constant variables in HRUs for parameter Trail file
    # add values for the constant variables in HRUs
for varname in pt.variables.keys():
    var = pt.variables[varname][0]
    c = np.full((hru_num,),var)
    try :
        paramfile.variables[varname][:]=c
    except IndexError: # size of data array does not conform to slice
        pass
#%% creating changing variables and adding values
# add values for the changing variables in HRUs
paramfile.variables['winterSAI'][:]=p1
paramfile.variables['summerLAI'][:]=p2
paramfile.variables['rootingDepth'][:]=p3
paramfile.variables['heightCanopyTop'][:]=p4
paramfile.variables['heightCanopyBottom'][:]=p5
paramfile.variables['throughfallScaleSnow'][:]=p6

paramfile.variables['albedoDecayRate'][:]=p7
#paramfile.variables['albedoMaxVisible'][:]=p8
#paramfile.variables['albedoMinVisible'][:]=p9
#paramfile.variables['albedoMaxNearIR'][:]=p10
#paramfile.variables['albedoMinNearIR'][:]=p11
#paramfile.variables['albedoRefresh'][:]=p12
#
#paramfile.variables['fixedThermalCond_snow'][:]=p13
#paramfile.variables['Fcapil'][:]=p14
#paramfile.variables['k_snow'][:]=p15
#paramfile.variables['mw_exp'][:]=p16
#
#paramfile.variables['z0Snow'][:]=p17
#paramfile.variables['critRichNumber'][:]=p18
#paramfile.variables['Louis79_bparam'][:]=p19
#paramfile.variables['Louis79_cStar'][:]=p20
#paramfile.variables['Mahrt87_eScale'][:]=p21
#
#paramfile.variables['newSnowDenMin'][:]=p22
#paramfile.variables['newSnowDenMult'][:]=p23
#paramfile.variables['newSnowDenScal'][:]=p24
#paramfile.variables['constSnowDen'][:]=p25
#paramfile.variables['newSnowDenAdd'][:]=p26
#paramfile.variables['newSnowDenMultTemp'][:]=p27
#paramfile.variables['newSnowDenMultWind'][:]=p28
#paramfile.variables['newSnowDenMultAnd'][:]=p29

paramfile.variables['hruIndex'][:]=hruidxID

for varname in paramfile.variables.keys():
    var = paramfile.variables[varname]
    print varname, var.dtype, var.dimensions, var.shape

print paramfile.variables['hruIndex'][:]
paramfile.close()
#%%
varcheck = Dataset ('summa_zParamTrial_variableDecayRate_SA_density.nc')
#print varcheck.variables['winterSAI'][:]
#I checked it in Check.py code
#%% # local attributes file
# create a new localAtribute file ---- summa_zLocalAttributes_swampAngel_vtest
local_atrbt = Dataset("summa_zLocalAttributes_swampAngel_density.nc",'w',format='NETCDF3_CLASSIC')
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
#%% # add values for the constant variables in HRUs for local atribute file
for varname in la.variables.keys():
    var = la.variables[varname][0]
    #print var
    c2 = np.full((hru_num,),var)
    #print c2
    try :
        local_atrbt.variables[varname][:]=c2
    except IndexError: # size of data array does not conform to slice
        pass
    #local_atrbt.variables[varname][:]=c2

#%% # get the hru, gru, and hru2gru in local_atribute file
newgru = np.array([111])
local_atrbt.variables['gruId'][:] = newgru

c3 = np.repeat(newgru[:,np.newaxis], hru_num, axis=1); newlad = c3.reshape(hru_num,)
local_atrbt.variables['hru2gruId'][:] = c3

local_atrbt.variables['hruId'][:] = hruidxID

#print local_atrbt.variables['hruId'][:]

local_atrbt.close()
#%%
laCheck = Dataset('summa_zLocalAttributes_swampAngel_density.nc')

print laCheck.variables['latitude'][:]
#for j in laCheck.variables:
#    print j
#for varname in check.variables.keys():
#    var = check.variables[varname]
#    print (varname, var.dtype, var.dimensions, var.shape)    
laCheck.close()
#%% # initial conditions file. summa_zInitialCond_vtest

in_condi = Dataset("summa_zInitialCond_density.nc",'w',format='NETCDF3_CLASSIC')
#print ic.variables.keys()

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

for varname in ic.variables.keys():
    infovar = ic.variables[varname]
    var = ic.variables[varname][:]
    cic = np.repeat(var[:,np.newaxis], hru_num, axis=1); newic = cic.reshape(infovar.shape[0],hru_num)
    in_condi.variables[varname][:]=newic

print in_condi.variables['nSoil'][:]

in_condi.close()
#%%
iccheck = Dataset("summa_zInitialCond_density.nc")
#for varname in iccheck.variables.keys():
#    var = iccheck.variables[varname]
#    print (varname, var.dtype, var.dimensions, var.shape)
print iccheck.variables['nSnow'][:]




#%%






