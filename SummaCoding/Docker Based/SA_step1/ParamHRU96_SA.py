###       /bin/bash runTestCases_docker.sh
# %matplotlib inline
import numpy as np
from netCDF4 import Dataset
import itertools
#%%
#albedoMax                 |       0.8400 |       0.7000 |       0.9500 $$$$$$ 2: 0.75 | 0.92
#albedoMinWinter           |       0.5500 |       0.6000 |       1.0000 $$$$$$ 2: 0.65 | 0.90
#albedoMinSpring           |       0.5500 |       0.3000 |       1.0000 $$$$$$ 2: 0.40 | 0.90
#albedoMaxVisible          |       0.9500 |       0.7000 |       0.9500 ****** 1: 0.95
#albedoMinVisible          |       0.7500 |       0.5000 |       0.7500 ****** 1: 0.75
#albedoMaxNearIR           |       0.6500 |       0.5000 |       0.7500 $$$$$$ 2: 0.55 | 0.70
#albedoMinNearIR           |       0.3000 |       0.1500 |       0.4500 $$$$$$ 2: 0.20 | 0.43
#albedoDecayRate           |       1.0d+6 |       0.1d+6 |       5.0d+6 $$$$$$ 3: 0.8d+6 | 2d+6 | 4.0d+6

p1 = [0.8, 0.92] #albedoMax      |       0.8400 |       0.7000 |       0.9500 $$$$$$ 2: 0.75 | 0.92
p2 = [0.64, 0.9] #albedoMinWinter   |       0.5500 |       0.6000 |       1.0000 $$$$$$ 2: 0.65 | 0.90
p3 = [0.5, 0.9] #albedoMinSpring    |       0.5500 |       0.3000 |       1.0000 $$$$$$ 2: 0.40 | 0.90
p4 = [0.55, 0.7] #albedoMaxNearIR   |       0.6500 |       0.5000 |       0.7500 $$$$$$ 2: 0.55 | 0.70
p5 = [0.25, 0.4] #albedoMinNearIR  |       0.3000 |       0.1500 |       0.4500 $$$$$$ 2: 0.20 | 0.43
p6 = [900000, 2000000, 3500000] #albedoDecayRate  |       1.0d+6 |       0.1d+6 |       5.0d+6 $$$$$$ 3: 0.8d+6 | 2d+6 | 4.0d+6

def hru_ix_ID(p1, p2, p3, p4, p5, p6):
    ix1 = np.arange(1,len(p1)+1)
    ix2 = np.arange(1,len(p2)+1)
    ix3 = np.arange(1,len(p3)+1)
    ix4 = np.arange(1,len(p4)+1)
    ix5 = np.arange(1,len(p5)+1)
    ix6 = np.arange(1,len(p6)+1)
    
    c = list(itertools.product(ix1,ix2,ix3,ix4,ix5,ix6))
    ix_numlist=[]
    for tup in c:
        ix_numlist.append(''.join(map(str, tup)))
    new_list = [float(i) for i in ix_numlist]

    return(new_list)  

hruidxID = hru_ix_ID(p1, p2, p3, p4, p5, p6)
# %store hruidxID
#%%
param_nam_list = ['albedoMax', 'albedoMinWinter', 'albedoMinSpring', 'albedoMaxNearIR', 'albedoMinNearIR', 'albedoDecayRate'] 
# function to create lists of each parameter, this will iterate through to make sure all combinations are covered
def param_fill(p1, p2, p3, p4, p5, p6): 
    b = list(itertools.product(p1, p2, p3, p4, p5, p6)) 
    p1l =[]; p2l =[]; p3l =[]; p4l=[]; p5l = []; p6l =[]
    for tup in b:
        p1l.append(tup[0]); p2l.append(tup[1]); p3l.append(tup[2]); p4l.append(tup[3]); p5l.append(tup[4]); p6l.append(tup[5])
    return(p1l, p2l, p3l, p4l, p5l, p6l)    

# call the function on the parameters
valst1 = param_fill(p1, p2, p3, p4, p5, p6)  
#%% #create new paramtrail.nc file and adding vaiables to it
#"settings/SH/t1paramtrial_36h.nc"
paramfile = Dataset("summa_zParamTrial_96hru.nc",'w',format='NETCDF3_CLASSIC') #create new paramtrail.nc file
hru = paramfile.createDimension('hru', None)
hidx = paramfile.createVariable('hruIndex', np.float64,('hru',)) # add hruIndex variable

# add 2 new variables to paramfile that we are going to change (use the param list to make the variables for the netCDF)
for param in param_nam_list:
    paramfile.createVariable(param, np.float64,('hru',))

# add any variable to paramfile that we are NOT going to change. Any variable that you are not going to change, it should include in this list
# Ava case in constant_parameter : 'windReductionParam'
constant_params = ['frozenPrecipMultip', 'rootingDepth', 'rootDistExp', 'theta_sat','theta_res','vGn_alpha','vGn_n','k_soil','critSoilWilting','critSoilTranspire','winterSAI','summerLAI','heightCanopyTop','heightCanopyBottom']
for params in constant_params:
    paramfile.createVariable(params, np.float64,('hru',))

#paramfile.close()
#%% # add values for the constant variables in 96 HRUs
# this isn't 100% right given different soil type properties, but it's efficient for now
f1 = Dataset('summa_zParamTrial_constantDecayRate.nc')
for j in f1.variables:
    print j
for varname in f1.variables.keys():
    # put first value of each variable in var
    var = f1.variables[varname][0]
    #print var
    c = np.full((96,),var)
    #print c
    try :
        paramfile.variables[varname][:]=c
    except IndexError: # size of data array does not conform to slice
        pass
    #paramfile.variables[varname][:]=c
#print f1.variables['hruIndex'][:]
#%% # add values for the changing variables in 6 HRUs
j = 0 
for var in param_nam_list:
    paramfile.variables[var][:]=valst1[j]
    j=j+1
# don't forget the HRU Index!!
paramfile.variables['hruIndex'][:]=hruidxID

#print paramfile.variables['hruIndex'][:]
#print paramfile.variables['newSnowDenMult'][:]
#print paramfile.variables['albedoMax'][:]
#print paramfile.variables['rootingDepth'][:]
#print paramfile.variables['albedoDecayRate'][:]
for varname in paramfile.variables.keys():
    var = paramfile.variables[varname]
    #print varname, var.dtype, var.dimensions, var.shape

#print paramfile.variables['hruIndex'][:]

#varcheck = Dataset ('t1paramtrial_6p.nc')
#print varcheck.variables['hruIndex'][:]
#I checked it in Check.py code
paramfile.close()
#%% #Local attributes and initial conditions
la = Dataset('summa_zLocalAttributes_senatorSheltered.nc') #('settings/wrrPaperTestCases/figure07/summa_zLocalAttributes_riparianAspen.nc')
ic = Dataset('summa_zInitialCond.nc') #('settings/wrrPaperTestCases/figure07/summa_zInitialCond.nc')
#print(ic.variables.keys())
#print(la.variables.keys())
#%% # local attributes file
# create a new localAtribute file
local_atrbt = Dataset("localatrbt6.nc",'w',format='NETCDF3_CLASSIC')
# define dimensions 
hru = local_atrbt.createDimension('hru', 6) 
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
#%% # add values for the constant variables in 6 HRUs
for varname in la.variables.keys():
    var = la.variables[varname][0]
    #print var
    c2 = np.full((6,),var)
    #print c2
    try :
        local_atrbt.variables[varname][:]=c2
    except IndexError: # size of data array does not conform to slice
        pass
    #local_atrbt.variables[varname][:]=c2

#for varname in la.variables.keys():
#    var = la.variables[varname][:]
#    c = np.repeat(var[:,np.newaxis], 6, axis=1); new = c.reshape(36,)
#    local_atrbt.variables[varname][:]=new

#%% #I probably do not need this, except I want to use another watershed that i will need to put the value by hand.
# '6' will be the length of your hru index list
#elevation = np.array([1900])
#len_elev = np.repeat(elevation[:,np.newaxis], 36, axis=1); len_elev=len_elev.reshape(36,)
#local_atrbt.variables['elevation'][:]=len_elev
#soiltype = np.array([7])
#len_st = np.repeat(soiltype[:,np.newaxis], 36, axis=1); len_st=len_st.reshape(36,)
#local_atrbt.variables['soilTypeIndex'][:]=len_st
#vegtype = np.array([14])
#len_vt = np.repeat(vegtype[:,np.newaxis], 36, axis=1); len_vt=len_vt.reshape(36,)
#local_atrbt.variables['vegTypeIndex'][:]=len_vt
#latty = np.array([35.888447])
#len_lat = np.repeat(latty[:,np.newaxis], 36, axis=1); len_lat=len_lat.reshape(36,)
#local_atrbt.variables['latitude'][:]=len_lat
#longg =np.array([-106.532114])
#len_lon= np.repeat(longg[:,np.newaxis], 36, axis=1); len_lon=len_lon.reshape(36,)
#local_atrbt.variables['longitude'][:]=len_lon 
#%% # get the hru, gru, and hru2gru in local_atribute file
newgru = np.array([111])
local_atrbt.variables['gruId'][:] = newgru

c3 = np.repeat(newgru[:,np.newaxis], 6, axis=1); newlad = c3.reshape(6,)
local_atrbt.variables['hru2gruId'][:] = c3

local_atrbt.variables['hruId'][:] = hruidxID

#print local_atrbt.variables['hruId'][:]

local_atrbt.close()
#%%
laCheck = Dataset('localatrbt6.nc')

print laCheck.variables['hruId'][:]

#for varname in check.variables.keys():
#    var = check.variables[varname]
#    print (varname, var.dtype, var.dimensions, var.shape)    
#print f3.variables['hruId'][:] 
#print f3.variables['gruId'][:]
#print f3.variables['longitude'][:]
laCheck.close()
#%% # initial conditions file. 

in_condi = Dataset("inCondi6.nc",'w',format='NETCDF3_CLASSIC')
#print ic.variables.keys()

# define dimensions 
midtoto = in_condi.createDimension('midToto',8)
midsoil = in_condi.createDimension('midSoil',8)
idctoto = in_condi.createDimension('ifcToto',9)
scalarv = in_condi.createDimension('scalarv', 1)
# this is the number you will change to the number of HRU's from your param trial file
hrud = in_condi.createDimension('hru', 6)
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
#%% # add values for the intial condition variables in 6 HRUs
icif = Dataset('summa_zInitialCond.nc')
#print icif.variables['nSoil'][:]

for varname in icif.variables.keys():
    infovar = icif.variables[varname]
    var = icif.variables[varname][:]
    cic = np.repeat(var[:,np.newaxis], 6, axis=1); newic = cic.reshape(infovar.shape[0],6)
    in_condi.variables[varname][:]=newic
#print in_condi.variables['nSoil'][:]

in_condi.close()
#%%
iccheck = Dataset("inCondi6.nc")
#for varname in iccheck.variables.keys():
#    var = iccheck.variables[varname]
#    print (varname, var.dtype, var.dimensions, var.shape)
#print iccheck.variables['nSoil'][:]
#%%

















    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    