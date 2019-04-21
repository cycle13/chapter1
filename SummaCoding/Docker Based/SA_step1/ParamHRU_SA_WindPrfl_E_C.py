###       /bin/bash runTestCases_docker.sh
# %matplotlib inline
import numpy as np
from netCDF4 import Dataset
import itertools
#%%
#! turbulent heat fluxes
#! ====================================================================
#z0Snow                    |       0.0010 |       0.0010 |      10.0000  $$$$$$ 3: 0.002 | 0.009 | 0.02
#z0Soil                    |       0.0100 |       0.0010 |      10.0000  $$$$$$ 3: 0.005 | 0.10  | 2
#z0Canopy                  |       0.1000 |       0.0010 |      10.0000  ******
#zpdFraction               |       0.6500 |       0.5000 |       0.8500  ******
#critRichNumber            |       0.2000 |       0.1000 |       1.0000  $$$$$$ 3: 0.2 | 0.5 | 0.8
#Louis79_bparam            |       9.4000 |       9.2000 |       9.6000  A_stability function
#Louis79_cStar             |       5.3000 |       5.1000 |       5.5000  A_stability function
#Mahrt87_eScale            |       1.0000 |       0.5000 |       2.0000  A_stability function
#leafExchangeCoeff         |       0.0100 |       0.0010 |       0.1000  ******
#windReductionParam        |       0.2800 |       0.0000 |       1.0000  $$$$$$ 3: 0.2 | 0.4 | 0.8

p1 = [0.002, 0.008, 0.02] #z0Snow |0.0010 |       0.0010 |      10.0000  $$$$$$ 3: 0.002 | 0.009 | 0.02
p2 = [0.005, 0.05, 0.5] #z0Soil |0.0100 |       0.0010 |      10.0000  $$$$$$ 3: 0.005 | 0.10  | 2
p3 = [0.18, 0.28, 0.7] #windReductionParam |0.2800 |       0.0000 |       1.0000  $$$$$$ 3: 0.2 | 0.4 | 0.8
p4 = [0.2, 0.5] #critRichNumber |0.2000 |       0.1000 |       1.0000  $$$$$$ 3: 0.2 | 0.5 | 0.8
p5 = [9.4, 9.5] #Louis79_bparam | 9.4000 |       9.2000 |       9.6000  A_stability function
p6 = [5.3, 5.4] #Louis79_cStar| 5.3000 |       5.1000 |       5.5000  A_stability function

hru_num = 216

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
param_nam_list = ['z0Snow', 'z0Soil', 'windReductionParam', 'critRichNumber', 'Louis79_bparam', 'Louis79_cStar'] 
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
paramfile = Dataset("summa_zParamTrial_216hru_WPE_V.nc",'w',format='NETCDF3_CLASSIC') #create new paramtrail.nc file
hru = paramfile.createDimension('hru', None)
hidx = paramfile.createVariable('hruIndex', np.float64,('hru',)) # add hruIndex variable

# add 2 new variables to paramfile that we are going to change (use the param list to make the variables for the netCDF)
for param in param_nam_list:
    paramfile.createVariable(param, np.float64,('hru',))

# add any variable to paramfile that we are NOT going to change. Any variable that you are not going to change, it should include in this list
# Ava case in constant_parameter : 'windReductionParam'
constant_params = ['frozenPrecipMultip', 'rootingDepth', 'rootDistExp', 'theta_sat','theta_res','vGn_alpha','vGn_n','k_soil','critSoilWilting','critSoilTranspire','winterSAI','summerLAI','heightCanopyTop','heightCanopyBottom', 'albedoDecayRate']
for params in constant_params:
    paramfile.createVariable(params, np.float64,('hru',))

#paramfile.close()
#%% # add values for the constant variables in HRUs
# this isn't 100% right given different soil type properties, but it's efficient for now
f1 = Dataset('summa_zParamTrial_variableDecayRate.nc')
#for j in f1.variables:
#    print j
for varname in f1.variables.keys():
    # put first value of each variable in var
    var = f1.variables[varname][0]
    #print var
    c = np.full((hru_num,),var)
    #print c
    try :
        paramfile.variables[varname][:]=c
    except IndexError: # size of data array does not conform to slice
        pass
    #paramfile.variables[varname][:]=c
#print f1.variables['hruIndex'][:]
#%% # add values for the changing variables in HRUs
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

print paramfile.variables['z0Soil'][:]

#varcheck = Dataset ('summa_zParamTrial_WPE_216hru.nc')
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
local_atrbt = Dataset("LocalAtributes_WP_216hru.nc",'w',format='NETCDF3_CLASSIC')
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
    #print var
    c2 = np.full((hru_num,),var)
    #print c2
    try :
        local_atrbt.variables[varname][:]=c2
    except IndexError: # size of data array does not conform to slice
        pass
    #local_atrbt.variables[varname][:]=c2

#for varname in la.variables.keys():
#    var = la.variables[varname][:]
#    c = np.repeat(var[:,np.newaxis], hru_num, axis=1); new = c.reshape(hru_num,)
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
newgru = np.array([111111])
local_atrbt.variables['gruId'][:] = newgru

c3 = np.repeat(newgru[:,np.newaxis], hru_num, axis=1); newlad = c3.reshape(hru_num,)
local_atrbt.variables['hru2gruId'][:] = c3

local_atrbt.variables['hruId'][:] = hruidxID

#print local_atrbt.variables['hruId'][:]

local_atrbt.close()
#%%
laCheck = Dataset('LocalAtributes_WP_216hru.nc')

print laCheck.variables['longitude'][:]
for j in laCheck.variables:
    print j
#for varname in check.variables.keys():
#    var = check.variables[varname]
#    print (varname, var.dtype, var.dimensions, var.shape)    
#print f3.variables['hruId'][:] 
#print f3.variables['gruId'][:]
#print f3.variables['longitude'][:]
laCheck.close()
#%% # initial conditions file. 

in_condi = Dataset("InitialCond_WP_216hru.nc",'w',format='NETCDF3_CLASSIC')
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
icif = Dataset('summa_zInitialCond.nc')
#print icif.variables['nSoil'][:]

for varname in icif.variables.keys():
    infovar = icif.variables[varname]
    var = icif.variables[varname][:]
    cic = np.repeat(var[:,np.newaxis], hru_num, axis=1); newic = cic.reshape(infovar.shape[0],hru_num)
    in_condi.variables[varname][:]=newic
#print in_condi.variables['nSoil'][:]

in_condi.close()
#%%
iccheck = Dataset("InitialCond_WP_216hru.nc")
#for varname in iccheck.variables.keys():
#    var = iccheck.variables[varname]
#    print (varname, var.dtype, var.dimensions, var.shape)
#print iccheck.variables['nSoil'][:]
#%%

















    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    