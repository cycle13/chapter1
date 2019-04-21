###       /bin/bash runTestCases_docker.sh 
# 2007 - 2008 as wet year for sensirivity analysis 1st step
import numpy as np
from netCDF4 import Dataset
import itertools

p1 = [0.015] #winterSAI
p2 = [0.020] #summerLAI
p3 = [0.100] #rootingDepth
p4 = [0.050] #heightCanopyTop
p5 = [0.010] #heightCanopyBottom
p6 = [0.890] #throughfallScaleSnow

p7 = [0.4] #0.2, 0.35 , 0.6] #fixedThermalCond_snow

p8 = [280000, 360000, 1000000] #albedoDecayRate ??????????????????????
p9 = [0.8000, 0.90000, 0.95000] #albedoMaxVisible 
p10 = [0.6000, 0.68000, 0.75000] #albedoMinVisible  
p11 = [0.5500, 0.65000, 0.70000] #albedoMaxNearIR
p12 = [0.2000, 0.30000, 0.40000] #albedoMinNearIR
p13 = [1]# 1, 3, 6] 1, 3, 7] #albedoRefresh

p14 = [2] #2, 3, 4] #mw_exp exponent for meltwater flow

p15 = [51] #51, 60, 70] #(m)newSnowDenMin 60 80 100     46 47 48
#p15 = [105] #51, 70, 105] #(c)constSnowDen 70.00, 100.0, 170.0    55 56 57 



#p21 = [0.700, 1.000, 1.500] #Mahrt87_eScale  
#p14 = [0.040, 0.060, 0.080] #Fcapil
#p15 = [0.100, 0.015, 0.350] #k_snow
#p18 = [0.150, 0.200, 0.400] #critRichNumber  
#p19 = [9.300, 9.400, 9.500] #Louis79_bparam  
#p20 = [5.200, 5.300, 5.400] #Louis79_cStar  
#p22 = [0.001, 0.002, 0.004] #z0Snow
paramfile = Dataset("summa_zParamTrial_variableDecayRate_SA_sa2_m.nc",'w',format='NETCDF3_CLASSIC') #create new paramtrail.nc file

def hru_ix_ID(p7, p8, p9, p10, p11, p12, p13, p14, p15):
    ix7 = np.arange(1,len(p7)+1)
    ix8 = np.arange(1,len(p8)+1)
    ix9 = np.arange(1,len(p9)+1)
    ix10 = np.arange(1,len(p10)+1)
    ix11 = np.arange(1,len(p11)+1)
    ix12 = np.arange(1,len(p12)+1)
    ix13 = np.arange(1,len(p13)+1)
    ix14 = np.arange(1,len(p14)+1)
    ix15 = np.arange(1,len(p15)+1)
    
    c = list(itertools.product(ix7,ix8,ix9,ix10,ix11,ix12,ix13,ix14,ix15))
    ix_numlist=[]
    for tup in c:
        ix_numlist.append(''.join(map(str, tup)))
    new_list = [float(i) for i in ix_numlist]

    return(new_list)  

hruidxID = hru_ix_ID(p7, p8, p9, p10, p11, p12, p13, p14, p15)
#
hru_num = np.size(hruidxID)
#%%
# function to create lists of each parameter, this will iterate through to make sure all combinations are covered
def param_fill(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15): 
    b = list(itertools.product(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15)) 
    p1l =[]; p2l =[]; p3l =[]; p4l=[]; p5l = []; p6l =[]; p7l =[]; p8l =[]; p9l =[]; p10l=[]; p11l =[]; p12l =[]; p13l =[]; p14l=[]; p15l = []
    for tup in b:
        p1l.append(tup[0]); p2l.append(tup[1]); p3l.append(tup[2]); p4l.append(tup[3]); p5l.append(tup[4]); p6l.append(tup[5]); p7l.append(tup[6]); p8l.append(tup[7]); p9l.append(tup[8]); p10l.append(tup[9]); p11l.append(tup[10]); p12l.append(tup[11]); p13l.append(tup[12]); p14l.append(tup[13]); p15l.append(tup[14])
    return(p1l, p2l, p3l, p4l, p5l, p6l, p7l, p8l, p9l, p10l, p11l, p12l, p13l, p14l, p15l)  

# call the function on the parameters
valst1 = param_fill(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15)  

#%% #create new paramtrail.nc file and adding vaiables to it --- summa_zParamTrial_variableDecayRate_test
#"settings/SH/t1paramtrial_36h.nc"
hru = paramfile.createDimension('hru', None)
hidx = paramfile.createVariable('hruIndex', np.float64,('hru',)) # add hruIndex variable

# add 2 new variables to paramfile that we are going to change (use the param list to make the variables for the netCDF)
param_nam_list = ['winterSAI', 'summerLAI', 'rootingDepth', 'heightCanopyTop', 'heightCanopyBottom', 'throughfallScaleSnow', 
                  'albedoDecayRate', 'albedoMaxVisible', 'albedoMinVisible', 'albedoMaxNearIR', 'albedoMinNearIR', 'albedoRefresh',
                  'fixedThermalCond_snow', 'mw_exp', 'z0Snow'] 
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
j = 0 
for var in param_nam_list:
    paramfile.variables[var][:]=valst1[j]
    j=j+1
# don't forget the HRU Index!!
paramfile.variables['hruIndex'][:]=hruidxID

for varname in paramfile.variables.keys():
    var = paramfile.variables[varname]
    #print varname, var.dtype, var.dimensions, var.shape

#print paramfile.variables['hruIndex'][:]

#varcheck = Dataset ('summa_zParamTrial_144hru_C.nc')
#print varcheck.variables['hruIndex'][:]
#I checked it in Check.py code
paramfile.close()
#%%
#varcheck = Dataset ('summa_zParamTrial_variableDecayRate_SA_sa2.nc')
#check1 =  varcheck.variables['albedoMinNearIR'][:]
#check2 =  varcheck.variables['albedoMaxNearIR'][:]
#check3 =  varcheck.variables['albedoMinVisible'][:]
#I checked it in Check.py code
#%% # local attributes file
# create a new localAtribute file ---- summa_zLocalAttributes_swampAngel_vtest
local_atrbt = Dataset("summa_zLocalAttributes_swampAngel_sa2.nc",'w',format='NETCDF3_CLASSIC')
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
newgru = np.array([1111111])
local_atrbt.variables['gruId'][:] = newgru

c3 = np.repeat(newgru[:,np.newaxis], hru_num, axis=1); newlad = c3.reshape(hru_num,)
local_atrbt.variables['hru2gruId'][:] = c3

local_atrbt.variables['hruId'][:] = hruidxID

#print local_atrbt.variables['hruId'][:]

local_atrbt.close()
#%%
laCheck = Dataset('summa_zLocalAttributes_swampAngel_sa2.nc')

print laCheck.variables['latitude'][:]
#for j in laCheck.variables:
#    print j
#for varname in check.variables.keys():
#    var = check.variables[varname]
#    print (varname, var.dtype, var.dimensions, var.shape)    
laCheck.close()
#%% # initial conditions file. summa_zInitialCond_vtest

in_condi = Dataset("summa_zInitialCond_sa2.nc",'w',format='NETCDF3_CLASSIC')
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
iccheck = Dataset("summa_zInitialCond_sa2.nc")
#for varname in iccheck.variables.keys():
#    var = iccheck.variables[varname]
#    print (varname, var.dtype, var.dimensions, var.shape)
print iccheck.variables['nSnow'][:]




#%%




