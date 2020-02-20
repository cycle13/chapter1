###       /bin/bash runTestCases_docker_hs.sh   snwDensity snwDensity
# 2007 - 2008 as wet year for sensirivity analysis 1st step
import numpy as np
from netCDF4 import Dataset
import itertools
import csv
import matplotlib.pyplot as plt 

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

def param_fill(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17):#, p18 
    b = list(itertools.product(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17)) #, p18
    p1l =[]; p2l =[]; p3l =[]; p4l=[]; p5l = []; p6l =[]; p7l =[]; p8l =[]; p9l =[]; p10l=[]; 
    p11l =[]; p12l =[]; p13l =[]; p14l=[]; p15l = []; p16l =[]; p17l=[]#; p18l = []
    for tup in b:
        p1l.append(tup[0]); p2l.append(tup[1]); p3l.append(tup[2]); p4l.append(tup[3]); 
        p5l.append(tup[4]); p6l.append(tup[5]); p7l.append(tup[6]); p8l.append(tup[7]); 
        p9l.append(tup[8]); p10l.append(tup[9]); p11l.append(tup[10]); p12l.append(tup[11]); 
        p13l.append(tup[12]); p14l.append(tup[13]); p15l.append(tup[14]);p16l.append(tup[15]); 
        p17l.append(tup[16])#; p18l.append(tup[17])
    return(p1l, p2l, p3l, p4l, p5l, p6l, p7l, p8l, p9l, p10l, p11l, p12l, p13l, p14l, p15l, p16l, p17l)# , p18l 

#%%  all parameters
with open("C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa2_bestSweSD/STAR_out_P13_233_L_lstCC.csv") as safd:
    reader = csv.reader(safd)
    params0 = [r for r in reader]
params1 = params0[1:]
sa_fd_column = []
for csv_counter1 in range (len (params1)):
    for csv_counter2 in range (22):
        sa_fd_column.append(float(params1[csv_counter1][csv_counter2]))
params_sa=np.reshape(sa_fd_column,(len (params1),22))

paramfile = Dataset("C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa2_bestSweSD/summa_zParamTrial_variableDecayRate_sa.nc",'w',format='NETCDF3_CLASSIC') #create new paramtrail.nc file

#hruidxID = [10317,10327,10336,10369,10375,10524,10526,10726] #p12
hruidxID = [10030,10041,10042,10053,10054,10077,10078,10079,10081,10082,10083,10084,10085,10094,10095,
            10096,10097,10098,10103,10104,10105,10106,10107,10108,10109,10110,10111,10112,10113,10114,
            10115,10116,10117,10118,10119,10120,10122,10126,10156,10223,10224,10225,10234,10235,10336,
            10337,10342,10348,10365,10379,10380,10381,10392,10393,10394,10398,10399,10400,10408,10409,
            10410,10411,10412,10413,10414,10421,10423,10426,10427,10428,10429,10430,10431,10432,10433,
            10436,10437,10438,10439,10440,10441,10442,10443,10444,10453,10454,10460,10461,10462,10617,
            10625,10626,10628,10637,10650,10652,10653,10669,10670,10737,10749,10752,10756,10767,10768,
            10769,10788,10790,10794,10795,10796,10825,10826,10827,10833,10834,10835,10836,10837,10841,
            10850,10865,10866,10867,10879,10880,10881,10882,10883,10884,10885,10886,10887,10888,10904,
            10910,10911,10912,10942,10945,10948,10951,10958,10959,10960,11272,11273,11274]
hru_num = np.size(hruidxID)

#%% #create new paramtrail.nc file and adding vaiables to it --- summa_zParamTrial_variableDecayRate_test
hru = paramfile.createDimension('hru', None)
hidx = paramfile.createVariable('hruIndex', np.float64,('hru',)) # add hruIndex variable


param_nam_list = ['LAIMIN','LAIMAX','winterSAI','summerLAI','rootingDepth','heightCanopyTop','heightCanopyBottom',
                  'throughfallScaleSnow','newSnowDenMin','albedoDecayRate','albedoMaxVisible','albedoMinVisible',
                  'albedoMaxNearIR','albedoMinNearIR','albedoRefresh','albedoSootLoad',
                  'Frad_vis','mw_exp','k_snow','critRichNumber','tempCritRain','fixedThermalCond_snow'] # 'frozenPrecipMultip'
#	
# call the function on the parameters
#valst1 = param_fill(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17) #,p18 

for param in param_nam_list:
    paramfile.createVariable(param, np.float64,('hru',))

constant_params = ['rootDistExp','theta_sat','theta_res','vGn_alpha','vGn_n','k_soil','critSoilWilting','critSoilTranspire', 'frozenPrecipMultip']
for params in constant_params:
    paramfile.createVariable(params, np.float64,('hru',))
#paramfile.close()
#%% parameterTrial, Local attributes and initial conditions for senatore beck
pt = Dataset('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/testVegFunctionsImpact/summa_zParamTrial_variableDecayRate.nc')
la = Dataset('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/testVegFunctionsImpact/summa_zLocalAttributes_senatorSheltered.nc') #('settings/wrrPaperTestCases/figure07/summa_zLocalAttributes_riparianAspen.nc')
ic = Dataset('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/testVegFunctionsImpact/summa_zInitialCond.nc') #('settings/wrrPaperTestCases/figure07/summa_zInitialCond.nc')
ic.variables['nSoil'][:]
#for j in pt.variables:
#    print j
#%% # add values for the constant variables in HRUs for parameter Trail file
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
    paramfile.variables[var][:]=params_sa[:,j]
    j=j+1
#don't forget the HRU Index!!
paramfile.variables['hruIndex'][:]=hruidxID

for varname in paramfile.variables.keys():
    var = paramfile.variables[varname]
    print varname, var.dtype, var.dimensions, var.shape

print (paramfile.variables['frozenPrecipMultip'][:])
aaa = paramfile.variables['LAIMIN'][:]
paramfile.close()
#%% 
#varcheck = Dataset ('C:/Users/HHS/summaTestCases_2.x/settings/swampAngel/sa_sa2_vars/summa_zParamTrial_variableDecayRate_sa_sa2.nc')#C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa_sa2_VARs_p12FPM_fTCs/
##print varcheck.variables['fixedThermalCond_snow'][:]
##print np.size(varcheck.variables['fixedThermalCond_snow'][:])
#
#for varname in varcheck.variables.keys():
#    var = paramfile.variables[varname]
#    print varname, var.dtype, var.dimensions, var.shape
#
#print varcheck.variables['frozenPrecipMultip'][:]
#I checked it in Check.py code
#%% # local attributes file
# create a new localAtribute file ---- summa_zLocalAttributes_swampAngel_vtest
local_atrbt = Dataset("C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa2_bestSweSD/summa_zLocalAttributes_swampAngel.nc",'w',format='NETCDF3_CLASSIC')
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

c4 = np.repeat([3.5], hru_num)
local_atrbt.variables['mHeight'][:] = c4

#print local_atrbt.variables['hruId'][:]

local_atrbt.close()
#%%
lacheck = Dataset('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa2_bestSweSD/summa_zLocalAttributes_swampAngel.nc')

print lacheck.variables['mHeight'][:]
#for j in laCheck.variables:
#    print j
for varname in lacheck.variables.keys():
    var = lacheck.variables[varname]
    print (varname, var.dtype, var.dimensions, var.shape)    
#lacheck.close()
#%% # initial conditions file. summa_zInitialCond_vtest

in_condi = Dataset("C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa2_bestSweSD/summa_zInitialCond.nc",'w',format='NETCDF3_CLASSIC')
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

print in_condi.variables['nSnow'][:]

in_condi.close()
#%%
iccheck = Dataset("C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa2_bestSweSD/summa_zInitialCond.nc")
#for varname in iccheck.variables.keys():
#    var = iccheck.variables[varname]
#    print (varname, var.dtype, var.dimensions, var.shape)
print len(iccheck.variables['nSoil'][0])


