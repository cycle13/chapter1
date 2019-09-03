###       /bin/bash runTestCases_docker_hs.sh   snwDensity snwDensity
# 2007 - 2008 as wet year for sensirivity analysis 1st step
import numpy as np
from netCDF4 import Dataset
import itertools

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

p1 = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,
      0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] #LAIMIN
p2 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] #LAIMAX
p3 = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,
      0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] #winterSAI
p4 = [0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,
      0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9] #summerLAI #https://www.sciencedirect.com/science/article/pii/S0924271608000166
p5 = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,
      0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5] #rootingDepth
p6 = [0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,
      0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4] #heightCanopyTop
p7 = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,
      0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01] #heightCanopyBottom
p8 = [0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,
      0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99] #throughfallScaleSnow
p9 = [55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,
      55,55,55,55,55,55] #newSnowDenMin

p10 = [500000,3000000,1000000,1000000,1000000,1000000,1000000,1000000,1000000,1000000,1000000,
       1000000,1000000,1000000,1000000,1000000,1000000,1000000,1000000,1000000,1000000,1000000,
       1000000,1000000,1000000,1000000,1000000,1000000,1000000,1000000,1000000,1000000,1000000,
       1000000,1000000,1000000] #[500000, 1000000, 1300000] ##albedoDecayRate |       1.0d+6 |       0.1d+6 |       5.0d+6 
p11 = [0.94,0.94,0.7,0.94,0.94,0.94,0.94,0.94,0.94,0.94,0.94,0.94,0.94,0.94,0.94,0.94,0.94,0.94,
       0.94,0.94,0.94,0.94,0.94,0.94,0.94,0.94,0.94,0.94,0.94,0.94,0.94,0.94,0.94,0.94,0.94,0.94] #[0.8, 0.9, 0.94] #albedoMaxVisible |       0.9500 |       0.7000 |       0.9500
p12 =[0.74,0.74,0.74,0.6,0.74,0.74,0.74,0.74,0.74,0.74,0.74,0.74,0.74,0.74,0.74,0.74,0.74,0.74,
      0.74,0.74,0.74,0.74,0.74,0.74,0.74,0.74,0.74,0.74,0.74,0.74,0.74,0.74,0.74,0.74,0.74,0.74] #[0.6, 0.68, 0.74] #albedoMinVisible |       0.7500 |       0.5000 |       0.7500
p13 =[0.7,0.7,0.7,0.7,0.55,0.75,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,
      0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7] #[0.55, 0.65, 0.7] #albedoMaxNearIR |       0.6500 |       0.5000 |       0.7500
p14 =[0.3,0.3,0.3,0.3,0.3,0.3,0.2,0.44,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,
      0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3] #[0.2, 0.3, 0.4] #albedoMinNearIR  |       0.3000 |       0.1500 |       0.4500
p15 =[3,3,3,3,3,3,3,3,1,8,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3] #[6]# 1, 3, 6] #albedoRefresh |       1.0000 |       1.0000 |      10.0000
p16 =[0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.15,0.45,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,
      0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3] #albedoSootLoad            |       0.3000 |       0.1000 |       0.5000

p17 = [273.16,273.16,273.16,273.16,273.16,273.16,273.16,273.16,273.16,273.16,273.16,273.16,274.16,
       274.16,274.16,274.16,274.16,274.16,274.16,274.16,274.16,274.16,274.16,274.16,274.16,274.16,
       274.16,274.16,274.16,274.16,274.16,274.16,274.16,274.16,274.16,274.16] #tempCritRain              |     273.1600

p18 =[0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.001,0.003,
      0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,
      0.002,0.002,0.002,0.002,0.002,0.002] #[0.002] #[0.001, 0.002] #z0Snow
p19 =[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,2,3.5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4] #[4] #2, 3, 4] #mw_exp exponent for meltwater flow
p20 =[0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.02,
      0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.09] #Fcapil                    |       0.0600 |       0.0100 |       0.1000
p21 =[0.015,0.015,0.015,0.015,0.015,0.015,0.015,0.015,0.015,0.015,0.015,0.015,0.015,0.015,0.015,
      0.015,0.015,0.015,0.015,0.045,0.015,0.015,0.015,0.015,0.015,0.015,0.015,0.015,0.015,0.015,
      0.015,0.015,0.015,0.015,0.015,0.015] #k_snow                    |       0.0150 |       0.0050 |       0.0500
p22 =[0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.2,0.95,
      0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7] #Frad_direct               |       0.7000 |       0.0000 |       1.0000
p23 =[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,
      0.2,0.8,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5] #Frad_vis                  |       0.5000 |       0.0000 |       1.0000

p24 = [9.4,9.4,9.4,9.4,9.4,9.4,9.4,9.4,9.4,9.4,9.4,9.4,9.4,9.4,9.4,9.4,9.4,9.4,9.4,9.4,9.4,9.4,
       9.4,9.4,9.2,9.6,9.4,9.4,9.4,9.4,9.4,9.4,9.4,9.4,9.4,9.4] #Louis79_bparam            |       9.4000 |       9.2000 |       9.6000
p25 = [5.3,5.3,5.3,5.3,5.3,5.3,5.3,5.3,5.3,5.3,5.3,5.3,5.3,5.3,5.3,5.3,5.3,5.3,5.3,5.3,5.3,5.3,
       5.3,5.3,5.3,5.3,5.1,5.5,5.3,5.3,5.3,5.3,5.3,5.3,5.3,5.3] #Louis79_cStar             |       5.3000 |       5.1000 |       5.5000
p26 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.7,1.6,1,1,1,1,1,1] #Mahrt87_eScale            |       1.0000 |       0.5000 |       2.0000
p27 = [0.28,0.28,0.28,0.28,0.28,0.28,0.28,0.28,0.28,0.28,0.28,0.28,0.28,0.28,0.28,0.28,0.28,
       0.28,0.28,0.28,0.28,0.28,0.28,0.28,0.28,0.28,0.28,0.28,0.28,0.28,0.1,0.7,0.28,0.28,0.28,
       0.28] #windReductionParam        |       0.2800 |       0.0000 |       1.0000

p28 =[6.6,6.6,6.6,6.6,6.6,6.6,6.6,6.6,6.6,6.6,6.6,6.6,6.6,6.6,6.6,6.6,6.6,6.6,6.6,6.6,6.6,6.6,
      6.6,6.6,6.6,6.6,6.6,6.6,6.6,6.6,6.6,6.6,3,9,6.6,6.6] #refInterceptCapSnow 6.6
p29= [0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,
      0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.2,0.7] #fixedThermalCond_snow     |       0.3500 |       0.1000 |       1.0000

paramfile = Dataset("C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa1/summa_zParamTrial_variableDecayRate_sa_sa1.nc",'w',format='NETCDF3_CLASSIC') #create new paramtrail.nc file

hruidxID = list(np.arange(101,137))
hru_num = np.size(hruidxID)

#%% #create new paramtrail.nc file and adding vaiables to it --- summa_zParamTrial_variableDecayRate_test
hru = paramfile.createDimension('hru', None)
hidx = paramfile.createVariable('hruIndex', np.float64,('hru',)) # add hruIndex variable

param_nam_list = ['LAIMIN','LAIMAX','winterSAI','summerLAI','rootingDepth','heightCanopyTop',
                  'heightCanopyBottom','throughfallScaleSnow','newSnowDenMin','albedoDecayRate',
                  'albedoMaxVisible','albedoMinVisible','albedoMaxNearIR','albedoMinNearIR',
                  'albedoRefresh','albedoSootLoad','tempCritRain','z0Snow','mw_exp','Fcapil',
                  'k_snow','Frad_direct','Frad_vis','Louis79_bparam','Louis79_cStar',
                  'Mahrt87_eScale','windReductionParam','refInterceptCapSnow',
                  'fixedThermalCond_snow'] 
# call the function on the parameters
#valst1 = param_fill(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17) #,p18 

for param in param_nam_list:
    paramfile.createVariable(param, np.float64,('hru',))

constant_params = ['frozenPrecipMultip','rootDistExp','theta_sat','theta_res','vGn_alpha','vGn_n','k_soil','critSoilWilting','critSoilTranspire']
for params in constant_params:
    paramfile.createVariable(params, np.float64,('hru',))
#paramfile.close()
#%% parameterTrial, Local attributes and initial conditions for senatore beck
pt = Dataset('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/testVegFunctionsImpact/summa_zParamTrial_variableDecayRate.nc')
la = Dataset('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/testVegFunctionsImpact/summa_zLocalAttributes_senatorSheltered.nc') #('settings/wrrPaperTestCases/figure07/summa_zLocalAttributes_riparianAspen.nc')
ic = Dataset('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/testVegFunctionsImpact/summa_zInitialCond.nc') #('settings/wrrPaperTestCases/figure07/summa_zInitialCond.nc')
pt.variables['frozenPrecipMultip'][:]
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
paramfile.variables['LAIMIN'][:]=p1
paramfile.variables['LAIMAX'][:]=p2
paramfile.variables['winterSAI'][:]=p3
paramfile.variables['summerLAI'][:]=p4
paramfile.variables['rootingDepth'][:]=p5
paramfile.variables['heightCanopyTop'][:]=p6
paramfile.variables['heightCanopyBottom'][:]=p7
paramfile.variables['throughfallScaleSnow'][:]=p8
paramfile.variables['newSnowDenMin'][:]=p9

paramfile.variables['albedoDecayRate'][:]=p10
paramfile.variables['albedoMaxVisible'][:]=p11
paramfile.variables['albedoMinVisible'][:]=p12
paramfile.variables['albedoMaxNearIR'][:]=p13
paramfile.variables['albedoMinNearIR'][:]=p14
paramfile.variables['albedoRefresh'][:]=p15
paramfile.variables['albedoSootLoad'][:]=p16

paramfile.variables['tempCritRain'][:]=p17
paramfile.variables['z0Snow'][:]=p18
paramfile.variables['mw_exp'][:]=p19
paramfile.variables['Fcapil'][:]=p20
paramfile.variables['k_snow'][:]=p21
paramfile.variables['Frad_direct'][:]=p22
paramfile.variables['Frad_vis'][:]=p23
paramfile.variables['Louis79_bparam'][:]=p24
paramfile.variables['Louis79_cStar'][:]=p25
paramfile.variables['Mahrt87_eScale'][:]=p26
paramfile.variables['windReductionParam'][:]=p27
paramfile.variables['refInterceptCapSnow'][:]=p28
paramfile.variables['fixedThermalCond_snow'][:]=p29

#j = 0 
#for var in param_nam_list:
#    paramfile.variables[var][:]=valst1[j]
#    j=j+1
# don't forget the HRU Index!!
paramfile.variables['hruIndex'][:]=hruidxID

for varname in paramfile.variables.keys():
    var = paramfile.variables[varname]
    print varname, var.dtype, var.dimensions, var.shape

print paramfile.variables['hruIndex'][:]
paramfile.close()
#%% 
varcheck = Dataset ('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa1/summa_zParamTrial_variableDecayRate_sa_sa1.nc')
#print varcheck.variables['fixedThermalCond_snow'][:]
#print np.size(varcheck.variables['fixedThermalCond_snow'][:])

for varname in varcheck.variables.keys():
    var = paramfile.variables[varname]
    print varname, var.dtype, var.dimensions, var.shape

check2 =  varcheck.variables['hruIndex'][:]
#I checked it in Check.py code
#%% # local attributes file
# create a new localAtribute file ---- summa_zLocalAttributes_swampAngel_vtest
local_atrbt = Dataset("C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa1/summa_zLocalAttributes_swampAngel.nc",'w',format='NETCDF3_CLASSIC')
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
lacheck = Dataset('C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa1/summa_zLocalAttributes_swampAngel.nc')

print lacheck.variables['hruId'][:]
#for j in laCheck.variables:
#    print j
for varname in lacheck.variables.keys():
    var = lacheck.variables[varname]
    print (varname, var.dtype, var.dimensions, var.shape)    
#lacheck.close()
#%% # initial conditions file. summa_zInitialCond_vtest

in_condi = Dataset("C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa1/summa_zInitialCond.nc",'w',format='NETCDF3_CLASSIC')
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
iccheck = Dataset("C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa1/summa_zInitialCond.nc")
#for varname in iccheck.variables.keys():
#    var = iccheck.variables[varname]
#    print (varname, var.dtype, var.dimensions, var.shape)
print iccheck.variables['nSoil'][:]


