from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np

#iccheck1 = Dataset ('c:/Users/HHS/summaTestCases_2.x/settings/wrrPaperTestCases/figure06/summa_zLocalAttributes_senatorSheltered.nc')
#for icvarname1 in iccheck1.variables:
#    print icvarname1
#
#print iccheck1.variables['hruId'][:]
varcheck = Dataset ('summa_zParamTrial_variableDecayRate_SA_sa1.nc')
print np.size(varcheck.variables['winterSAI'][:])


#%%
lacheck1 = Dataset('summa_zLocalAttributes_senatorSheltered.nc')
for lavarname1 in lacheck1.variables:
    print lavarname1

print lacheck1.variables['albedoMaxVisible'][:]


#lacheck1 = Dataset('c:/Users/HHS/summaTestCases_2.x/settings/wrrPaperTestCases/figure06/summa_zParamTrial_variableDecayRate.nc')
#for lavarname1 in lacheck1.variables:
#    print lavarname1
#
#print lacheck1.variables['z0Snow'][:]
#
#print "***********************"



#%%
#ptcheck1 = Dataset ('c:/Users/HHS/summaTestCases_2.x/settings/wrrPaperTestCases/figure06/summa_zInitialCond6.nc')
#for ptvarname1 in ptcheck1.variables:
#    print ptvarname1
#
#print ptcheck1.variables['mLayerTemp'][:]
#
#print "***********************"

#ptcheck2 = Dataset ('0summa_zInitialCond0.nc')
#for ptvarname2 in ptcheck2.variables:
#    print ptvarname2
#
#print ptcheck2.variables['mLayerTemp'][:]

#%%
#ptcheck5 = Dataset ('c:/Users/HHS/summaTestCases_2.x/testCases_data/inputData/fieldData/senatorBeck/SenatorBeck_forcing6.nc')
#for ptvarname1 in ptcheck5.variables:
#    print ptvarname1
#
#print ptcheck5.variables['latitude'][:]
#
#print "***********************"


#%%
#lacheck1 = Dataset('senatorBeck_SASP_1hr.nc')
#for lavarname1 in lacheck1.variables:
#    print lavarname1
#
#print np.size(lacheck1.variables['snowDepth'][:])
#
#lacheck1.variables['time'][:]
##wstest = testfd.variables['windspd'][:]
##plt.plot(timetest,wstest)
##plt.show()






#%%
#SWE = check3.variables['scalarSWE'][:]
#timec = check3.variables['time'][:]
#
#SWE0 = check3.variables['scalarSWE'][:,0]
#SWE1 = check3.variables['scalarSWE'][:,5]
#difSWE = SWE1 - SWE0
#plt.plot(timec,SWE[:,1])
#plt.show()



#p1 = [0.75, 0.84, 0.91] #albedoMax_ConsDR_VarDR      |       0.8400 |       0.7000 |       0.9500 $$$$$$ 2: 0.75 | 0.92
#p2 = [0.45, 0.55, 0.8] #albedoMinWinter   |       0.5500 |       0.6000 |       1.0000 $$$$$$ 2: 0.65 | 0.90
#p3 = [0.4, 0.7] #albedoMinSpring    |       0.5500 |       0.3000 |       1.0000 $$$$$$ 2: 0.40 | 0.90
#p4 = [0.55, 0.71] #albedoMaxNearIR   |       0.6500 |       0.5000 |       0.7500 $$$$$$ 2: 0.55 | 0.70
#p5 = [0.25, 0.4] #albedoMinNearIR  |       0.3000 |       0.1500 |       0.4500 $$$$$$ 2: 0.20 | 0.43

#hruidxID = [1001, 1002, 1003, 1004, 1005, 1006]
#hru_num = 6
#
## %store hruidxID
##%%
#param_nam_list = ['albedoMax', 'albedoMinWinter'] 
#
##%% #create new paramtrail.nc file for constantDecayRate model
#paramfile_c = Dataset("summa_zParamTrial_constantDecayRate_SA.nc",'w',format='NETCDF3_CLASSIC') #create new paramtrail.nc file
#hru = paramfile_c.createDimension('hru', None)
#hidx = paramfile_c.createVariable('hruIndex', np.float64,('hru',)) # add hruIndex variable
#
#for param in param_nam_list:
#    paramfile_c.createVariable(param, np.float64,('hru',))
#
#constant_params = ['albedoDecayRate', 'frozenPrecipMultip', 'rootingDepth', 'rootDistExp', 'theta_sat','theta_res','vGn_alpha','vGn_n','k_soil','critSoilWilting','critSoilTranspire','winterSAI','summerLAI','heightCanopyTop','heightCanopyBottom']
#for params in constant_params:
#    paramfile_c.createVariable(params, np.float64,('hru',))



































