#%% Snow surface temperature observation data
with open("C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa2_bestSweSD/sst_1hr_2006-2008.csv") as sast:
    reader2 = csv.reader(sast)
    raw_sst = [st for st in reader2]
sa_sst_column1 = []
for csv_counter3 in range (len (raw_sst)):
    for csv_counter4 in range (6):
        sa_sst_column1.append(raw_sst[csv_counter3][csv_counter4])
sa_sst=np.reshape(sa_sst_column1,(len (raw_sst),6))
sa_sst_obs=[float(val2) for val2 in sa_sst[1:len(raw_sst),5]]
sa_sst_obs[:][0:4567:12]
#sa_sd_obs = [float(value) for value in sa_snowdepth1]
sa_sst_obs_date = pd.DatetimeIndex(sa_sst[1:len(raw_sst),0])

sst_obs_df0 = pd.DataFrame(sa_sst_obs, columns = ['observed_snowSurfaceTemp']) 
sst_obs_df0.set_index(sa_sst_obs_date,inplace=True)

#%% #Swamp Angel forcing data
#swampangel_forcing = open('swamp_angel_forcingdata2_corrected.csv', 'rb')
#sa_forcing = csv.reader(swampangel_forcing)#, delimiter=',')
with open("C:/1UNRuniversityFolder/Dissertation/Chapter 1-Snowmelt/swamp_angel/sa_sa2_vars/sa2_bestSweSD/swamp_angel_forcingdata2_corrected_precipCalib_final_L.csv") as safd:
    reader = csv.reader(safd)
    data_forcing = [r for r in reader]
data_forcing2 = data_forcing[1:]
sa_fd_column = []
for csv_counter1 in range (len (data_forcing2)):
    for csv_counter2 in range (11):
        sa_fd_column.append(float(data_forcing2[csv_counter1][csv_counter2]))
sa_forcing=np.reshape(sa_fd_column,(len (data_forcing2),11))

#%%correction of lst observations
h = 6.626 * 10 ** (-34.) #m2kg/s 
c = 3. * 10 ** 8 #m/s c is the speed of light (c = 3 × 108m/s),
landa = 11 * 10 ** (-6)        #is the wavelength of the radiation, and 
kB= 1.3806 * 10 ** (-23.)   #kBis the Boltzmann con-stant (m2kg/(s2K))
CI1 = (2 * h * c ** 2)/ (landa ** 5)
CI2 = h * c / (landa *kB)

temp_obj = sst_obs_df0['observed_snowSurfaceTemp']+273.15
I_landa_T_obj_tot = CI1/(np.exp(CI2/temp_obj)-1)
fie_tot = I_landa_T_obj_tot.copy()

temp_air = sa_forcing[720:15636,7]
I_landa_T_air = CI1/(np.exp(CI2/temp_air)-1)
fie_air = I_landa_T_air.copy()

LWRin = sa_forcing[720:15636,6]
temp_sky = (LWRin/(5.67*(10**-8)))**0.25
I_landa_T_sky = CI1/(np.exp(CI2/temp_sky)-1)
fie_sky = I_landa_T_sky.copy()

pres_atm = sa_forcing[720:15636,9]
hum_sp = sa_forcing[720:15636,10]
e_t = (pres_atm * hum_sp)/0.622
p_da = pres_atm - e_t
e_star_t = 611*(np.exp((17.27*(temp_air-273.15))/(temp_air-273.15+237.3)))
rh = e_t/e_star_t

d = 1 #m
temp_air_c = temp_air - 273.16

h_d_rh = (0.000166667* temp_air_c** 3.+ 0.01 * temp_air_c**2. + 0.383333*temp_air_c + 5)*rh*d

table1 = [0.998,0.994,0.988,0.975,0.940,0.883]
h_mm_km = [0.2,0.5,1,2,5,10]

pH2O = []
for values in h_d_rh:
    if values<=0.35 :
        pH2O.append(0.998)
    if values>0.35 and values<=0.75:
        pH2O.append(0.994)
    if values>0.75 and values<=1.5:
        pH2O.append(0.988)
    if values>1.5 and values<=3.5:
        pH2O.append(0.975)
    if values>3.5 and values<=7.5:
        pH2O.append(0.940)
    if values>7.5: 
        pH2O.append(0.883)

table2 = [1.,1.,0.999,0.997,0.994,0.989]

pCO2 = []
for values2 in h_d_rh:
    if values2<=0.75 :
        pCO2.append(1.)
    if values2>0.75 and values2<=1.5:
        pCO2.append(0.999)
    if values2>1.5 and values2<=3.5:
        pCO2.append(0.997)
    if values2>3.5 and values2<=7.5:
        pCO2.append(0.994)
    if values2>7.5: 
        pCO2.append(0.989)

p_atm = np.array(pH2O) * np.array(pCO2)
tau_atm = p_atm.copy()

sigma_snow = 0.98
sigma_sky = 1.

fie_leaf = (1/(sigma_snow*tau_atm))*(fie_tot - tau_atm*(1-sigma_snow)*sigma_sky*fie_sky - (1-tau_atm)*fie_air)

I_landa_T_leaf = fie_leaf.copy()

T_leaf_correct = CI2/(np.log(1+(CI1/I_landa_T_leaf)))
bias_T_obj = T_leaf_correct - temp_obj

sst_obs_df = pd.DataFrame(T_leaf_correct.copy(), columns = ['observed_snowSurfaceTemp'])
