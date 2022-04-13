# Author: Cecile Coulon
#
# ---------------------------------- Readme -----------------------------------
#
# Modify the working directory and path to MDFLOW-2005
working_dir = './'
modflow_path = '/srv/common/Programmes/bin/mf2005'
#
#
# Run the script below
#
#----------------------------------- Script -----------------------------------
#
#%%------------------------- load paths and libraries -------------------------

# Set working directory
import os
os.chdir(working_dir)

# Libraries and function
import pickle, copy, pyemu, flopy
import pandas as pd
import numpy as np
from scipy.special import erfcinv
import functions_model_run as fmod

# Input & output files
path_model_outputs = 'model_outputs/' # Folder with model outputs
path_model_params = 'model_params/' # Folder with model inputs
mnw2_data_file = 'muni_wells_mnw2_data.csv' # MNW2 information
krig_factors_file = 'pp_30max.fac' # Kriging factors
hk_pp_file = 'hk1pp.dat' # hk values at pilot points
param_file = 'param.dat' # Other parameter values
qmuni_file = 'qmuni.dat' # qmuni values (= decision variables)

# Bin files generated with QGridder
path_bin_in = 'preproc_IDM_20m_v6.bin' #path_pp_zones_bin_in = 'pp_zones_v1.bin'

# Files containing the initial conditions for the pumping optimization
hds_file_init = 'idm_transient_swi_allwells_thiem_nopumping.hds'
bynd_file_init = 'idm_transient_swi_allwells_thiem_nopumping.bynd' 
zta_file_init = 'idm_transient_swi_allwells_thiem_nopumping.zta'

name_simu = 'idm_opt_run' # Model name
model_ws = './' # Path to save simulation files


#%%----------------------- Load GIS data from bin file ------------------------

# Retrieve data from the bin file generated with QGridder
with open(path_bin_in, 'rb') as handle:
     objects = pickle.load(handle)
     (nrow, ncol, delr, delc, ibound, sea, dem, geology, thk_valley, cx, cy, 
      muni_wells_row_col, muni_pumping, domestic_wells_row_col, ind_old_wells_row_col,
      obs_wells_row_col, obs_head, obs_zeta, tdem_row_col, ert_row_col) = objects

# Create row_col dictionaries for wells with head and zeta observations or forecasts
Dict_z = {key: value for key, value in obs_zeta.items() if value == 1}
obs_z_row_col = {key: value for key, value in obs_wells_row_col.items() if key in Dict_z.keys()}
obs_z_row_col.update(muni_wells_row_col)

Dict_h = {key: value for key, value in obs_head.items() if value == 1}
obs_h_row_col = {key: value for key, value in obs_wells_row_col.items() if key in Dict_h.keys()}
obs_h_row_col.update(muni_wells_row_col)


#%%----------------------------- Model parameters -----------------------------

# Read parameters
param_df = pd.read_csv(os.path.join(path_model_params, param_file), index_col=0, 
                       delim_whitespace=True, header=None, names=['name','value'])

hk_dunes = param_df.loc['hk_dunes', 'value']
hk_valley = param_df.loc['hk_valley', 'value']
hk_seabed = param_df.loc['hk_seabed', 'value']
rech_mmyr = param_df.loc['rech_mmyr', 'value']
alpha_T = param_df.loc['alpha_T', 'value']
sy = param_df.loc['sy', 'value']
ss = param_df.loc['ss', 'value']
ne = param_df.loc['ne', 'value']
slvl = param_df.loc['slvl', 'value']
alpha_L = param_df.loc['alpha_L', 'value'] # Longitudinal dispersivity (m)
M = param_df.loc['M', 'value'] # Initial width of the mixing zone (m)

# Create hk grid from hk values determined at pilot points ('hk1pp.dat' pp_file) & kriging factors ('pp.fac' factors_file)
hk_EDC_array = pyemu.utils.fac2real(pp_file = path_model_params + hk_pp_file, 
                                    factors_file = path_model_params + krig_factors_file, 
                                    out_file = None, fill_value = np.nan)

# Decision variables: Q at municipal wells
qmuni_df = pd.read_csv(os.path.join(path_model_params, qmuni_file), index_col = 0, 
                       delim_whitespace = True, header = None, names = ['name','value'])

#%%----------------- Get initial head & interface elevations ------------------
# Initial conditions: at the last timestep 

# Freshwater heads
hds = flopy.utils.binaryfile.HeadFile(os.path.join('./', hds_file_init))
times = hds.get_times()
h_init = hds.get_data(totim = times[-1])

# MNW2 data
bynd = pd.read_csv(os.path.join('./', bynd_file_init), header = 0,
                   delim_whitespace=True,  index_col=0, usecols = [0,5,7,8], 
                   names = ['name', 'totim', 'hwell', 'hcell'])
bynd.index = bynd.index.str.lower()
hwell_init = bynd['hwell'].tail(9).to_dict()

# Interface data
zta = flopy.utils.binaryfile.CellBudgetFile(os.path.join('./', zta_file_init))
kstpkper = zta.get_kstpkper()
zeta_init = zta.get_data(kstpkper = kstpkper[-1], text='ZETASRF  1')[0]

#%%---------------------------- Define parameters -----------------------------

# Discretization package
nlay = 1 # Number of layers (along height)
top= 100 # Top elevation of layer
botm = -300 # Bottom elevation
itmuni = 1 # Time units (1=seconds)
lenuni = 2 # Length units (2=meters)

nper = 1 # Number of stress periods
perlen_yr = 200 # Length of stress periods (years)
stplen_obj_day = 30 # Desired length of time steps (days)
steady = False # Type of simulation (steady or transient)
tsmult = 1 # Time step multiplier

# Layer-Property Flow package
laytyp = 1 # Convertible layer (T = f(head) throughout simulation)
thk_dunes = 10 # m Thickness of sand dunes

# General-Head Boundary package
seabed_thk = 150 # Seabed thickness (m) = aquifer thickness/2
rho_f = 1000 # Freshwater density (kg/m3)
rho_s = 1025 # Saltwater density (kg/m3)
epsilon = (rho_s - rho_f)/rho_f # Density ratio
corr_factor = 1 - ( alpha_T/(dem.mean()-botm) ) ** (1/4) # Density ratio correction factor (Lu & Werner 2013)
epsilon_corr = epsilon * corr_factor # Corrected density ratio

# Well package WEL
domestic_use_m3d = 80 # Total amount of pumped water from individual wells (Q3/day)

# Revised Multi-Node Well package MNW2
nnodes = 0 # Number of cells associated with the well
losstype = 'thiem' # Model for well loss
ppflag = 0 # Partial penetration flag

# Saltwater Intrusion package
nsolver = 2 # (2 = use PCG solver)
toeslope = 0.16
tipslope = toeslope
nu = [0, epsilon_corr] # [0, density ratio]
isource = np.zeros((nrow, ncol), np.int) # Sources/sinks have same fluid density as active zone at the top of the aquifer 
isource[ sea == 1 ] = -2 # Ocean bottom: infiltrating water = SW and exfiltrating water = same type as water at the top of the aquifer

#%%---- Process parameters

# Discretization package
perlen = int(perlen_yr*365.25*24*3600) # Convert perlen from years to seconds
stplen_obj = stplen_obj_day*24*3600 # Convert stplen_obj from days to seconds
nstp = round(perlen/stplen_obj) # Number of time steps = length of stress period / length of time steps
stplen = round( (perlen/nstp)/(24*3600) , 2) # Real length of time steps (days)

# Layer-Property Flow package
hk_array = hk_EDC_array # Initialize
hk_valleyH = ( hk_EDC_array * (dem-botm-thk_valley) + hk_valley * thk_valley ) / (dem-botm)
hk_dunesH = ( hk_EDC_array * (dem-botm-thk_dunes) + hk_dunes * thk_dunes ) / (dem-botm)
hk_array[geology == 3] = hk_valleyH[geology == 3] # Paleoglacial valley
hk_array[geology == 4] = hk_dunesH[geology == 4] # Sand dunes

# Recharge package
recharge_ms = rech_mmyr/(1000*365.25*24*3600) # Convert recharge values from mm/year to m/s
rech = np.zeros((nrow,ncol)) # Recharge grid
rech[ sea == 0 ] = recharge_ms # For onshore cells, assign recharge

# General-Head Boundary package
ghbc_EDC = (hk_seabed/seabed_thk) * delr * delc # Boundary conductance for EDC geology (m2/s)
ghb_cond = np.ones((nrow,ncol)) * ghbc_EDC # Default geology = EDC
ghb_stage = (epsilon+1) * slvl + (-epsilon) * dem # Stage (FW head at the ocean bottom) (m)
ghb_stage[sea != 1] = np.nan # GHB stage = nan for onshore cells

idx = np.logical_and(sea == 1, ibound == 1) # Boolean (indexes of active & ocean cells)
ghb_sp_data = [] # GHB grid
for row_col in np.argwhere(idx): # For indices where idx is True
    row, col = row_col
    ghb_sp_data.append( [0, row , col, ghb_stage[row,col], ghb_cond[row,col]] )
    # Each ghb cell defined by: [layer (here only 1), row, column, stage, conductance]

# Well package WEL
domestic_use_m3s = domestic_use_m3d/(24*3600) # Total domestic use: conversion from m3/day to m3/s
domestic_use_m3s_pc = domestic_use_m3s/len(domestic_wells_row_col) # Use per domestic well = total use/number of wells, in m3/s

wel_sp_data = {} # Build WEl dictionary = {nper: [layer, row, column, pumping rate]}
lay = 0 # Layer in which wells are pumping
for i in range(nper): # Compute for each stress period (here only 1)
    wells_data = []
    for well_id in domestic_wells_row_col.keys(): # Create one well_data array per well
        well_data = [lay] # Layers in which well are pumping
        well_data.append(domestic_wells_row_col[well_id][0][0]) # Append row to list
        well_data.append(domestic_wells_row_col[well_id][0][1]) # Append column to list
        well_data.append(-domestic_use_m3s_pc) # Append Q to list (m3/s, <0 for withdrawal)
        wells_data.append(well_data) # Append each individual well_data array to wells_data array
    wel_sp_data[i] = wells_data # Add wells_data array to WEL list   

# Revised Multi-Node Well package MNW2
node_data_df = pd.read_csv(os.path.join(path_model_params, mnw2_data_file)) # Table with MNW2 data by node
node_data_df['nnodes'] = nnodes
node_data_df['ppflag'] = ppflag
node_data_df['losstype'] = losstype
node_data = node_data_df.to_records() # Convert dataframe to a rec array for compatibility with flopy

# Stress period information for MNW2
per = [0] # List of stress periods
active_mnw2_wells = len(node_data_df) # Number of active MNW2 wells
wel_sp_data_mnw2_df = pd.DataFrame(list(zip(per*active_mnw2_wells, node_data_df.wellid, -qmuni_df.value)),      
               columns =['per', 'wellid', 'qdes']) # qdes = actual volumetric pumping rate at the well (m3/s, <0 for withdrawal)

pers = wel_sp_data_mnw2_df.groupby('per') # Group wel_sp_data_mnw2_df by periods
wel_sp_data_mnw2 = {i: pers.get_group(i).to_records() for i in range(nper)} # Convert df to dictionary

# Multi-Node Well Information Package MNWI
unit_mnwi, qndflag, qbhflag = 35, 0, 0 # Unit number for the output files, flags for writing additional flow information in the output files
mnwi_list = [list(x) for x in zip(node_data_df.wellid, [unit_mnwi]*active_mnw2_wells, \
             [qndflag]*active_mnw2_wells, [qbhflag]*active_mnw2_wells)]

# Output control
#spd = {(0, nstp-1): ['save head', 'save budget']} # Save head and budget at the end of the simulation only
output_nstp = 30 # Output frequency for zeta 
spd = {}
for kstp in range(0, nstp, output_nstp): # For istp ranging from 0 to nstp, with step = output_nstp
    spd[ (0, kstp) ] = ['save head']#, 'save budget']  # Creat dictionary = {(stress period i, timestep i): save head}

#%%------------------------------- Build model --------------------------------

ipakcb = None # Flag used to determine if cell-by-cell budget data should be saved

# New model
mf = flopy.modflow.Modflow(modelname = name_simu, model_ws = model_ws, 
                           exe_name = modflow_path) 

# Discretization package
discret = flopy.modflow.ModflowDis(mf, nlay=nlay, nrow=nrow, ncol=ncol, nper=nper, 
                                   delr=delr, delc=delc, top=top, botm=botm, 
                                   perlen=perlen, nstp=nstp, tsmult=tsmult, 
                                   steady=steady, itmuni=itmuni, lenuni=lenuni)
# Basic package
bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=h_init) 
# Recharge package
rch = flopy.modflow.ModflowRch(mf, rech = rech, ipakcb=ipakcb)
# Layer-Property Flow package ## adapted to pilot points
lpf = flopy.modflow.ModflowLpf(mf, laytyp=laytyp, hk=hk_array, sy=sy, ss=ss)
# General-Head Boundary package
ghb = flopy.modflow.ModflowGhb(mf, stress_period_data = ghb_sp_data, ipakcb=ipakcb)
# Well package
wel = flopy.modflow.ModflowWel(mf, stress_period_data = wel_sp_data, ipakcb=ipakcb)
# Revised Multi-Node Well package
mnw2 = flopy.modflow.ModflowMnw2(mf, mnwmax = active_mnw2_wells, node_data = node_data, 
                                 stress_period_data = wel_sp_data_mnw2, 
                                 itmp = [active_mnw2_wells] * nper,
                                 ipakcb = ipakcb)
# Multi-Node Well Information Package
mnwi = flopy.modflow.ModflowMnwi(mf, byndflag=36, mnwobs=active_mnw2_wells,
                                 wellid_unit_qndflag_qhbflag_concflag = mnwi_list)
# Output control package
oc = flopy.modflow.ModflowOc(mf, stress_period_data = spd)
# Preconditioned Conjugate-Gradient package
pcg = flopy.modflow.ModflowPcg(mf)
# Saltwater-Intrusion package
swi = flopy.modflow.ModflowSwi2(mf, nsrf=1, istrat=1, ipakcb=ipakcb, 
                                toeslope=toeslope, tipslope=tipslope, nu=nu, 
                                zeta=zeta_init, alpha=0.1, beta=0.1, ssz=ne, 
                                isource=isource, nsolver=nsolver, iswizt=55)
# Write input files
mf.write_input()

#%%-------------------------------- Run model ---------------------------------
mf.run_model(silent=False)

#%%------------------------------- Read outputs -------------------------------

#mf = flopy.modflow.Modflow.load(os.path.join(model_ws, name_simu + '.nam'), verbose=True, forgive=True)

# Freshwater heads

hds_swi = flopy.utils.binaryfile.HeadFile(os.path.join(model_ws, name_simu + '.hds'))
times = hds_swi.get_times() #[i/(365.25*24*3600) for i in times]
hswi = hds_swi.get_alldata()[:,:,:,:]

# MNW2 data
bynd = pd.read_csv(os.path.join(model_ws, name_simu + '.bynd'), header = 0,
                   delim_whitespace=True,  index_col=0, usecols = [0,5,7,8], 
                   names = ['name', 'totim', 'hwell', 'hcell'])
bynd.index = bynd.index.str.lower()
hwell_last = bynd['hwell'].tail(9).to_dict()

hwell_all = {} # Dictionary with {well: df with all timesteps for that well}
for well in muni_wells_row_col.keys():
    df = bynd.groupby(bynd.index).get_group(well)
    df.reset_index(drop=True, inplace=True)
    df = df[0::output_nstp] # MNW2 does not have the same timestep as HDS,
    #here is a good enough approximation (at 1e-4m) for plotting #df = df[df['totim'].isin(times)]
    hwell_all[well] = list(df['hwell']) #[i/(365.25*24*3600) for i in mnw2[well]['totim']]

# Interface data

zta = flopy.utils.binaryfile.CellBudgetFile(os.path.join(model_ws, name_simu + '.zta'))
kstpkper = zta.get_kstpkper() # List of unique stress periods & timesteps (nper,nstp)
zeta = []
for kk in kstpkper:
    zeta.append(zta.get_data(kstpkper=kk, text='ZETASRF  1')[0])
zeta = np.array(zeta)


#%% Compute the elevation of the 1% seawater salinity contour

pct = 0.01 # Percent seawater salinity (under fraction form, e.g. 1% is 0.01)
hcell_init, zcell_init, zwell_init, zbotm = {}, {}, {}, {}
hcell, hwell, zcell, zwell, z_1pct = {}, {}, {}, {}, {}

for well in muni_wells_row_col.keys():
    
    #---- Get row, col of the well & well bottom elevation
    
    row, col = muni_wells_row_col[well][0][0], muni_wells_row_col[well][0][1]
    
    zbotm[well] = node_data_df.loc[node_data_df.wellid == well, 'zbotm'].values[0]
    
    #---- Define hcell_init & zcell_init values (hwell_init already defined)
    
    hcell_init[well] = h_init[0, row, col]
    zcell_init[well] = zeta_init[0, row, col]
        
    #---- Define hcell, zcell values
    
    #===================== FOR PLOTTING
#    hcell[well] = hswi[:, 0, row, col]
#    zcell[well] = zeta[:, 0, row, col]
#    hwell[well] = hwell_all[well] 
    #=====================
    
    #===================== FOR OPTIMIZATION
    hcell[well] = hswi[-1, 0, row, col]
    zcell[well] = zeta[-1, 0, row, col]
    hwell[well] = hwell_last[well]
    #=====================
        
    #---- Correct for cell-to-well upconing: obtain zwell and zwell_init (Eq 4)
    
    zwell[well] = zcell[well] + (hcell[well] - hwell[well])/epsilon
    zwell_init[well] = zcell_init[well] + (hcell_init[well] - hwell_init[well])/epsilon
    
    #Note: hwell_init = hcell_init & zwell_init = zcell_init since there is no 
    #pumping in the initial conditions, so no cell-to-well drawdown or upconing
    
    #---- Correct for dispersion: obtain z_1pct (1% seawater salinity contour)
    
    b = ( 1 / ( 4 * erfcinv(0.048) ) )**2 # Eq 6
    
    z_1pct[well] = zwell[well] + 2 * np.sqrt( b * ( M**2 ) + alpha_L * np.abs( zwell[well] - zwell_init[well] ) ) * erfcinv(2 * pct) # Eq 5


#%%#===================== PLOT
#
#import matplotlib.pyplot as plt
#
## Interface
#for well in muni_wells_row_col.keys():#['muni1']:#
#    
#    fig = plt.figure(figsize=(7, 5))
#    plt.rc('font', family='serif', size=10)
#    ax = fig.add_subplot(1, 1, 1)
#
#    row, col = muni_wells_row_col[well][0][0], muni_wells_row_col[well][0][1]
#    
#    # zcell
#    ax.plot([i/(365.25*24*3600) for i in times], zcell[well], marker = '.', 
#              ls = 'solid', color = 'red', label = r'$\zeta_{\rm cell}$')
#    
#    # zwell
#    ax.plot([i/(365.25*24*3600) for i in times], zwell[well], marker = '.', 
#             ls = 'dashed', color = 'orange', label = r'$\zeta_{\rm well}$')
#    
#    # z_1pct
#    ax.plot([i/(365.25*24*3600) for i in times], z_1pct[well], marker = '.', 
#             ls = 'dashed', color = 'blue', label = r'$\zeta_{\rm 1 \%}$')
#    
#    # Well bottom elevation
#    ax.axhline(y = zbotm[well], ls = 'solid', color = 'k', 
#               label = r'$z_{\rm botm}$')
#    
#    plt.legend()
#    plt.xlabel('Years')
#    plt.ylabel('Elevation (masl)')
#    plt.tight_layout()
#    plt.savefig(os.path.join('figure_opt', 'zopt_' + well), dpi = 300)
#
## Head
#for well in muni_wells_row_col.keys():#['muni1']:#
#    
#    fig = plt.figure(figsize=(7, 5))
#    plt.rc('font', family='serif', size=10)
#    ax = fig.add_subplot(1, 1, 1)
#
#    row, col = muni_wells_row_col[well][0][0], muni_wells_row_col[well][0][1]
#    
#    # hcell
#    ax.plot([i/(365.25*24*3600) for i in times], hcell[well], marker = '.', 
#              ls = 'solid', color = 'red', label = r'$h_{\rm cell}$')
#    
#    # hwell
#    ax.plot([i/(365.25*24*3600) for i in times], hwell[well], marker = '.', 
#             ls = 'dashed', color = 'orange', label = r'$h_{\rm well}$')
#    
#    # Sea level
#    ax.axhline(y = slvl, ls = 'solid', color = 'k', 
#               label = 'Sea level')
#    
#    plt.legend()
#    plt.xlabel('Years')
#    plt.ylabel('Elevation (masl)')
#    plt.tight_layout()
#    plt.savefig(os.path.join('figure_opt', 'hopt_' + well), dpi = 300)
#
#=====================

#%%------------------------- Prepare PESTPP-OPT file --------------------------

tstep_save = -1 # We're interested in the new steady state (i.e. last timestep)


#---- Create dataframes with the data simulated at observation/forecast locations

# Head observations

hwells_df = fmod.sim_to_df(obs_h_row_col, hswi, tstep_save) # contains hcell values
for well in muni_wells_row_col.keys(): # Municipal wells: save hwell instead of hcell
    hwells_df.loc[hwells_df['name'] == well, 'value'] = hwell[well]


# Interface observations

ztdem_df = fmod.sim_to_df(tdem_row_col, zeta, tstep_save) # TDEM
zert_df = fmod.sim_to_df(ert_row_col, zeta, tstep_save) # ERT

zwells_all = fmod.sim_to_df(obs_z_row_col, zeta, tstep_save) # Wells: contains zcell values

zwells_df = zwells_all[zwells_all['name'].str.startswith('pz')] # Observations
zmuni_df = zwells_all[zwells_all['name'].str.startswith('muni')] # zcell forecasts

zmuni_pct_df = copy.deepcopy(zmuni_df) # z_1pct forecasts: save z_1pct instead of zcell
for well in zmuni_pct_df.name:
    zmuni_pct_df.loc[zmuni_pct_df.name == well, 'value'] = z_1pct[well]
#---- Drop several observations

# Inconsistent TDEM observations
ztdem_df.drop(ztdem_df[ (ztdem_df['name'] == 'tdem07') | 
        (ztdem_df['name'] == 'tdem23') | (ztdem_df['name'] == 'tdem67') | 
        (ztdem_df['name'] == 'tdem82') ].index, inplace=True)

# ERT observations close (< 100 m) to pumping wells
#zert_df.drop(zert_df[ (zert_df['name'] == 'zert08') | 
#        (zert_df['name'] == 'zert62') | (zert_df['name'] == 'zert71') | 
#        (zert_df['name'] == 'zert72') | (zert_df['name'] == 'zert73') | 
#        (zert_df['name'] == 'zert77') | (zert_df['name'] == 'zert78') | 
#        (zert_df['name'] == 'zert79') ].index, inplace=True)

    
#---- Assign observation name for the model

# Model observations
hwells_df.name = 'h' + hwells_df.name.astype(str)
zwells_df.name = 'z' + zwells_df.name.astype(str)
zmuni_df.name = 'z' + zmuni_df.name.astype(str)

# Model forecasts
zmuni_pct_df.name = 'z' + zmuni_pct_df.name.astype(str) + '_pct'
ztdem_df.name = 'z' + ztdem_df.name.astype(str)


#---- Export simulated data to .dat files

# Observations
fmod.write_df_to_dat(path_model_outputs, 'hwells', hwells_df[['name','value']])
fmod.write_df_to_dat(path_model_outputs, 'zwells', zwells_df[['name','value']])
fmod.write_df_to_dat(path_model_outputs, 'ztdem', ztdem_df[['name','value']])
fmod.write_df_to_dat(path_model_outputs, 'zert', zert_df[['name','value']])

# Forecasts
fmod.write_df_to_dat(path_model_outputs, 'zmuni', zmuni_df[['name','value']])
fmod.write_df_to_dat(path_model_outputs, 'zmuni_pct', zmuni_pct_df[['name','value']])
