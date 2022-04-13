# Author: Cecile Coulon
#
# ---------------------------------- Readme -----------------------------------
#
# Modify the working directory and path to MDFLOW-2005
working_dir = './'
modflow_path = 'mf2005'
#
#
# Run the script below without modifications to obtain Fig 5 of paper
#
#----------------------------------- Script -----------------------------------
#
#%%------------------------- load paths and libraries -------------------------

# Set working directory
import os
os.chdir(working_dir)

# Libraries and function
import pickle, pyemu, flopy, re
import pandas as pd
import numpy as np
from scipy.special import erfcinv
import matplotlib.pyplot as plt

# Input & output files
path_model_params = os.path.join('..', 'postcalib', 'model_params') # Folder with model inputs

mnw2_data_file = os.path.join(path_model_params, 'muni_wells_mnw2_data.csv') # MNW2 information
krig_factors_file = os.path.join(path_model_params, 'pp_30max.fac') # Kriging factors
hk_pp_file = os.path.join(path_model_params, 'hk1pp.dat') # hk values at pilot points
param_file = os.path.join(path_model_params, 'param.dat') # Other parameter values
qmuni_file = os.path.join(path_model_params, 'qmuni.dat') # qmuni values (= decision variables)

# Bin files generated with QGridder
bin_in = os.path.join('..', 'postcalib', 'preproc_IDM_20m_v6.bin')

# Files containing the initial conditions for the pumping optimization
hds_file_init = os.path.join('..', 'postcalib', 'idm_transient_swi_allwells_thiem_nopumping.hds')
bynd_file_init = os.path.join('..', 'postcalib', 'idm_transient_swi_allwells_thiem_nopumping.bynd')
zta_file_init = os.path.join('..', 'postcalib', 'idm_transient_swi_allwells_thiem_nopumping.zta')

model_ws = './' # Path to save simulation files


#%%----------------------- Load GIS data from bin file ------------------------

# Retrieve data from the bin file generated with QGridder
with open(bin_in, 'rb') as handle:
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
param_df = pd.read_csv(param_file, index_col = 0, delim_whitespace = True, 
                       header = None, names = ['name','value'])

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

node_data_df = pd.read_csv(mnw2_data_file) # Table with MNW2 data by node

# Create hk grid from hk values determined at pilot points ('hk1pp.dat' pp_file) & kriging factors ('pp.fac' factors_file)
hk_EDC_array = pyemu.utils.fac2real(pp_file = hk_pp_file, 
                                    factors_file = krig_factors_file, 
                                    out_file = None, fill_value = np.nan)

# Decision variables: Q at municipal wells
qmuni_df = pd.read_csv(qmuni_file, index_col = 0, delim_whitespace = True, 
                       header = None, names = ['name','value'])


#%%----------------- Get initial head & interface elevations ------------------
# Initial conditions: at the last timestep 

# Freshwater heads
hds = flopy.utils.binaryfile.HeadFile(hds_file_init)
times = hds.get_times()
h_init = hds.get_data(totim = times[-1])

# MNW2 data
bynd = pd.read_csv(bynd_file_init, header = 0, delim_whitespace=True, 
                   index_col = 0, usecols = [0,5,7,8], 
                   names = ['name', 'totim', 'hwell', 'hcell'])
bynd.index = bynd.index.str.lower()
hwell_init = bynd['hwell'].tail(9).to_dict()

# Interface data
zta = flopy.utils.binaryfile.CellBudgetFile(zta_file_init)
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
node_data_df['nnodes'] = nnodes
node_data_df['ppflag'] = ppflag
node_data_df['losstype'] = losstype
node_data = node_data_df.to_records() # Convert dataframe to a rec array for compatibility with flopy

# Stress period information for MNW2
per = [0] # List of stress periods
active_mnw2_wells = len(node_data_df) # Number of active MNW2 wells

# Multi-Node Well Information Package MNWI
unit_mnwi, qndflag, qbhflag = 35, 0, 0 # Unit number for the output files, flags for writing additional flow information in the output files
mnwi_list = [list(x) for x in zip(node_data_df.wellid, [unit_mnwi] * active_mnw2_wells, \
             [qndflag] * active_mnw2_wells, [qbhflag] * active_mnw2_wells)]

# Output control
#spd = {(0, nstp-1): ['save head', 'save budget']} # Save head and budget at the end of the simulation only
output_nstp = 30 # Output frequency for zeta 
spd = {}
for kstp in range(0, nstp, output_nstp): # For istp ranging from 0 to nstp, with step = output_nstp
    spd[ (0, kstp) ] = ['save head']#, 'save budget']  # Creat dictionary = {(stress period i, timestep i): save head}


#%% Iterate over pumping rates
 
Q = np.arange(0, 175, 25).tolist()

for q in Q:
    
    #---- Name of simulation
    
    name_simu = 'idm_opt_run'
    name_simu = name_simu + '_' + str(q)
    
    #---- Change pumping rate
    
    q = q / (24*3600)
    qmuni_df['value'] = q
    
    #---- Stress period information for MNW2
    
    wel_sp_data_mnw2_df = pd.DataFrame(list(zip(per*active_mnw2_wells, 
                                                node_data_df.wellid, 
                                                -qmuni_df.value)), 
                                       columns = ['per', 'wellid', 'qdes'])
    
    pers = wel_sp_data_mnw2_df.groupby('per') # Group wel_sp_data_mnw2_df by periods
    wel_sp_data_mnw2 = {i: pers.get_group(i).to_records() for i in range(nper)} # Convert df to dictionary

    #---- Build model
    
    # New model
    mf = flopy.modflow.Modflow(modelname = name_simu, model_ws = model_ws, 
                               exe_name = modflow_path) 
    ipakcb = None
    
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
    
    #---- Write input files
    
#    mf.write_input()
    
    #---- Run model
    
#    mf.run_model(silent=False)

#%%------------------------------- Read outputs -------------------------------

times_dic = {}
hswi_dic = {}
zeta_dic = {}
z_1pct_dic = {}


for q in Q:
    
    #---- Name of simulation
    name_simu = 'idm_opt_run'
    name_simu = name_simu + '_' + str(q)
    
    #---- Read outputs
    
    # Freshwater heads
    hds_swi = flopy.utils.binaryfile.HeadFile(os.path.join(model_ws, name_simu + '.hds'))
    times = hds_swi.get_times() #[i/(365.25*24*3600) for i in times]
    hswi = hds_swi.get_alldata()[:,:,:,:]
    hswi_dic[q] = hswi
    times_dic[q] = times
    
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
    zeta_dic[q] = zeta

    #---- Compute the elevation of the 1% seawater salinity contour
    pct = 0.01 # 1% seawater salinity
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
        hcell[well] = hswi[:, 0, row, col]
        zcell[well] = zeta[:, 0, row, col]
        hwell[well] = hwell_all[well] 
        #=====================
        
        #---- Correct for cell-to-well upconing: obtain zwell and zwell_init (Eq 4)
        
        zwell[well] = zcell[well] + (hcell[well] - hwell[well])/epsilon
        zwell_init[well] = zcell_init[well] + (hcell_init[well] - hwell_init[well])/epsilon
        
        #Note: hwell_init = hcell_init & zwell_init = zcell_init since there is no 
        #pumping in the initial conditions, so no cell-to-well drawdown or upconing
        
        #---- Correct for dispersion: obtain z_1pct (1% seawater salinity contour)
        
        b = ( 1 / ( 4 * erfcinv(0.048) ) )**2 # Eq 6
        
        z_1pct[well] = zwell[well] + 2 * np.sqrt( b * ( M**2 ) + alpha_L * np.abs( zwell[well] - zwell_init[well] ) ) * erfcinv(2 * pct) # Eq 5
    z_1pct_dic[q] = z_1pct


#%%---------------------------- Plotting functions ----------------------------

def cm2inch(*tupl):
    ''' From a tuple containing centimeters, return a tuple containing inches '''
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

# Common export parameters
fontsze = 11
plt.rcParams['font.size'] = fontsze
plt.rc('font',**{'family':'sans-serif', 'sans-serif':['Arial']})


#%%------------------------- Plot z_1pct vs Q: Fig 5 --------------------------

# Set up figure

figsze = cm2inch(13,11)
fig, ax = plt.subplots(1, 1, figsize=figsze)
fig.subplots_adjust(wspace=0.2, hspace=0.2, left=0.125, right=1, bottom=0, top=1)

colors_dict = {list(z_1pct_dic.keys())[i]: 
    plt.rcParams['axes.prop_cycle'].by_key()['color'][i] for i in range(len(z_1pct_dic.keys()))} # 1 color/test

# Plot
for well in muni_wells_row_col.keys():
    
    well_no = re.findall(r'\d+', well)[0]
    row, col = muni_wells_row_col[well][0][0], muni_wells_row_col[well][0][1]
    zsal1_plot = []   
    for key in times_dic.keys():
        zsal1_plot.append(z_1pct_dic[key][well][-1])
    
    # Plot Q vs steady-state interface elevation    
    plt.plot(Q[:-1], zsal1_plot[:-1], marker='o', lw = 1.5, label = 'Well no. ' + well_no)

plt.legend(ncol = 1)
ax.set_xlabel('Pumping rate (m$^{3}$/day)')
ax.set_ylabel(r'Steady-state elevation of $\zeta_{\rm 1\%}$ (masl)')
plt.tight_layout()
plt.savefig('Fig5-linearity.jpeg', dpi = 600)
