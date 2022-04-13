# Author: Cecile Coulon
#
# ---------------------------------- Readme -----------------------------------
#
# Modify the working directory
working_dir = './'
#
#
# Run the script below without modifications to obtain Figs 7, 8, 9, 10 of paper
#
#----------------------------------- Script -----------------------------------
#
#%%------------------------- load paths and libraries -------------------------

# Set working directory
import os
os.chdir(working_dir)

# Libraries and function
import pyemu, glob, flopy, re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erfcinv

# Input & output files

name_pst = 'pestpp_opt' # Name of post calibration PESTPP-OPT run
name_pst_priorunc = 'pestpp_opt_priorunc' # Name of prior uncertainty PESTPP-OPT run

muni_data = pd.read_csv(os.path.join('..', 'pestpp-opt files', 'postcalib files', 
                                     'model_params', 'muni_wells_mnw2_data.csv'), index_col = 3)

current_Q_m3day = pd.read_csv('Current_max_pumping_rates.csv', index_col = 0)
current_Q_m3day.loc['total', :] = current_Q_m3day.sum(axis=0)

zta_file_init = os.path.join('..', 'pestpp-opt files', 'postcalib files', 
                             'idm_transient_swi_allwells_thiem_nopumping.zta')

M = pd.read_csv(os.path.join('..', 'pestpp-opt files', 'postcalib files', 
                             'model_params', 'param.dat'), index_col = 0, 
                delim_whitespace = True, header = None, names = ['name','value']).loc['M', 'value']

pst_file = os.path.join('..', 'pestpp-opt files', 'postcalib files', '0_50_pestpp_opt.pst')


# Compute initial steady state interface elevations without pumping

zta = flopy.utils.binaryfile.CellBudgetFile(zta_file_init)
kstpkper = zta.get_kstpkper()
zeta_init = zta.get_data(kstpkper = kstpkper[-1], text = 'ZETASRF  1')[0]

zcell_init, z_1pct_init = {}, {}

for well in muni_data.index:
    
    row, col = muni_data.loc[well, 'i'], muni_data.loc[well, 'j']
    well = well.replace('_pct', '')
    zcell_init[well] = zeta_init[0, row, col]
    zwell_init = zcell_init # no pumping: zwell = zcell
    
    # Correct for dispersion: obtain z_1pct (1% seawater salinity contour)
    b = ( 1 / ( 4 * erfcinv(0.048) ) )**2 # Eq 6
    z_1pct_init[well] = zwell_init[well] + 2 * np.sqrt( b * ( M**2 ) ) * erfcinv(2 * 0.01) # Eq 5 with zwell = zwell_init


#%%------------------------- Read PEST control files --------------------------

# Post calibration risk values
list_risk = sorted([0.01] + list(np.arange(0.05, 1.00, 0.05)) + [0.96] + [0.97] + [0.98])
list_risk = ['%.2f' % x for x in list_risk]

# Prior uncertainty risk values
list_risk2 = sorted(list(np.arange(0.15, 0.60, 0.05)) + [0.56])
list_risk2 = ['%.2f' % x for x in list_risk2]

# Lists of names
pst_0_50 = pyemu.Pst(pst_file)
dvar_nme = [x for x in pst_0_50.par_names if x.startswith('qmuni')] # decision variables
c_nme = [x for x in pst_0_50.obs_names if x.endswith('pct')] # constraints
zmuni = [x for x in pst_0_50.obs_names if x.startswith('zmuni') and x not in c_nme] # zmuni values


#%%-------------------- Retrieve optimal decision variable --------------------
#-------------------- & decision objective function values --------------------

def get_opt_dvars(name_pst, folder, dvar_nme, list_risk):
    
    ''' Create a dataframe containing optimal decision variables & decision 
    objective function for each risk stance '''
    
    par_dic = {}
    
    # Retrieve decision vars from all .par files
    
    for risk in list_risk:
        
        parfile = os.path.join(folder, risk.replace('.', '_') + '_' + name_pst + '.par')
        par_df = pyemu.pst_utils.read_parfile(parfile)
        par_df = par_df.loc[dvar_nme, :]
        
        # When the solution is infeasible, pestpp-opt writes extreme negative values to the par file:
        if par_df.parval1.sum() < 0: 
            print('infeasible at risk ' + risk)
            continue
        
        par_dic[risk] = par_df.parval1
    
    # Process the dvar dictionary for plotting
    
    par_df = pd.concat(par_dic, axis = 1).T
    par_df.index = [int(float(idx) * 100) for idx in par_df.index]
    par_df['phi'] = par_df.sum(axis=1) # Compute decision objective function
    
    return par_df


par_df = get_opt_dvars(name_pst, 'postcalib files', dvar_nme, list_risk) # post calibration
par_df_priorunc = get_opt_dvars(name_pst_priorunc, 'priorunc files', dvar_nme, list_risk2) # prior uncertainty


# Convert m3/s to m3/day
par_df_m3day = par_df * 24 * 3600
par_df_m3day_priorunc = par_df_priorunc * 24 * 3600


#%%------------------------ Retrieve constraint values ------------------------

def get_opt_cons(name_pst, folder, list_outputs, dvar_nme, list_risk):
    
    ''' Create a dataframe containing constraints at the end of the 
    optimization, for each risk stance '''
    
    res_dic = {}
    
    # Retrieve constraints from the last .sim.rei files
    
    for risk in list_risk:        
        
        # When the solution is infeasible, pestpp-opt writes extreme negative values to the par file:
        parfile = os.path.join(folder, risk.replace('.', '_') + '_' + name_pst + '.par')
        par_df = pyemu.pst_utils.read_parfile(parfile)
        par_df = par_df.loc[dvar_nme, :]
        if par_df.parval1.sum() < 0: 
            print('infeasible at risk ' + risk)
            continue
        
        res_files_list = glob.glob(
        os.path.join(folder, risk.replace('.', '_') + '_' + name_pst + '.?.sim.rei'))
                
        resfile = res_files_list[-1]
        res_df = pyemu.pst_utils.read_resfile(resfile)
        res_df = res_df.loc[list_outputs, :]
        
        res_dic[risk] = res_df.modelled
        
    # Process the dictionaries for plotting
    
    res_df = pd.concat(res_dic, axis = 1).T
    res_df.index = [int(float(idx) * 100) for idx in res_df.index]
    
    return res_df


res_df_cons = get_opt_cons(name_pst, 'postcalib files', c_nme, dvar_nme, list_risk)
res_df_zmuni = get_opt_cons(name_pst, 'postcalib files', zmuni, dvar_nme, list_risk)

res_df_cons_priorunc = get_opt_cons(name_pst_priorunc, 'priorunc files', c_nme, dvar_nme, list_risk2)
res_df_zmuni_priorunc = get_opt_cons(name_pst_priorunc, 'priorunc files', zmuni, dvar_nme, list_risk2)


#%%------------------------ Retrieve constraint stdevs ------------------------

def get_opt_stdevs(name_pst, folder, list_outputs, dvar_nme, list_risk):
    
    ''' Create a dataframe containing constraint standard deviations at the end of the 
    optimization, for each risk stance '''
    
    # Drop dvars from prior param covariance matrix
    parcov_file = os.path.join('..', 'pestpp-opt files', folder, 'prior_cov_matrix.jcb')
    parcov_mat = pyemu.Jco.from_binary(parcov_file)
    parcov_mat.drop(dvar_nme, axis = 0) # remove from rows
    parcov_mat.drop(dvar_nme, axis = 1) # remove from cols
    parcov_nodv = os.path.join('postproc', 'parcov_' + folder + '.nodvars.jcb')
    parcov_mat.to_binary(parcov_nodv)
    
    
    stdev_prior_dic = {}
    stdev_post_dic = {}
    
    for risk in list_risk:
        
        # When the solution is infeasible, pestpp-opt writes extreme negative values to the par file:
        parfile = os.path.join(folder, risk.replace('.', '_') + '_' + name_pst + '.par')
        par_df = pyemu.pst_utils.read_parfile(parfile)
        par_df = par_df.loc[dvar_nme, :]
        if par_df.parval1.sum() < 0: 
            print('infeasible at risk ' + risk)
            continue
        
        
        if risk != '0.50':
            
            # Get all jacobian matrices from .jcb files
            
            jcb_files_list = glob.glob(
                    os.path.join(folder, risk.replace('.', '_') + '_' + name_pst + '.?.jcb'))
                        
            # Drop dvars from the jacobian matrices & export
            
            jcb_file = jcb_files_list[-1]
            jcb_mat = pyemu.Jco.from_binary(jcb_file)
            jcb_mat.drop(dvar_nme, axis = 1) # remove dvars from cols
            jcb_nodv = os.path.join('postproc', jcb_file.split('\\')[1].replace('.jcb', '.nodvar.jcb'))
            jcb_mat.to_binary(jcb_nodv)
            
            # Linear analysis
            
            pst = pyemu.Pst(os.path.join(folder, risk.replace('.', '_') + '_' + name_pst + '.pst'))
    
            
            la = pyemu.Schur(jco = jcb_nodv, pst = pst, 
                             parcov = parcov_nodv, obscov = None, 
                             forecasts = c_nme, sigma_range = 4.0)
            
            # Get prior & posterior constraint stdevs
            
            for_sum = la.get_forecast_summary() # variances
            for_sum['prior_stdev'] = np.sqrt(for_sum.prior_var) # stdevs
            for_sum['post_stdev'] = np.sqrt(for_sum.post_var) # stdevs
            
            # Save all the constraint stdevs from the risk stance to dictionary
            
            for_sum.drop(columns = ['prior_var', 'post_var', 'percent_reduction'],
                         inplace = True)
                    
            # Save only the final constraint stdevs to dictionary
            risk_value = int(float(risk) * 100)
            stdev_prior_dic[risk_value] = for_sum['prior_stdev']
            stdev_post_dic[risk_value] = for_sum['post_stdev']
        
        
        elif risk == '0.50': # no FOSM analysis for risk = 0.5
            risk_value = int(float(risk) * 100)
            stdev_prior_dic[risk_value] = pd.DataFrame({'prior_stdev': 0}, index = c_nme)['prior_stdev']
            stdev_post_dic[risk_value] = pd.DataFrame({'post_stdev': 0}, index = c_nme)['post_stdev']
        
        # Process the dictionaries for plotting
        
        stdev_prior_df = pd.concat(stdev_prior_dic, axis = 1).T
        stdev_post_df = pd.concat(stdev_post_dic, axis = 1).T
    
    return stdev_post_df
#    return stdev_prior_df, stdev_post_df

stdev_post_df = get_opt_stdevs(name_pst, 'postcalib files', c_nme, dvar_nme, list_risk)
stdev_post_df_priorunc = get_opt_stdevs(name_pst_priorunc, 'priorunc files', c_nme, dvar_nme, list_risk2)


#%%---------------------------- Plotting functions ----------------------------

def cm2inch(*tupl):
    ''' From a tuple containing centimeters, return a tuple containing inches '''
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def gaussian_distribution_modif(mean, stdev, num_stdevs, num_pts = 50):
    ''' Modified version of pyemu.plot_utils.gaussian_distribution:
        plot Gaussian distribution as the mean +/- num_stdevs standard deviations '''
    
    xstart = mean - (num_stdevs * stdev)
    xend = mean + (num_stdevs * stdev)
    
    x = np.linspace(xstart, xend, num_pts)
    y = (1.0 / np.sqrt(2.0 * np.pi * stdev * stdev)) * np.exp(
        -1.0 * ((x - mean) ** 2) / (2.0 * stdev * stdev))
    
    return x, y

# x-axis for all figures
xaxis = [1] + list(np.arange(5, 100, 5)) + [99]

# Common export parameters
fontsze = 11
plt.rcParams['font.size'] = fontsze
plt.rc('font',**{'family':'sans-serif', 'sans-serif':['Arial']})


#%%---------- Plot decision objective function value vs risk: Fig 7 -----------

# Set up figure

figsze = cm2inch(19,11)
fig, ax = plt.subplots(1, 1, figsize=figsze)
fig.subplots_adjust(wspace=0.2, hspace=0.2, left=0.125, right=1, bottom=0, top=1)

# Optimal Q with prior uncertainty

ax.plot(par_df_m3day_priorunc.index.values, par_df_m3day_priorunc.phi, 
        marker = '.', ms = 7, lw = 1, ls = '--', c = 'tab:orange',
        label = r'Optimization with prior $\zeta_{1\%}$ uncertainty')

# Optimal Q with post uncertainty

ax.plot(par_df_m3day.index.values, par_df_m3day.phi, 
        marker = '.', ms = 7, lw = 1, ls = '--', c = 'tab:blue',
        label = r'Optimization with posterior $\zeta_{1\%}$ uncertainty')

# Current Q

ax.plot(50, current_Q_m3day.loc['total', 'Qm3_day'], 
        marker = 'x', ms = 7, ls = 'none', c = 'k',
        label = r'Current maximum allowable pumping')

# Current water demand

ax.axhline(y = 110, #(MELCC) vs (181 - Alexe)
           ls = '-', color = 'grey', 
           label = r'Current water demand')


# x-axis options

ax.set_xticks(xaxis)
ax.set_xticklabels( [str(x) for x in xaxis] )
ax.set_xlim([0, 100])
ax.set_xlabel('Reliability of the optimization (%)')
ax2 = ax.twiny()
ax2.set_xticks(xaxis)
ax2.set_xticklabels( [str( int(100-x) ) for x in xaxis] )
ax2.set_xlim([0, 100])
ax2.set_xlabel('Risk of well salinization (%)')

# Others

ax.legend(loc = 'upper left', bbox_to_anchor = (0.38, 0.9))
ax.grid(axis = 'both', lw = 0.3)
ax2.set_ylim([0, 1300])
ax.set_ylabel(r'Total pumping rate (m$^{3}$/day)')
plt.tight_layout()
plt.savefig(os.path.join('Fig7-phi.jpeg'), dpi = 600)


#%%---------- Plot constraint + stdev at the max reliability: Fig 8 -----------

cons = 'zmuni6_pct' # Choose which constraint to plot

# Set up figure

figsze = cm2inch(9.5,8.5)
fig, ax = plt.subplots(1, 1, figsize=figsze)
fig.subplots_adjust(wspace=0.2, hspace=0.2, left=0.125, right=1, bottom=0, top=1)

colors_dict = {c_nme[i]: 
    plt.rcParams['axes.prop_cycle'].by_key()['color'][i] for i in range(len(c_nme))}

# Plot prior distribution (Gaussian)

mean = res_df_cons_priorunc.tail(1)[cons].values[0]
stdev = stdev_post_df_priorunc.tail(1)[cons].values[0]
reliab = stdev_post_df_priorunc.tail(1).index[0]
qtot = par_df_m3day_priorunc.loc[reliab, 'phi']
label = r'Prior $\zeta_{1\%}$ uncertainty'
x1, y1 = gaussian_distribution_modif(mean, stdev, num_stdevs = 3)
ax.plot(y1, x1, ls = 'dashed', color = colors_dict[cons], label = label)

# Plot posterior distribution (Gaussian)

mean = res_df_cons.tail(1)[cons].values[0]
stdev = stdev_post_df.tail(1)[cons].values[0]
reliab = stdev_post_df.tail(1).index[0]
qtot = par_df_m3day.loc[reliab, 'phi']
label = r'Posterior $\zeta_{1\%}$ uncertainty'
x2, y2 = gaussian_distribution_modif(mean, stdev, num_stdevs = 3)
ax.fill_betweenx(x2, y2, 0, alpha = 0.5, color = colors_dict[cons], label = label)

# Well bottom elevation

well_no = re.findall(r'\d+', cons)[0]
ax.axhline(y = muni_data.loc['muni' + well_no, 'zbotm'], 
           lw = 0.75, ls = '-', color = 'k', 
           label = 'Well bottom')   
 
# x-axis options

Max = max(y1.max(), y2.max())
ax.set_xlim(0, Max + Max * 0.05)
ax.set_xlabel('Probability density')

# Others

handles, labels = ax.get_legend_handles_labels()
list_labels = sorted(zip(labels, handles), key=lambda t: t[0])
labels, handles = zip(*list_labels)
ax.legend(handles, labels)

ax.set_ylabel(r'Elevation of $\zeta_{1\%}$ (masl)')
plt.tight_layout()
plt.savefig(os.path.join('Fig8-uncer_' + cons + '.jpeg'), dpi = 600)


#%%------------------ Plot decision variables vs risk: Fig 9 ------------------

# Choose which dvars to plot:

# Files from the post calibration PESTPP-OPT run:
par_m3day = par_df_m3day
nmefig = 'post_'

# Or files from theprior uncertainty PESTPP-OPT run:
#par_m3day = par_df_m3day_priorunc
#nmefig = 'prior_'

# Set up figure

figsze = cm2inch(19,11)
fig, ax = plt.subplots(1, 1, figsize=figsze)
fig.subplots_adjust(wspace=0.2, hspace=0.2, left=0.125, right=1, bottom=0, top=1)

colors_dict = {dvar_nme[i]: 
    plt.rcParams['axes.prop_cycle'].by_key()['color'][i] for i in range(len(dvar_nme))}

# Plot dvars

for dvar in dvar_nme:
    
    well_no = re.findall(r'\d+', dvar)[0]
    
    # Optimal Q
    ax.plot(par_m3day.index.values, par_m3day.loc[:, dvar], 
        marker = '.', ms = 7, lw = 1, ls = '--', c = colors_dict[dvar],
        label = dvar.replace('qmuni', '$Q$').replace('1', '$_{1}$').replace('2', '$_{2}$')\
            .replace('3', '$_{3}$').replace('4', '$_{4}$').replace('5', '$_{5}$')\
            .replace('6', '$_{6}$').replace('7', '$_{7}$').replace('8', '$_{8}$')\
            .replace('9', '$_{9}$'))

    # Current Q
    ax.plot(50, current_Q_m3day.loc['muni' + well_no, 'Qm3_day'], 
            marker = 'x', ms = 7, ls = 'none', c = colors_dict[dvar], 
            label = '_')

# x-axis options

ax.set_xticks(xaxis)
ax.set_xticklabels( [str(x) for x in xaxis] )
ax.set_xlim([0, 100])
ax.set_xlabel('Reliability of the optimization (%)')
ax2 = ax.twiny()
ax2.set_xticks(xaxis)
ax2.set_xticklabels( [str( int(100-x) ) for x in xaxis] )
ax2.set_xlim([0, 100])
ax2.set_xlabel('Risk of well salinization (%)')

# Others

ax.legend(ncol = 3)
ax.grid(axis = 'both', lw = 0.3)
ax.set_ylabel(r'Pumping rate (m$^{3}$/day)')
plt.tight_layout()
plt.savefig(os.path.join('Fig9-' + nmefig + 'dvars.jpeg'), dpi = 600)


#%%--------------------- Plot constraint vs risk: Fig 10 ----------------------

cons = 'zmuni6_pct' # Choose which constraint to plot

# Files from the post calibration PESTPP-OPT run:
res_cons = res_df_cons
res_zmuni = res_df_zmuni
yerr = stdev_post_df
nme_fig = 'post_'

# Or files from theprior uncertainty PESTPP-OPT run:
#res_cons = res_df_cons_priorunc
#res_zmuni = res_df_zmuni_priorunc
#yerr = stdev_post_df_priorunc
#nme_fig = 'prior_'

# Set up figure:
    
figsze = cm2inch(19,11)
fig, ax = plt.subplots(1, 1, figsize=figsze)
fig.subplots_adjust(wspace=0.2, hspace=0.2, left=0.125, right=1, bottom=0, top=1)

colors_dict = {c_nme[i]: 
    plt.rcParams['axes.prop_cycle'].by_key()['color'][i] for i in range(len(c_nme))}


# Plot well bottom elevation
well_no = re.findall(r'\d+', cons)[0]
ax.axhline(y = muni_data.loc['muni' + well_no, 'zbotm'], 
           lw = 0.75, ls = '-', color = 'k', 
           label = 'Well bottom')

# #    ax.plot(np.NaN, np.NaN, '-', color = 'none', label = '_') # to add a blank spot in the legend
# ax.axhline(y = muni_data.loc['muni' + well_no, 'zbotm'], 
#            lw = 0.75, ls = '-', color = 'none', 
#            label = ' ')

# Plot zeta_1pct (constraint) vs risk
ax.errorbar(res_cons.index.values, res_cons.loc[:, cons], 
            yerr = yerr.loc[:, cons]*2, 
            marker = '.', lw = 1, ls = '--', c = colors_dict[cons], 
            label = r'$\zeta_{\rm 1\%}$ (95% C.I.)')

# Plotzeta_50pct vs risk
ax.plot(res_zmuni.index.values, res_zmuni.loc[:, cons.replace('_pct', '')],
        marker = '.', lw = 1, ls = '--', c = 'gray', 
        label = r'$\zeta_{\rm cell\%}$')

# Plot initial zeta_1pct elevation
ax.axhline(y = z_1pct_init['muni' + well_no], 
           lw = 1, ls = '--', c = colors_dict[cons], 
           label = r'Initial $\zeta_{\rm 1\%}$')


# Plot initial zeta_50pct elevation
ax.axhline(y = zcell_init['muni' + well_no], 
           lw = 1, ls = '--', c = 'gray', 
           label = r'Initial $\zeta_{\rm cell\%}$')

# Plot initial zeta_50pct elevation + 60%
# ax.axhline(y = zcell_init['muni' + well_no] + 0.6 * (muni_data.loc['muni' + well_no, 'zbotm'] - zcell_init['muni' + well_no]), 
#            lw = 1, ls = '--', c = 'red', 
#            label = r'Initial $\zeta_{\rm cell\%}$')

# Plot initial zeta_50pct elevation + 40%
# ax.axhline(y = zcell_init['muni' + well_no] + 0.4 * (muni_data.loc['muni' + well_no, 'zbotm'] - zcell_init['muni' + well_no]), 
#            lw = 1, ls = '--', c = 'orange', 
#            label = r'Initial $\zeta_{\rm cell\%}$')

# x-axis options

ax.set_xticks(xaxis)
ax.set_xticklabels( [str(x) for x in xaxis] )
ax.set_xlim([0, 100])
ax.set_xlabel('Reliability of the optimization (%)')
ax2 = ax.twiny()
ax2.set_xticks(xaxis)
ax2.set_xticklabels( [str( int(100-x) ) for x in xaxis] )
ax2.set_xlim([0, 100])
ax2.set_xlabel('Risk of well salinization (%)')

# Others

handles, labels = ax.get_legend_handles_labels()
list_labels = sorted(zip(labels, handles), key=lambda t: t[0])
list_labels.insert(0, list_labels.pop()) # Last element to first element
labels, handles = zip(*list_labels)
ax.legend(handles, labels, ncol = 3)
ax.grid(axis = 'both', lw = 0.3)
ax.set_ylabel('Elevation (masl)')
plt.tight_layout()
plt.savefig(os.path.join('Fig10-' + nme_fig + cons + '.jpeg'), dpi = 600)
