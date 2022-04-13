# Author: Cecile Coulon
#
# ---------------------------------- Readme -----------------------------------
# No modifications needed to the script below
#
#----------------------------------- Script -----------------------------------

# Load python libraries
import itertools
import pandas as pd


''' Float formatter function to apply when exporting data '''
FFMT = lambda x: "{0:<20.10E} ".format(float(x)) # <: align left, 20: length of output, 10: # of charact after decimal point, E: sci notation

def SFMT(item):
    
    ''' String formatter function to apply when exporting data '''
    
    try:
        s = "{0:<20s} ".format(item.decode())
    except:
        s = "{0:<20s} ".format(str(item))
    return s

# Define which functions to apply to various exported columns
DIC_FMT = {'name': SFMT, 'value': FFMT, 'ins': SFMT, 'tpl': SFMT}

# Width of values (number of characters)
VAL_START = 12
VAL_CHAR_LEN = 30


def get_row_col_id(dict_row_col):
    
    ''' Create unsorted dataframe with index=name and 
    columns=row, col, for model cells specified in dict_row_col'''
    
    rows=[]
    cols=[]
    names=[]
    for elem in dict_row_col.values():
        row, col, corr = itertools.chain(elem[0]) # Unpack tuple
        rows.append(row) # Create list for rows
        cols.append(col) # Create list for columns
    for name in dict_row_col.keys():
        names.append(name) # Create list for names
    df_row_col = pd.DataFrame(data={'row': rows, 'col': cols, 'name': names}) # Create df
    return df_row_col


def sim_to_df(dict_row_col, model_output, tstp):
    
    ''' Create sorted df containing the data simulated at specific model cells
    Inputs:
        model_output = h_swiP or zetaP '''
    
    df_sim = get_row_col_id(dict_row_col) # Create df with index=name, columns= row, col
    df_sim['value'] = model_output[tstp, 0, df_sim['row'], df_sim['col']] # Add column with model_output for selected timestep
    df_sim = df_sim.sort_values(by=['name'], ascending=True) # Sort data with names in ascending order
    df_sim = df_sim.reset_index(drop=True) # Reset index from 0 after sorting, and do not turn the old index into a column
    return(df_sim)


def write_df_to_dat(file_path, file_name, df_subset):
    
    ''' # Export dataframe to .dat file:
    Inputs:
        file_name = string,
        columns=list of strings '''
    
    dest_file = open(file_path + file_name + '.dat','w')
    dest_file.write(df_subset.to_string(col_space=0, formatters=DIC_FMT, justify="left", 
                                 header=False, index=False))
    dest_file.close()


def write_df_to_tpl(file_path, file_name, df_subset):
    
    ''' # Write parameters to PEST template (.tpl) file '''
    
    f_tpl = open(file_path + file_name + '.tpl','w')
    f_tpl.write("ptf ~\n")
    f_tpl.write(df_subset.to_string(col_space=0, formatters=DIC_FMT, justify='left', 
                             header=False, index=False))


def write_df_to_ins(file_path, file_name, df_subset):
    
    ''' Write dataframe to PEST instruction (.ins) file '''
    
    f_ins = open(file_path + file_name + '.ins','w')
    f_ins.write("pif #\n")
    f_ins.write(df_subset.to_string(col_space=0, formatters=DIC_FMT, justify='left', 
                             header=False, index=False))
