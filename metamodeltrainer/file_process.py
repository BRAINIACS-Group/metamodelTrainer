
#This file contains all the functions I used to interact with .csv files

#STL imports
import os
import shutil
import re
from pathlib import Path
from uuid import uuid4
from itertools import chain
from typing import List

#3rd party imports
import numpy as np
import pandas as pd

#pyVlab imports
from pyVlab import ParameterHandler
from pyVlab.geometry import Geometry,Cylinder

#local imports
from metamodeltrainer.explore_param_space import *
from metamodeltrainer.utility import convert_to_force_disp,convert_to_stress

def read_paramfile(pfile, parameters=['alpha','mu','deviatoric_20viscosity'],verbose=0): 
    
    # Uses regular expressions to find the specified parameters, and return them (with semi-appropriate names)
    # Jan might have a better function...? 

    with open(pfile,'r') as f:
        data = f.read()
    values = []
    columns = []
    for p in parameters:
        p_regex = f'"{p}"'+': {\n\s+"value": "(-?\d+\.\d+)",'
        p_values = re.findall(p_regex,data)
        if verbose == 1: print(f"{len(p_values)} value(s) found for {p}")
        values += p_values
        columns += [p+'_'+str(i) for i in range(len(p_values))]
    if verbose == 1: print(f"{len(values)} material parameters found in total.")
    return [float(x) for x in values],columns

def get_material_parameters_fix(prm:ParameterHandler):
    '''quick fix to reliably get parameters for maxwell model with ogden 
    springs'''
    
    prm_dict = {}
    sec_path = 'experiment/sample/'\
        'constitutive@[type=maxwell_wiechert,instance=1]'

    ogden_inf_path = sec_path +\
        '/constitutive@[type=modified_one_term_ogden,instance=1]'
    prm_dict['alpha_inf'] = prm.get_float(ogden_inf_path, 'alpha')
    prm_dict['mu_inf']    = prm.get_float(ogden_inf_path, 'mu')

    maxwell_element_path = sec_path +\
        '/constitutive@[type=maxwell_element,instance=1]'
    prm_dict['eta_1']     = prm.get_float(maxwell_element_path,
        'deviatoric viscosity')

    maxwell_spring_path = maxwell_element_path+\
        '/constitutive@[type=modified_one_term_ogden,instance=1]'
    prm_dict['alpha_1'] = prm.get_float(maxwell_spring_path, 'alpha')
    prm_dict['mu_1']    = prm.get_float(maxwell_spring_path, 'mu')

    return prm_dict

def import_csv(data_dir,parameters:List[str],verbose:int=0,stress:bool=True):
    #parameters=['alpha','mu','deviatoric_20viscosity'],
    #inputs=['time','displacement','angle'],outputs=['force','torque'],
    
    #ONLY COMP_TEN for now
    #Reads and formats the data from a single simulation, given a path to its directory
    #Separates into inputs (material parameters and given input columns) and outputs

    #mat_p, names = read_paramfile(Path(data_dir,"parameter_file.json"),parameters=parameters,verbose=verbose)
    prm_file = data_dir / "parameter_file.prm"
    parameter_handler = ParameterHandler.from_file(prm_file)
    mat_p_dict = get_material_parameters_fix(parameter_handler)
    geom = Geometry.from_prm(parameter_handler,'experiment/sample/geometry')

    if set(parameters) != set(mat_p_dict.keys()):
        raise ValueError('different parameters found')
    mat_p_dict = {k:mat_p_dict[k] for k in parameters}

    inputs = ['time']
    if stress:
        inputs  += ['stretch','shear']
        outputs = ['normal_stress','shear_stress']
    else:
        inputs  += ['displacement','angle']
        outputs = ['force','torque']
        
    default = {k:0. for k in chain(inputs,outputs)}
    #default = {**mat_p_dict, **default}

    #dataset = pd.DataFrame(columns=columns)
    #Get data
    df_list = []
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(Path(data_dir, file),dtype=np.float32)\
                .rename(columns = str.strip)
            if stress:
                df = convert_to_stress(df,geom)
            df_list.append(pd.DataFrame({**default, **df},dtype=np.float32))
            
    dataset = pd.concat(df_list, ignore_index=True)\
        .sort_values(by=["time"])\
        .assign(**mat_p_dict)

    #Convert into usable ExData
    P = Sample(np.array([list(mat_p_dict.values())]),
        columns=list(mat_p_dict.keys()))
    X = P.spread(np.array(dataset[inputs]).reshape(len(dataset),len(inputs)),
        input_columns=inputs)
    Y = ExData(np.array(dataset[outputs]).reshape(1,len(dataset),len(outputs)),
        p=0,columns=outputs)
    
    return X,Y

def load_FE(data_dir,parameters:List[str],verbose=1,stress:bool=True):
    
    #Loads up all simulation results from a given folder (applies import_csv to all subdirectories)
    #Returns two ExData objects with all the results (input & output)
    
    dir_list = [x for x in os.listdir(data_dir) 
        if os.path.exists(Path(data_dir,x,"parameter_file.json"))]
    
    X,Y = import_csv(Path(data_dir,dir_list[0]),parameters,verbose=1,stress=stress)
    if verbose == 1: print(f"{dir_list[0]} loaded.")
    
    for i in range(1,len(dir_list)):
        X_A,Y_A = import_csv(Path(data_dir,dir_list[i]),parameters,verbose=1,
            stress=stress)
        try:
            X = X.append(X_A)
            Y = Y.append(Y_A)
            if verbose == 1: print(f"{dir_list[i]} loaded.")
        except:
            print(dir_list[i] + " not loaded : mismatched shape.")
    X = ExData(X.reshape(X.n*X.t,X.f).reshape(X.shape),p=X.p,columns=X.columns)
    Y = ExData(Y.reshape(Y.n*Y.t,Y.f).reshape(Y.shape),p=Y.p,columns=Y.columns) #Necessary but don't know why
    return X,Y



def res_to_file(X,Y,input_dir:Path = Path(Path(__file__).resolve().parents[1],'FE','data','input','10.01.2022ALG_5_GEL_5_P2'),
                output_dir = Path(Path(__file__).resolve().parents[1],'out',str(uuid4())[:8]),
                parameter_file = Path(Path(__file__).resolve().parents[1],'FE','data','prm','reference.prm')
                ): #Takes the predicted results from the model and writes them to a folder following input structure

    '''
    Writes two ExData objects to .csv files following the structure of files from input_dir
    The function sorts all files from input_dir using as key the integer before _ in the file name
    (should sort them according to time) 
    Then creates in the output_dir new files with the same names, and the corresponding inputs/outputs from X and Y
    
    Note : Correspondance is decided by number of lines in the input files,
    So make sure that the input_dir corresponds to what generate X in the first place !

    MISSING : A way to write the corresponding parameter file...
    '''

    if not os.path.exists(output_dir): os.mkdir(output_dir)

    if X.n == 1:

        X.flatten()
        Y.flatten()
        res_df = pd.DataFrame(np.hstack((X[:,X.p:],Y)),columns = X.columns[X.p:]+Y.columns)
        
        prm = ParameterHandler.from_file(parameter_file)
        geom_prm = Geometry.from_prm(prm,
            'simulation/experiment/sample/geometry')
        if 'displacement' in X.columns:
            res_df = convert_to_stress(res_df,
                geom=Cylinder(radius=4e-3,height=0.00369233203125))
        res_df = convert_to_force_disp(res_df,
        geom=geom_prm)

        ldir = sorted([x for x in os.listdir(input_dir) if x.endswith('.csv')], key=lambda x: int(x.split('_')[0]))
        i = 0
        for file in ldir:
            df = pd.read_csv(Path(input_dir, file),dtype=np.float32)
            n = len(df)
            res_df[i:i+n].to_csv(Path(output_dir,file),index=False)
            i += n
        with open(parameter_file,'r') as f:
            pref = f.read()
        
        variables = {}
        for i in range(len(X.columns)):
            variables[X.columns[i]] = X[0,i]
        variables['simulation_output_directory'] = output_dir

        def evaluate_expression(match):
            expression = match.group(1)
            return str(eval(expression, variables))

        pstring = re.sub(r'{([^{}]+)}', evaluate_expression, pref)

        with open(Path(output_dir,"parameter_file.prm"),'w') as f:
            f.write(pstring)

    else:
        for i in range(X.n):
            res_to_file(X[i],Y[i],input_dir,Path(output_dir,str(uuid4())[:8]))
