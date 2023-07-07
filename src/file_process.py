
#This file contains all the functions I used to interact with .csv files

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from explore_param_space import *
from uuid import uuid4

#Finds the value of the specified parameters in the parameter file
def read_pfile(pfile, names=['alpha','mu','deviatoric_20viscosity'],verbose=0): 
    with open(pfile,'r') as f:
        data = f.read()
    values = []
    columns = []
    for p in names:
        p_regex = f'"{p}"'+': {\n\s+"value": "(-?\d+\.\d+)",'
        p_values = re.findall(p_regex,data)
        if verbose == 1: print(f"{len(p_values)} value(s) found for {p}")
        values += p_values
        columns += [p+'_'+str(i) for i in range(len(p_values))]
    if verbose == 1: print(f"{len(values)} material parameters found in total.")
    return [float(x) for x in values],columns


#Reads and formats the data from a single simulation, given a path to its directory
#Separates into inputs (material parameters and given input columns) and outputs
def import_csv(data_dir,verbose=0,parameters=['alpha','mu','deviatoric_20viscosity'],inputs=['time','displacement','angle'],outputs=['force','torque']): #ONLY COMP_TEN for now
    file_list = []
    mat_p, names = read_pfile(Path(data_dir,"parameter_file.json"),names=parameters,verbose=verbose)
    default = {'time': 0, 'displacement': 0, 'force': 0, 'angle': 0, 'torque': 0}
    for i,x in enumerate(names):
        default[x] = mat_p[i]
    columns = names + ['time', 'displacement', 'force', 'angle', 'torque']
    dataset = pd.DataFrame(columns=columns)
    #Get data
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):# and file[1] == '_' and int(file[0]) < 6:
            df = pd.read_csv(Path(data_dir, file),dtype=np.float32).rename(columns = {' displacement': 'displacement', ' force':'force', ' angle':'angle', ' torque':'torque'})
            df = pd.DataFrame({**default, **df},dtype=np.float32)
            dataset = pd.concat((dataset, df), ignore_index=True).sort_values(by=["time"])
    #Convert into usable Sample
    P = Sample(np.array([mat_p]),columns=names)
    X = P.spread(np.array(dataset[inputs]).reshape(len(dataset),len(inputs)),input_columns=inputs)
    Y = ExData(np.array(dataset[outputs]).reshape(1,len(dataset),len(outputs)),p=0,columns=outputs)
    return X,Y

#Loads up all simulation results from a given folder
def load_FE(data_dir,verbose=1,parameters=['alpha','mu','deviatoric_20viscosity'],inputs=['time','displacement','angle'],outputs=['force','torque']):
    dir_list = [x for x in os.listdir(data_dir) if os.path.exists(Path(data_dir,x,"parameter_file.json"))]
    X,Y = import_csv(Path(data_dir,dir_list[0]),verbose=1,parameters=parameters,inputs=inputs,outputs=outputs)
    if verbose == 1: print(f"{dir_list[0]} loaded.")
    for i in range(1,len(dir_list)):
        X_A,Y_A = import_csv(Path(data_dir,dir_list[i]),inputs=inputs,outputs=outputs)
        try: 
            X = X.append(X_A)
            Y = Y.append(Y_A)
            if verbose == 1: print(f"{dir_list[i]} loaded.")
        except:
            print(dir_list[i] + " not loaded : mismatched shape.")
    X = ExData(X.reshape(X.n*X.t,X.f).reshape(X.shape),p=X.p,columns=X.columns)
    Y = ExData(Y.reshape(Y.n*Y.t,Y.f).reshape(Y.shape),p=Y.p,columns=Y.columns) #Necessary but don't know why
    return X,Y

#Ask Jan : ParameterHandler might be very useful
#Goal : Fully replace call to simulation ? Give a parameter file, read the useful parameters, testing devices, input files
#Write output files

def res_to_file(X,Y,input_dir = Path(Path(__file__).resolve().parents[1],'FE','data','input','10.01.2022ALG_5_GEL_5_P2'),output_dir = str(uuid4())[:8]): #Takes the predicted results from the model and writes them to a folder following input structure
    
    if not os.path.exists(output_dir): os.mkdir(output_dir)

    if X.n == 1:

        X.flatten()
        Y.flatten()
        res_df = pd.DataFrame(np.hstack((X[:,X.p:],Y)),columns = X.columns[X.p:]+Y.columns)
        ldir = sorted([x for x in os.listdir(input_dir) if x.endswith('.csv')], key=lambda x: int(x.split('_')[0]))
        i = 0
        for file in ldir:
            df = pd.read_csv(Path(input_dir, file),dtype=np.float32)
            n = len(df)
            res_df[i:i+n].to_csv(Path(output_dir,file),index=False)
            i += n
    else:
        for i in range(X.n):
            res_to_file(X[i],Y[i],input_dir,Path(output_dir,str(uuid4())[:8]))

        
