#These functions are built to read and format FE data to be used by the models

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from explore_param_space import *

#Finds the value of the specified parameters in the parameter file
def read_pfile(pfile, names=['alpha','mu','deviatoric_20viscosity'],verbose=0): 
    with open(pfile,'r') as f:
        data = f.read()
    values = []
    for p in names:
        p_regex = f'"{p}"'+': {\n\s+"value": "(-?\d+\.\d+)",'
        p_values = re.findall(p_regex,data)
        if verbose == 1: print(f"{len(p_values)} value(s) found for {p}")
        values += p_values
    if verbose == 1: print(f"{len(values)} material parameters found in total.")
    return [float(x) for x in values]


#Reads and formats the data from a single simulation, given a path to its directory
#Separates into inputs (material parameters and given input columns) and outputs
def import_csv(data_dir,verbose=0,parameters=['alpha','mu','deviatoric_20viscosity'],inputs=['time','displacement','angle'],outputs=['force','torque']): #ONLY COMP_TEN for now
    file_list = []
    mat_p = read_pfile(Path(data_dir,"parameter_file.json"),names=parameters,verbose=verbose)
    default = {'time': 0, 'displacement': 0, 'force': 0, 'angle': 0, 'torque': 0}
    names = []
    for i in range(len(mat_p)):
        name = f'mat_p_{str(i)}'
        names.append(name)
        default[name] = mat_p[i]
    columns = names + ['time', 'displacement', 'force', 'angle', 'torque']
    dataset = pd.DataFrame(columns=columns)
    #Get data
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):# and file[1] == '_' and int(file[0]) < 6:
            df = pd.read_csv(Path(data_dir, file),dtype=np.float32).rename(columns = {' displacement': 'displacement', ' force':'force', ' angle':'angle', ' torque':'torque'})
            df = pd.DataFrame({**default, **df},dtype=np.float32)
            dataset = pd.concat((dataset, df), ignore_index=True).sort_values(by=["time"])
    #Convert into usable Sample
    P = Sample(np.array([mat_p]))
    X = P.spread(np.array(dataset[inputs]).reshape(len(dataset),len(inputs)))
    Y = ExData(np.array(dataset[outputs]).reshape(1,len(dataset),len(outputs)),p=0)
    return X,Y

#Loads up all simulation results from a given folder
def load_FE(data_dir,verbose=1,parameters=['alpha','mu','deviatoric_20viscosity'],inputs=['time','displacement','angle'],outputs=['force','torque']):
    dir_list = [x for x in os.listdir(data_dir) if os.path.exists(Path(data_dir,x,"parameter_file.json"))]
    X,Y = import_csv(Path(data_dir,dir_list[0]),verbose=1,parameters=parameters,inputs=inputs,outputs=outputs)
    for i in range(1,len(dir_list)):
        X_A,Y_A = import_csv(Path(data_dir,dir_list[i]),inputs=inputs,outputs=outputs)
        X = X.append(X_A)
        Y = Y.append(Y_A)
        if verbose == 1: print(f"{dir_list[i]} loaded.")
    Y = ExData(Y.reshape(Y.n*Y.t,Y.f).reshape(Y.shape),p=Y.p) #Necessary but don't know why
    return X,Y

