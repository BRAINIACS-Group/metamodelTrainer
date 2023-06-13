import os
import re
import numpy as np
import pandas as pd
from explore_param_space import *

#Import functions load the content of a single experiment
#Load functions load the content of all experiments within a folder

#TO DO : GENERALIZE PARAMETER SPREAD IN IMPORT VE

HE_path = ""
VE_path = "C:/Users/ferla/OneDrive/Documents/BRAINIACS/Simulation_Data/ADA-GEL_R_20221221_LR_B1-S1-P2a-ADA0p02-GEL0p05"

def import_HE(file):
    with open(file,'r') as f:
        f.readline()
        line2 = f.readline().strip().split(',')
        alpha = float(line2[0].split(':')[1])
        mu = float(line2[1].split(':')[1])
        df = pd.read_csv(f, skiprows=3, header=0, delimiter=',')
    n = len(df)
    strain = df.values[:,1].reshape(n,1)
    stress = df.values[:,2].reshape(n,1)
    P = Sample(np.array([[alpha,mu]]))
    X = P.spread(strain)
    Y = ExData(stress,n=1)
    X.flatten()
    Y.flatten()
    return X,Y

def load_HE(data_dir,ref_name = 'compression_tension'):
    file_list = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.startswith(ref_name):
                file_list.append(os.path.join(root, file))
    X,Y = import_HE(file_list[0])
    for i in range(1,len(file_list)):
        X_A,Y_A = import_HE(file_list[i])
        X_A = X.append(X_A)
        Y_A = Y.append(Y_A)
    X.flatten()
    Y.flatten()
    return X,Y


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

def import_VE(data_dir,verbose=0,parameters=['alpha','mu','deviatoric_20viscosity'],inputs=['time','displacement','angle'],outputs=['force','torque']): #ONLY COMP_TEN for now
    file_list = []
    mat_p = read_pfile(os.path.join(data_dir,"parameter_file.json"),names=parameters,verbose=verbose)
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
            df = pd.read_csv(os.path.join(data_dir, file),dtype=np.float32).rename(columns = {' displacement': 'displacement', ' force':'force', ' angle':'angle', ' torque':'torque'})
            df = pd.DataFrame({**default, **df},dtype=np.float32)
            dataset = pd.concat((dataset, df), ignore_index=True).sort_values(by=["time"])
    #Convert into usable Sample
    P = Sample(np.array([mat_p]))
    X = P.spread(np.array(dataset[inputs]).reshape(len(dataset),len(inputs)))
    Y = ExData(np.array(dataset[outputs]).reshape(1,len(dataset),len(outputs)),p=0)
    return X,Y

def load_VE(data_dir,verbose=1,parameters=['alpha','mu','deviatoric_20viscosity'],inputs=['time','displacement','angle'],outputs=['force','torque']):
    dir_list = [x for x in os.listdir(data_dir) if os.path.exists(os.path.join(data_dir,x,"parameter_file.json"))]
    X,Y = import_VE(os.path.join(data_dir,dir_list[0]),verbose=1,parameters=parameters,inputs=inputs,outputs=outputs)
    for i in range(1,len(dir_list)):
        X_A,Y_A = import_VE(os.path.join(data_dir,dir_list[i]),inputs=inputs,outputs=outputs)
        X = X.append(X_A)
        Y = Y.append(Y_A)
        if verbose == 1: print(f"{dir_list[i]} loaded.")
    Y = ExData(Y.reshape(Y.n*Y.t,Y.f).reshape(Y.shape),p=Y.p) #Necessary but don"t know why
    return X,Y


def get_protocol(data_dir,targets=['time','displacement','angle']):
    X,Y = import_VE(data_dir,verbose=0,outputs=targets)
    Y.flatten()
    return pd.DataFrame(Y, columns=targets)

