#Find repositories
from pathlib import Path
import sys
print(Path(__file__).resolve())
parent_dir = Path(__file__).resolve().parents[2]
pyLabPath = parent_dir / "efiPyVlab/src"
pyPostPath = parent_dir / "efiPostProc/src"
pyOptPath = parent_dir / "efiOpt/src"
sys.path += [str(pyLabPath),
str(pyPostPath),str(pyOptPath)]

#STL imports
from pathlib import Path
import math
import os
from uuid import uuid4

#3rd party imports
import matplotlib.pyplot as plt
import numpy as np

#Carl imports
from file_process import load_FE
from explore_param_space import Sample, ExData

#EFI imports
from efiopt.simulation import FESimulation
from pyVlab.parameters import ParameterHandler
from pyVlab.optvars    import OptVars,OptVarData
from pyVlab.testing_device import TestingDevice
from pyVlab.geometry   import Geometry

from efiPostProc import SimRes

def get_execPath():
    if not 'VLAB_EXEC' in os.environ.keys():
        return None
    return os.environ['VLAB_EXEC']

def label(X): #A really inelegant way to get the material parameters in the correct order for the model
    path = run_sim(X)
    X_res, Y_res = load_FE(path)
    P,S = X_res[0].separate()
    print(P.shape)
    inputs = X_res.columns[X_res.p:]
    comp = np.sum(X - P,axis = 1)
    k = 0
    for j in range(len(comp)):
        if abs(comp[j]) < abs(comp[k]):
            k = j
    X_cor = P[k].spread(S,input_columns=inputs)
    Y_cor = Y_res[k]

    for i in range(1,len(X_res)):
        P,S = X_res[i].separate()
        inputs = X_res.columns[X_res.p:]
        comp = np.sum(X - P,axis = 1)
        k = 0
        for j in range(len(comp)):
            if abs(comp[j]) < abs(comp[k]):
                k = j
        X_cor = X_cor.append(X[k].spread(S,input_columns=inputs))
        Y_cor = Y_cor.append(Y_res[k])

    return X_cor, Y_cor


def run_sim(X):
    prm_file = Path('../FE/data/prm/reference.prm')
    prm = ParameterHandler.from_file(prm_file)
    
    variables = X  
    dir_name = str(uuid4())[:8]
    
    wd = Path(f'../FE/out/{dir_name}').resolve()
    if not wd.is_dir():
        wd.mkdir()

    sim = FESimulation(prm,execPath=get_execPath(),
        wd=wd,out_dir=wd,nworker=4,nthreads_per_proc=4)

    optVars = [OptVars(
        alpha_inf = OptVarData(variables[i][variables.columns.index('alpha_inf')],  bounds=(-100,100)),
        mu_inf    = OptVarData(variables[i][variables.columns.index('mu_inf')],bounds=(0,math.inf)),
        alpha_1   = OptVarData(variables[i][variables.columns.index('alpha_1')],  bounds=(-100,100)),
        mu_1      = OptVarData(variables[i][variables.columns.index('mu_1')],    bounds=(0,math.inf)),
        eta_1     = OptVarData(variables[i][variables.columns.index('eta_1')],   bounds=(0,math.inf))
    ) for i in range(len(variables))]
    
    #create job (should name that better)
    job = sim.start_job(optVars)
    #call to start_job is non blocking
    job.start()
    print('normal program flow continues and jobs run in separate processes')
    #call to eval is blocking 
    res = job.eval()

    #these are just exemplary calls to create Geometry and testing_devices objects
    geom = Geometry.from_prm(prm,path='simulation/experiment/sample/geometry')
    testing_devices = TestingDevice.from_prm(prm,path='simulation')

    #The SimRes objects uses the TestingDevice and Geometry object under the hood 
    #to calculate stresses and so on to plot as well as write out results
    simRes = SimRes(prm,output_dir=wd)
    
    return Path(f'../FE/out/{dir_name}').resolve()
   
    
  
