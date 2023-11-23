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

#from efiPostProc import SimRes

def get_execPath():
    if not 'VLAB_EXEC' in os.environ.keys():
        return None
    return os.environ['VLAB_EXEC']

def label(X,prm_file = Path('../FE/data/prm/reference.prm')):
    path = run_sim(X,prm_file)
    #path = "/home/jan/Projects/PythonCodeBase/metamodelTrainer/FE/out/d5d93e59"
    X_res, Y_res = load_FE(path)
    P,S = X_res.separate()
    inputs = X_res.columns[X_res.p:]
    comp = np.sum(P - X[0],axis = 1)
    k = 0
    for j in range(len(comp)):
        if abs(comp[j]) < abs(comp[k]):
            k = j
    X_cor = Sample([P[k]],columns=X.columns).spread(S,input_columns=inputs)
    Y_cor = Y_res[k]

    for i in range(1,len(X)):
        inputs = X_res.columns[X_res.p:]
        comp = np.sum(P - X[i],axis = 1)
        k = 0
        for j in range(len(comp)):
            if abs(comp[j]) < abs(comp[k]):
                k = j
        X_cor = X_cor.append(Sample([P[k]],columns=X.columns).spread(S,input_columns=inputs))
        Y_cor = Y_cor.append(Y_res[k])
    X_cor.reform()
    Y_cor.reform()
    return X_cor, Y_cor


def run_sim(X,prm_file:Path,
    out_dir:Path=Path(__file__).resolve().parents[1] / Path('FE/out') ):
    
    prm = ParameterHandler.from_file(prm_file)
    
    variables = X  
    dir_name = str(uuid4())[:8]
    
    wd = out_dir / dir_name
    wd = wd.resolve()
    if not wd.is_dir():
        wd.mkdir()

    sim = FESimulation(prm,execPath=get_execPath(),
        wd=wd,out_dir=wd,nworker=8,nthreads_per_proc=2)

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
    #simRes = SimRes(prm,output_dir=wd)
    
    return wd
   
