#Carl imports
from explore_param_space import *
from file_process import *
from models import *
from run_simulation import label

#Third-party imports
import numpy as np
import matplotlib.pyplot as mpl
from pathlib import Path
from uuid import uuid4

#Define important paths
cwd = Path(__file__).resolve().parents[1]
data_path = Path(cwd,"data","LHCU_20")
save_path = Path(cwd,"models")

#Define parameter space boundaries
#Same order as OptVars in run_simulation.py

PSpace = ParameterSpace(
    alpha_inf = (-20,20),
    mu_inf    = (100,2000),
    alpha_1   = (-20,20),
    mu_1      = (100,2000),
    eta_1     = (0,10000)
)
print(PSpace.keys(),PSpace.values())
#Create an initial sampling of the parameter space (k points, Latin Hypercube method)
k = 4
S = PDskSample(PSpace,k)

#Label them by running simulations, and keep in a variable the folder in which the results are saved
#res_path = label(S)
#res_path = Path(cwd,"FE","out","6bda37d2")
#Load the results and format them for model training
#X_T,Y_T = load_FE(res_path)

X_T, Y_T = label(S)

#Define model HyperParameters
HP = HyperParameters(layers=[64,64],
                     loss='mae',
                     dropout_rate=0.5,
                     interpolation=1000)

#Build model
model = RecModel(X_T,Y_T,HP)
#model = MegaModel(X_T,Y_T,10,'RNN',HP)

#train model
model.train(n_epochs=10, verbose=1)

model.save(Path(save_path,"model_test_base"))

#label_fn = lambda X: load_FE(label(X))

for i in range(4):
    model_bis = improve(model,label,PSpace,k=4)
    model_bis.save(Path(save_path,f"model_test_improved_{str(i).zfill(3)}"))
    model = model_bis

X_T,Y_T = model_bis.X_T, model_bis.Y_T

HPf = HyperParameters(layers=[64,64],
                     loss='mae',
                     dropout_rate=0,
                     interpolation=1000)

final = RecModel(X_T,Y_T,HPf)
final.train(100,1)
final.save(Path(save_path,"model_test_final"))