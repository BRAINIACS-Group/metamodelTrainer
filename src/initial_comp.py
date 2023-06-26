from explore_param_space import *
from models import *
from file_process import *
from run_simulation import *
import matplotlib.pyplot as mpl

#Define important paths
cwd = Path(__file__).resolve().parents[1]
data_path = Path(cwd,"data","LHCU_20")
save_path = Path(cwd,"models")

#Define parameter space boundaries
#Same order as OptVars in run_simulation.py
R = [(-20,20),(100,2000),(-20,20),(100,2000),(0,10000)]
#Define model HyperParameters
HP = HyperParameters(layers=[64,64],
                     loss='mae',
                     dropout_rate=0,
                     interpolation=1000)

#PDisk method
'''
X_T,Y_T = load_FE(label(PDskSample(R,64)))
model = RecModel(X_T,Y_T,HP)
model.train(1000,1)
model.save(Path(save_path,"model_poisson64"))

#Random method
X_T,Y_T = load_FE(label(RandSample(R,64)))
model = RecModel(X_T,Y_T,HP)
model.train(1000,1)
model.save(Path(save_path,"model_random64"))

#Latin Hypercube
X_T,Y_T = load_FE(label(LHCuSample(R,64)))
model = RecModel(X_T,Y_T,HP)
model.train(1000,1)
model.save(Path(save_path,"model_lhcu64"))
'''
#Grid method
P = GridSample(R,2).append(GridSample(R,2))
print(len(P))
X_T,Y_T = load_FE(label(P))
model = RecModel(X_T,Y_T,HP)
model.train(1000,1)
model.save(Path(save_path,"model_grid64"))
