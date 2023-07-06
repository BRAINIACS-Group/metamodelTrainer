from explore_param_space import *
from models import *
from file_process import *

import numpy as np
import matplotlib.pyplot as mpl
import os
from sklearn.metrics import r2_score

#Path definition
cwd = Path(__file__).resolve().parents[1]
data_path = Path(cwd,"data","all_data")
save_path = Path(cwd,"models")
res_path = Path(cwd,"results")

X,Y = load_FE(data_path)
X.save(Path(cwd,'data','all_data_X'))
Y.save(Path(cwd,'data','all_data_Y'))

def get_r2(name):
    if os.path.exists(Path(res_path,name+'.npy')):
        return np.load(Path(res_path,name+'.npy'))
    print("Computing "+name)
    model = load_model(Path(save_path,name))
    Y_P = model.predict(X)
    r2 = [r2_score(Y_P[i],Y[i]) for i in range(len(X))]
    np.save(Path(res_path,name),np.array(r2))

AL = get_r2("MegaModel_AL_Final")
RL = get_r2("MegaModel_RL_Final")

mpl.subplot(1,2,1)
mpl.scatter([i for i in range(len(AL))], AL)
mpl.subplot(1,2,2)
mpl.scatter([i for i in range(len(AL))], AL)
mpl.show()