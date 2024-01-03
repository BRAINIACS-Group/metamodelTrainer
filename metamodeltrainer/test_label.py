from pathlib import Path
import sys
parent_dir = Path(__file__).resolve().parents[2]
pyLabPath = parent_dir / "efiPyVlab/src"
sys.path += [str(pyLabPath),
        str(Path(__file__).resolve().parents[1])]

#STL modules
from pathlib import Path

#3rd party modules

from metamodeltrainer.explore_param_space import ParameterSpace, ExData, Sample, PDskSample  #Data structure & Sampling method
from metamodeltrainer.models import HyperParameters, RecModel, improve, load_model           #Neural Network management
from metamodeltrainer.run_simulation import label


PSpace = ParameterSpace(
    alpha_inf = (-20,20),
    mu_inf    = (100,2000),
    alpha_1   = (-20,20),
    mu_1      = (100,2000),
    eta_1     = (0,10000)
)

cur_dir = Path(__file__).resolve().parent
label_fn = lambda S: label(S,prm_file = cur_dir / Path('../FE/data/prm/HBE_05_16_red.prm'))

data_dir = cur_dir / Path('../data/PDsk_64_231121')
if not data_dir.is_dir():
    data_dir.mkdir()

#S = PDskSample(PSpace, k = 1) # k indicates the number of points to sample
S = Sample([[-10.0,400.0,-5.0,300.0,5000.0]],columns = ['alpha_inf','mu_inf','alpha_1','mu_1','eta_1'])
print(S.columns)
print(S)

X,Y = label_fn(S)

HP = HyperParameters(layers=[64,64],        #Architecture of the network
                        loss='mae',            #Loss to minimize
                        dropout_rate=0.5,      #Allows uncertainty
                        interpolation=1000)    #Faster computing

model = RecModel(X,Y,HP)
model.train(n_epochs=100,verbose=1)

