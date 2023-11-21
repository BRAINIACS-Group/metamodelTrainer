
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
label_fn = lambda S: label(S,prm_file = cur_dir / Path('../FE/data/prm/HBE_05_16.prm'))

data_dir = cur_dir / Path('../data/PDsk_64_test')
if not data_dir.is_dir():
    data_dir.mkdir()

S = PDskSample(PSpace, k = 2) # k indicates the number of points to sample
X_T, Y_T = label_fn(S)
X_T.save(data_dir / 'X_T')
Y_T.save(data_dir / 'Y_T')