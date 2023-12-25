from pathlib import Path
import sys
parent_dir = Path(__file__).resolve().parents[2]
pyLabPath = parent_dir / "efiPyVlab/src"
sys.path += [str(pyLabPath),
        str(Path(__file__).resolve().parents[1])]

#STL modules
from pathlib  import Path
from datetime import datetime

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
label_fn = lambda S: label(S,
    prm_file = cur_dir / Path('../FE/data/prm/HBE_05_16_red.prm'),stress=False)

data_dir = cur_dir / Path(
    f"../data/PDsk_100_red_{datetime.today().strftime('%Y%m%d')}")
if not data_dir.is_dir():
    data_dir.mkdir()

S = PDskSample(PSpace, k = 100) # k indicates the number of points to sample
X_T, Y_T = label_fn(S)
X_T.save(data_dir / 'X_T')
Y_T.save(data_dir / 'Y_T')