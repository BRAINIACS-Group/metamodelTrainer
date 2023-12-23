
from pathlib import Path
import sys
parent_dir = Path(__file__).resolve().parents[2]
pyLabPath = parent_dir / "efiPyVlab/src"
sys.path += [str(pyLabPath),
        str(Path(__file__).resolve().parents[1])]

import pyVlab
#STL modules
from datetime import datetime
from pathlib import Path

#Local models

from metamodeltrainer.explore_param_space import ParameterSpace, ExData, Sample, PDskSample  #Data structure & Sampling method
from metamodeltrainer.models import HyperParameters, RecModel, improve, load_model           #Neural Network management
from metamodeltrainer.run_simulation import label



cwd = Path(__file__).resolve().parents[1]
save_path = Path(cwd,f"models_HBE_05_16_red_active_{datetime.today().strftime('%Y%m%d')}")
if not save_path.is_dir():
    save_path.mkdir()

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

S = PDskSample(PSpace, k = 25) # k indicates the number of points to sample
X_T, Y_T = label_fn(S)

HP = HyperParameters(layers=[64,64],        #Architecture of the network
                     loss='mae',            #Loss to minimize
                     dropout_rate=0.5,      #Allows uncertainty
                     interpolation=1000)    #Faster computing

model = RecModel(X_T,Y_T,HP)
model.train(n_epochs=100,verbose=1)
model.save(save_path / "model_base",overwrite=True)

for i in range(95):
    model_bis = improve(model,label_fn,PSpace,k=5)
    model_bis.train(100,1)
    model_bis.save(save_path / f"model_improved_{str(i).zfill(3)}",overwrite=True)
    model = model_bis

model_final = model.finalize(n_epochs = 1000)
model_final.save(save_path / "model_final")
