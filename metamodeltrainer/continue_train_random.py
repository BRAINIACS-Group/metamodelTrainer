'''

This script is an example pipeline with step-by-step instructions for building a neural network from scratch, 
using active learning, to fit a given constitutive visco-elastic model, following a certain protocol.

Overall, the structure of the process is :
1. Define a parameter space to explore
2. Generate an initial sampling of the parameter space
3. Build and train a model equipped with an uncertainty indicator on this sampling.
4. Use active learning to find new points to add to the training sample and train a new model on it.
5. Repeat step 4 until satisfied.
6. Build and train a "final" model (and save it).

The following example is based on the ___ model (don't know the name - one term-ogden ...?).

'''
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

cur_dir = Path(__file__).resolve().parent
label_fn = lambda S: label(S,prm_file = cur_dir / Path('../FE/data/prm/HBE_05_16.prm'))

cwd = Path(__file__).resolve().parents[1]
save_path = Path(cwd,"models_HBE_05_16_random20231116")

HP = HyperParameters(layers=[64,64],        #Architecture of the network
                     loss='mae',            #Loss to minimize
                     dropout_rate=0.5,      #Allows uncertainty
                     interpolation=1000)    #Faster computing

PSpace = ParameterSpace(
    alpha_inf = (-20,20),
    mu_inf    = (100,2000),
    alpha_1   = (-20,20),
    mu_1      = (100,2000),
    eta_1     = (0,10000)
)

model = load_model(save_path / "model_improved_047")

for i in range(48,96+1):
    model_bis = improve(model,label_fn,PSpace,k=2)
    model_bis.train(100,1)
    model_bis.save(save_path / f"model_improved_{str(i).zfill(3)}",overwrite=True)
    model = model_bis

model_final = model.finalize(n_epochs = 1000)
model_final.save(save_path / f"model_{i}_final")
