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

import time
#STL modules
from datetime import datetime
from pathlib import Path

#Local models

from metamodeltrainer.explore_param_space import (ParameterSpace, ExData,
    Sample, PDskSample,load_data)  #Data structure & Sampling method
from metamodeltrainer.models import (HyperParameters, RecModel, improve_random,
    load_model)           #Neural Network management
from metamodeltrainer.run_simulation import label_from_dataset

CONTINUE = True

cwd = Path(__file__).resolve().parents[1]
save_path = Path(cwd,
    f"models_precalc_force_HBE_05_16_random_20231220")#{datetime.today().strftime('%Y%m%d')}")
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
base_dir = cur_dir.parent
data_dir = base_dir / "data/FE_train_force_231220"
data_pool = (load_data(data_dir / "X.pkl"), load_data(data_dir / "Y.pkl"))
label_fn = lambda S: label_from_dataset(data_pool,S)

model_path = save_path / "model_base"

if model_path.is_dir():
    if not CONTINUE:
        raise FileExistsError(f'{model_path} already exists but CONTINUE set '
            'to false')
    model = load_model(model_path)
else:

    S = PDskSample(PSpace, k = 20) # k indicates the number of points to sample

    HP = HyperParameters(layers=[64,64],        #Architecture of the network
                        loss='mae',            #Loss to minimize
                        dropout_rate=0.5,      #Allows uncertainty
                        interpolation=1000)    #Faster computing

    X_T, Y_T = label_fn(S)
    model = RecModel(X_T,Y_T,HP)
    train_cpu_start  = time.process_time()
    train_wall_start = time.perf_counter()
    model.train(n_epochs=200,verbose=1)
    train_cpu  = time.process_time()  - train_cpu_start
    train_wall = time.perf_counter() - train_wall_start


    model.save(model_path,overwrite=True)

    with open(model_path/'times.txt','w',encoding='utf-8') as f:
        f.write(f"cpu_time_train,{train_cpu}\n"
                f"wall_time_train,{train_wall}\n"
                )

get_it_from_name = lambda d: int(d.name.split('_')[-1])
model_paths = sorted(save_path.glob("model_improved_*"),
    key=get_it_from_name)
if model_paths:
    last_model_path = model_paths[-1]
    if not CONTINUE:
        raise FileExistsError(f'{last_model_path} exists but CONTINUE set to False')
    i_start = get_it_from_name(last_model_path) + 1
    model = load_model(last_model_path)
else:
    i_start = 0

for i in range(i_start,99):
    model_path = save_path / f"model_improved_{str(i).zfill(3)}"
    improve_cpu_start = improve_wall_start = 0.
    train_cpu_start   = train_wall_start = 0.

    improve_cpu_start  = time.process_time()
    improve_wall_start = time.perf_counter()
    model_bis = improve_random(model,label_fn,PSpace,k=4,keep_weights=True)
    improve_cpu  = time.process_time() - improve_cpu_start
    improve_wall = time.perf_counter() - improve_wall_start

    train_cpu_start  = time.process_time()
    train_wall_start = time.perf_counter()
    model_bis.train(200,1)
    train_cpu  = time.process_time() - train_cpu_start
    train_wall = time.perf_counter() - train_wall_start

    model_bis.save(model_path,overwrite=True)
    with open(model_path / 'times.txt','w',encoding='utf-8') as f:
        f.write(f"cpu_time_train,{train_cpu}\n"
                f"wall_time_train,{train_wall}\n"
                f"cpu_time_improve {improve_cpu}:\n"
                f"wall_time_improve: {improve_wall}\n"
                )
    model = model_bis

'''
STEP 6 : 6. Build and train a "final" model (and save it).
'''

model_final = model.finalize(n_epochs = 1000)
model_final.save(save_path / "model_final")
