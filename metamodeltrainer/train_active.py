
from pathlib import Path
import sys
parent_dir = Path(__file__).resolve().parents[2]
pyLabPath = parent_dir / "efiPyVlab/src"
sys.path += [str(pyLabPath),
        str(Path(__file__).resolve().parents[1])]

import pyVlab
#STL modules
import time
from datetime import datetime
from pathlib import Path

#Local models

from metamodeltrainer.explore_param_space import (ParameterSpace, ExData, Sample,
    PDskSample,LHCuSample,load_data)  #Data structure & Sampling method
from metamodeltrainer.models import (HyperParameters, RecModel, improve,
    load_model)           #Neural Network management
from metamodeltrainer.run_simulation import label,label_from_dataset

CONTINUE = True

cwd = Path(__file__).resolve().parents[1]
save_path = Path(cwd,f"models_HBE_05_16_red_stress_active_{datetime.today().strftime('%Y%m%d')}")
if not save_path.is_dir():
    save_path.mkdir()

PSpace = ParameterSpace(
    alpha_inf = (-20,20),
    mu_inf    = (50,2000),
    alpha_1   = (-20,20),
    mu_1      = (50,2000),
    eta_1     = (0,10000)
)

cur_dir = Path(__file__).resolve().parent
label_fn = lambda S: label(S,
    prm_file = cur_dir / Path('../FE/data/prm/HBE_05_16_red.prm'),stress=True)


model_path = save_path / "model_base"

if model_path.is_dir():
    if not CONTINUE:
        raise ValueError(f'found model at {model_path} but CONTINUE set to false')
    model = load_model(model_path)
else:
    S = LHCuSample(PSpace, k = 25) # k indicates the number of points to sample

    label_cpu_start  = time.process_time()
    label_wall_start = time.perf_counter()
    X_T, Y_T = label_fn(S)
    label_cpu  = time.process_time() - label_cpu_start
    label_wall = time.perf_counter() - label_wall_start

    HP = HyperParameters(layers=[64,64],        #Architecture of the network
                        loss='mae',            #Loss to minimize
                        dropout_rate=0.5,      #Allows uncertainty
                        interpolation=1000)    #Faster computing

    model = RecModel(X_T,Y_T,HP)
    #model = load_model("/home/jan/Projects/Active_Learning/plots/data/models/"
    #    "models_red_force/models_HBE_05_16_red_active_20231223/model_base")
    train_cpu_start  = time.process_time()
    train_wall_start = time.perf_counter()
    model.train(n_epochs=100,verbose=1)
    train_cpu  = time.process_time()  - train_cpu_start
    train_wall = time.perf_counter() - train_wall_start

    model.save(model_path,overwrite=True)

    with open(model_path/'times.txt','w',encoding='utf-8') as f:
        f.write(f"cpu_time_train,{train_cpu}\n"
                f"wall_time_train,{train_wall}\n"
                f"cpu_time_label,{label_cpu}\n"
                f"wall_time_label,{label_wall}\n"
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

for i in range(i_start,95):

    improve_cpu_start  = time.process_time()
    improve_wall_start = time.perf_counter()
    model_bis = improve(model,label_fn,PSpace,k=5)
    improve_cpu  = time.process_time() - improve_cpu_start
    improve_wall = time.perf_counter() - improve_wall_start

    train_cpu_start  = time.process_time()
    train_wall_start = time.perf_counter()
    model_bis.train(100,1)
    train_cpu  = time.process_time() - train_cpu_start
    train_wall = time.perf_counter() - train_wall_start

    model_path = save_path / f"model_improved_{str(i).zfill(3)}"

    model_bis.save(model_path,overwrite=True)
    with open(model_path / 'times.txt','w',encoding='utf-8') as f:
        f.write(f"cpu_time_train,{train_cpu}\n"
                f"wall_time_train,{train_wall}\n"
                f"cpu_time_improve {improve_cpu}:\n"
                f"wall_time_improve: {improve_wall}\n"
                )


    model = model_bis

model_final = model.finalize(n_epochs = 1000)
model_final.save(save_path / "model_final")
