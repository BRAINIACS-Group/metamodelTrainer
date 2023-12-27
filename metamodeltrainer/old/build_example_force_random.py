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

from metamodeltrainer.explore_param_space import ParameterSpace, ExData, Sample, PDskSample  #Data structure & Sampling method
from metamodeltrainer.models import HyperParameters, RecModel, improve_random, load_model           #Neural Network management
from metamodeltrainer.run_simulation import label



cwd = Path(__file__).resolve().parents[1]
save_path = Path(cwd,
    f"models_force_HBE_05_16_random_{datetime.today().strftime('%Y%m%d')}")
if not save_path.is_dir():
    save_path.mkdir()   

'''
STEP 1 : Define a parameter space to explore

Use the ParameterSpace object. It takes any number of arguments and passes the names as keys and the values
as values in a dictionary, containing the ranges to explore.
'''

PSpace = ParameterSpace(
    alpha_inf = (-20,20),
    mu_inf    = (100,2000),
    alpha_1   = (-20,20),
    mu_1      = (100,2000),
    eta_1     = (0,10000)
)

'''
STEP 2 : Generate an initial sampling of the parameter space

Use the 'PDskSample' method to sample the parameter space (best method found), as well as the 'label' function
to get FE simulation results.

NOTE : For new constitutive model, you'll have to go into the run_simulation file and modify the 'run_sim' function
so that the correct FE simulation is used. I have not touched the code that much, so it should not be too complicated.
Also, the 'label' function works in a way that the simulation function from run_sim are saved in a folder, and then 
loaded back from that folder in the 'label' function. This might not be as robust as I hoped, so I would advise you 
try to see if using your own objects is more consistent.
The most important part is that the function should return two ExData objects, inputs and outputs, which is are
3-dimensional arrays equipped with names for the columns and a parameter p indicating how many material parameters there are.
For instance, in the example I worked on, you could load the FE results however you want into a 3D-array X of shape (n,t,f), where:
- n is the number of different samples
- t the number of time steps per sample (in the protocol)
- f the number of features (material parameters + inputs)
And then return ExData(X, p = 5, columns = ['alpha_inf','mu_inf','alpha_1','mu_1','viscosity','time','displacement','angle'])
where p = 5 indicates that the first 5 columns are material parameters
Similarly, you would return as well ExData(Y, p = 0, columns = ['force','torque'])
'''
cur_dir = Path(__file__).resolve().parent
label_fn = lambda S: label(S,
    prm_file = cur_dir / Path('../FE/data/prm/HBE_05_16.prm'),
    stress=False)

S = PDskSample(PSpace, k = 4) # k indicates the number of points to sample


'''
STEP 3 : Build and train a model equipped with an uncertainty indicator on this sampling.

Use 'RecModel' to build a recurrent neural network, after defining the desired HyperParameters. Setting 'dropout_rate' to a
non-zero value will allow for an uncertainty indicator to arise.
Training for 100 epochs seems sufficient to observe uncertainty trends arise.
'''

HP = HyperParameters(layers=[64,64],        #Architecture of the network
                     loss='mae',            #Loss to minimize
                     dropout_rate=0.5,      #Allows uncertainty
                     interpolation=1000)    #Faster computing

X_T, Y_T = label_fn(S)
model = RecModel(X_T,Y_T,HP)
train_cpu_start  = time.process_time()
train_wall_start = time.perf_counter()
model.train(n_epochs=100,verbose=1)
train_cpu  = time.process_time()  - train_cpu_start
train_wall = time.perf_counter() - train_wall_start

model_path = save_path / "model_base"
model.save(model_path,overwrite=True)

with open(model_path/'times.txt','w',encoding='utf-8') as f:
    f.write(f"cpu_time_train,{train_cpu}\n"
            f"wall_time_train,{train_wall}\n"
            )

'''
STEPS 4 & 5 : Use active learning to find new points to add to the training sample and train a new model on it.

As of now, there is no reliable way to determine whether enough points have been added, since acceptable loss values 
might depend on the fitted model, as well as acceptable training dataset size. I would advise to use your own metric
every once in a while to determine if you want to keep improving the model. However, to get accurate results, you
would need to have a model without dropout, and trained for much longer : 
Using model.finalize(n_epochs) will return such a model, trained for that long (n_epochs = 1000 is recommended)

The 'improve' function requires 4 arguments : the base model, a labeling function (see STEP 2 for the requirements),
the parameter space to sample, and a number of points k to add to the training sample.
It is good practice, although not necessary, to save the model at each iteration
'''

for i in range(99):
    model_path = save_path / f"model_improved_{str(i).zfill(3)}"
    improve_cpu_start = improve_wall_start = 0.
    train_cpu_start   = train_wall_start = 0.

    improve_cpu_start  = time.process_time()
    improve_wall_start = time.perf_counter()
    model_bis = improve_random(model,label_fn,PSpace,k=4)
    improve_cpu  = time.process_time() - improve_cpu_start
    improve_wall = time.perf_counter() - improve_wall_start

    train_cpu_start  = time.process_time()
    train_wall_start = time.perf_counter()
    model_bis.train(100,1)
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
