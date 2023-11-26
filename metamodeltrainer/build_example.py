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



cwd = Path(__file__).resolve().parents[1]
save_path = Path(cwd,f"models_stress_HBE_05_16_active_{datetime.today().strftime('%Y%m%d')}")
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
label_fn = lambda S: label(S,prm_file = cur_dir / Path('../FE/data/prm/HBE_05_16.prm'))

S = PDskSample(PSpace, k = 2) # k indicates the number of points to sample
X_T, Y_T = label_fn(S)

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

model = RecModel(X_T,Y_T,HP)
model.train(n_epochs=100,verbose=1)
model.save(save_path / "model_base",overwrite=True)

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
    model_bis = improve(model,label_fn,PSpace,k=2)
    model_bis.train(100,1)
    model_bis.save(save_path / f"model_improved_{str(i).zfill(3)}",overwrite=True)
    model = model_bis

'''
STEP 6 : 6. Build and train a "final" model (and save it).
'''

model_final = model.finalize(n_epochs = 1000)
model_final.save(save_path / "model_final")
