'''

This script is an example of how to use a model, once built (see build_example to understand how to do that). Steps are :

1. Load the saved model.
2. Create the parameter set you want to get results from.
3. Predict results.

'''
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

from metamodeltrainer.explore_param_space import Sample
from metamodeltrainer.models import load_model



cwd = Path(__file__).resolve().parents[1]
save_path = Path(cwd,"models")

'''
STEP 1 : Load the saved model.

Fairly straightforward.

'''

model = load_model(Path(save_path,"model_final"))

'''
STEP 2 : Create the parameter set you want to get results from.

There are several ways one can go about this.
First, create a Sample object, containing the material parameters for which you want to get a prediction, 
with correctly named columns corresponding to those the model was trained on (check 'summary.txt' when in doubt)
Then, either create the ExData object yourself (see build_example step 2) and use model.predict in step 3, or 
use model.run in step 3.

NOTE : A sample should be a 2D array, so if you only want to test a single parameter set, make sur it is of shape
(n,1) and not (n,).
'''

#S = Sample([[5,200,-5,100,1000]],columns = ['alpha_inf','mu_inf','alpha_1','mu_1','eta_1'])
S = Sample([[-10,400,-5,300,5000]],columns = ['alpha_inf','mu_inf','alpha_1','mu_1','eta_1'])


'''
STEP 3 : Predict results.

If you created an ExData object (3D-array with proper columns), you can use model.predict(X). Otherwise, use 
model.run(S) (which is meant to mimic the behaviour of the FE simulation).

model.run(S) returns the name of a folder where the results of the prediction are saved. Some additional parameters can be
given as input :
- input_dir : where the script should look for input data (displacement & angle)
              default is '../FE/data/input/10.01.2022ALG_5_GEL_5_P2'
- output_dir : where the script should write the output data. If there are several parameter sets, the output dir will contain
              one subdirectory per set
              default is '../out/_____' (random 8 character sequence)
- parameter_file : an example parameter file that will be copied containing geometry information, for instance
             However, as I don't know how to use ParameterHandler, the actual material parameter values will
             be written in a separate file
             default is '../FE/data/prm/reference.prm'
'''
parameter_filepath = Path(__file__).parents[1] / "FE/data/prm/HBE_05_16.prm"
output_dir = Path(__file__).parents[1] / "out" / "HBE_05_16"
input_dir  = Path(__file__).parents[1] / "FE" / "data" / "input" / "HBE_05_16_P2a_FC_38"
path = model.run(S,parameter_file=parameter_filepath,output_dir=output_dir,
    input_dir=input_dir)
print(path)

