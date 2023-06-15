#Carl imports
from explore_param_space import *
from file_process import *
from models import *

#Third-party imports
import numpy as np
import matplotlib.pyplot as mpl
from pathlib import Path
from uuid import uuid4

#Path definition
cwd = Path(__file__).resolve().parents[1]
data_path = Path(cwd,"data","LHCU_20")
save_path = Path(cwd,"models")

#Load the training data
X_T, Y_T = load_FE(data_path,
                   parameters = ['alpha','mu','deviatoric_20viscosity'],
                   inputs = ['time','displacement','angle'],
                   outputs = ['force','torque'])

#Define hyper parameters
HP = HyperParameters(layers=[64,64],
                     loss='mae',
                     dropout_rate=0,
                     interpolation=1000)

#Build model
model = RecModel(X_T,Y_T,HP)
model.summary()

#Train model
model.train(5,verbose=1)


#Save model
model.save(Path(save_path,"model_"+str(uuid4())[:8]))