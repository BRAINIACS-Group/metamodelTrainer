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
print(X_T.columns,Y_T.columns)
#Define hyper parameters
HP = HyperParameters(layers=[64,64],
                     loss='mae',
                     dropout_rate=0,
                     interpolation=1000)

#Build model
model = ForwardModel(X_T,Y_T,HP)
model.summary()

#Train model
model.train(5,verbose=1)


#Save model
name = "test"
model.save(Path(save_path,name))
model = load_model(Path(save_path,name))
Y_P =model.predict(X_T)
Y_P.reform()
print(np.mean(abs(Y_P-Y_T)))