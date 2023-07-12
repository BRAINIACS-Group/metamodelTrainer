from explore_param_space import *
from models import *

import numpy as np
import matplotlib.pyplot as mpl
import os
from sklearn.metrics import r2_score

#Path definition
cwd = Path(__file__).resolve().parents[1]
data_path = Path(cwd,"data","all_data")
save_path = Path(cwd,"models")
res_path = Path(cwd,"results")

X,Y = load_data(Path(cwd,'data','all_data_X.pkl')), load_data(Path(cwd,'data','all_data_Y.pkl'))

def update_load_model(name):
    if os.path.exists(Path(name,"model.h5")):
        return update_load_single(name)
    else:
        return update_load_mega(name)

def update_load_single(name): #Loads a model from a given folder
    model = klm(Path(name,"model.h5"))
    with open(Path(name,'aux.pkl'),'rb') as f:
        data = pickle.load(f)
    return Model(model,
                 ExData(data['X_T'],columns=['alpha_0','alpha_1','mu_0','mu_1','deviatoric_20viscosity_0','time','displacement','angle']),ExData(data['Y_T'],columns = ['force','torque']),
                 data['preprocessX'],data['preprocessY'],data['postprocessY'],
                 data['summary'])

def update_load_mega(name):
    file_list = os.listdir(name)
    models = [update_load_single(Path(name,x)) for x in file_list]
    MM = MegaModel(models[0].X_T,models[0].Y_T)
    MM.models = models
    MM.sum['method'] == str(len(MM))+'M'+MM.sum['method']
    return MM

def update_model(name):
    model = update_load_model(Path(save_path,name))
    model.sum['input_col'] = model.X_T.columns
    model.sum['output_col'] = model.Y_T.columns
    for m in model.models:
        m.sum['input_col'] = model.X_T.columns
        m.sum['output_col'] = model.Y_T.columns
    return model

model = load_model(Path(save_path,'ogden_final'))

model.run(Sample([[10,200,-5,700,1000]],columns = ['alpha_inf','mu_inf','alpha_1','mu_1','eta_1']))



#print(P,P.columns)

#HP = model.sum['HP']

#new_model = RecModel(X_T,Y_T,HP)
#new_model.train(100,1)

'''
def get_r2(name):
    if os.path.exists(Path(res_path,name+'.npy')):
        return np.load(Path(res_path,name+'.npy'))
    print("Computing "+name)
    model = update_model(Path(save_path,name))
    Y_P = model.predict(X)
    print('Done')
    Y_P.columns = ['force','torque']
    r2 = [r2_score(Y_P[i],Y[i]) for i in range(len(X))]
    np.save(Path(res_path,name),np.array(r2))
    return r2

AL = get_r2("MegaModel_AL_Final")
RL = get_r2("MegaModel_RL_Final")
print(np.mean(AL),np.std(AL))
print(np.mean(RL),np.std(RL))
mpl.subplot(1,2,1)
mpl.scatter([i for i in range(len(AL))], AL)
mpl.subplot(1,2,2)
mpl.scatter([i for i in range(len(RL))], RL)
mpl.show()'''
