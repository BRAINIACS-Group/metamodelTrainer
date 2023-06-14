import os
import numpy as np
import matplotlib.pyplot as mpl
from explore_param_space import *
from pathlib import Path
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, TimeDistributed
from keras.callbacks import EarlyStopping
import keras.backend as K
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import pickle
import scipy.interpolate as spi
from datetime import datetime

#Summary

#HyperParameters that will be used 
class HyperParameters(dict):
    
    def __init__(self,layers=[64,64,64],loss='mae',dropout_rate=0,window_size=100,interpolation=None):
        self._keys = set(["layers", "loss","dropout_rate","window_size","interpolation"])
        super().__setitem__("layers",layers)
        super().__setitem__("loss",loss)
        super().__setitem__("dropout_rate",dropout_rate)
        super().__setitem__("window_size",window_size)
        super().__setitem__("interpolation",interpolation)
        
        
    def __getitem__(self,key):
        if key not in self._keys:
            raise KeyError(f"Key '{key}' not found.")
        return super().__getitem__(key)
    
    def __setitem__(self, key, value):
        if key not in self._keys:
            raise KeyError(f"Key '{key}' not found.")
        return super().__setitem__(key, value)

#Interpolates new_time values from a given X, whose values are ordered by old_time
def interpolate(X,new_time,old_time=None,tii=False):
    res = []
    for i in range(X.n):
        P,curve = ExData(X[i],p=X.p,n=1).separate()
        if np.all(old_time) == None:
            time = curve[:,0]
        elif old_time.ndim != 2: time = old_time
        else: time = np.asarray(old_time)[i]
        if tii: curve = curve[:,1:]
        fn = spi.CubicSpline(time,curve)
        if new_time.ndim != 3:
            res.append(P.spread(fn(new_time))[0])
        else:
            res.append(P.spread(fn(new_time[i]))[0])
    return ExData(res,n=X.n,p=X.p)
        

#Generic model ; only serves as a SuperClass
#A Model contains the actual model, its training set X_T & Y_T, and preprocessing/postprocessing functions
#called before and after the prediction by the base model
class Model(): 
    
    def __init__(self,model,X_T,Y_T,preprocessX,preprocessY,postprocessY,summary,history):
        self.model = model
        self.X_T = X_T
        self.Y_T = Y_T
        self.preprocessX = preprocessX
        self.preprocessY = preprocessY
        self.postprocessY = postprocessY
        self.sum = summary
        self.history = history


    def train(self,n_epochs=100,verbose=1):
        if verbose == 1: verbose = 2 
        elif verbose == 2: verbose = 1
        preX_fn, preX_arg = self.preprocessX
        preY_fn, preY_arg = self.preprocessY
        Xs = preX_fn(self.X_T,*preX_arg)
        Ys = preY_fn(self.Y_T,self.X_T,*preY_arg)
        #cb = EarlyStopping(monitor='val_loss',restore_best_weights=True,patience=max(int(n_epochs/10),50),start_from_epoch=int(n_epochs))
        history = self.model.fit(Xs,Ys,epochs=n_epochs,verbose=verbose,batch_size=int(self.X_T.n/16),validation_split=0.1)
        self.history.append(history)
        self.sum += f"Trained for {n_epochs} epochs on {self.X_T.n} samples. Call obj.history[{str(len(self.history)-1)}] to see training history.\n"
        return history

    def predict(self,X,return_var=False):
        preX_fn, preX_arg = self.preprocessX
        postY_fn, postY_arg = self.postprocessY
        Xs = preX_fn(X,*preX_arg)
        if return_var:
            predictions = np.array([postY_fn(self.model(Xs,training=True),X,*postY_arg) for _ in range(16)])
            Y = np.mean(predictions,axis=0)
            V = np.var(predictions,axis=0)
            return ExData(Y,n=X.n), ExData(V,n=X.n)
        else:
            return postY_fn(self.model.predict(Xs,verbose=0),X,*postY_arg)

    def evaluate(self,X,Y,verbose=1):
        if verbose == 1: verbose = 2 
        elif verbose == 2: verbose = 1
        preX_fn, preX_arg = self.preprocessX
        preY_fn, preY_arg = self.preprocessY
        Xs = preX_fn(self.X_T,*preX_arg)
        Ys = preY_fn(self.Y_T,self.X_T,*preY_arg)
        return self.model.evaluate(Xs,Ys,verbose=verbose)

    def enrich(self,X_A,Y_A): #Adds values to the training set
        self.X_T = self.X_T.append(X_A)
        self.Y_T = self.Y_T.append(Y_A)

    def summary(self):
        print(self.sum)

    def save(self,name):
        if str(name)[-4:] != '.pkl':
            name = str(name) + '.pkl'

        data = {
            'model': self.model,
            'X_T': self.X_T,
            'Y_T': self.Y_T,
            'preprocessX': self.preprocessX,
            'preprocessY': self.preprocessY,
            'postprocessY': self.postprocessY,
            'summary': self.sum,
            'history': self.history
        }
        
        with open(name, 'wb') as f:
            pickle.dump(data, f)
        
        with open(name[:-4]+'.txt','w') as f:
            f.write(self.sum)
        

    

#####################
####FORWARD MODEL####
#####################

#WORK IN PROGRESS !!!

#Used for Hyperelastic data : one input = one output

def F_preX_fn(X,scalerX):
        X.flatten()
        return X.scale(scalerX)

def F_preY_fn(Y,X,scalerY):
        Y.flatten()
        return Y.scale(scalerY)

def F_postY_fn(Y,X,scalerY):
        return ExData(scalerY.inverse_transform(Y),n=X.n)


def ForwardModel(X_T,Y_T,HP = HyperParameters()):
    X_T.flatten()
    Y_T.flatten()

    model = Sequential()
    model.add(Dense(X_T.f, input_shape = (X_T.f,),activation='relu'))
    for n in HP['layers']:
        model.add(Dense(n,activation='relu'))
    model.add(Dropout(HP['dropout_rate']))
    model.add(Dense(Y_T.f))

    if HP['loss'] == 'mae': metric = 'mse'
    else: metric = 'mae'
        
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=HP['loss'], metrics = [metric])

    #Build Scalers
    scalerX = StandardScaler().fit(X_T)
    scalerY = StandardScaler().fit(Y_T)
    
    preprocessX = (F_preX_fn,[scalerX])
    preprocessY = (F_preY_fn,[scalerY])
    postprocessY = (F_postY_fn,[scalerY])

    summary = f"Forward neural network created on {datetime.now().strftime('%d-%m-%Y %H:%M')}.\n\n"
    summary += "Architecture :\n"
    summary += f"Input shape : ({str(X_T.f)},)\n"
    for i in range(len(HP['layers'])):
        summary += f"Layer {str(2*i+1)} : LSTM, {str(HP['layers'][i])} cells, activation = tanh\n"
        summary += f"Layer {str(2*i+2)} : Dropout, Rate = {str(HP['dropout_rate'])}\n"
    summary += f"Output Layer : Dense, {Y_T.f} neurons, activation = linear \n"
    summary += "Call obj.model.summary() for more details on architecture (provided by keras)\n\n"
    summary += f"Optimizer : Adam (learning rate = 0.001)\n"
    summary += f"Loss measure : {HP['loss']}\n\n"
    summary += f"Data is scaled before and after prediction using StandardScalers.\n\n"

    history = []

    return Model(model,X_T,Y_T,preprocessX,preprocessY,postprocessY,summary,history)


#####################
###RECURRENT MODEL###
#####################

#Used for Viscoelastic data : output depends on history (LSTM model)

def R_preX_fn(X,scalerX):
    X.reform()
    return X.scale(scalerX)

def R_preY_fn(Y,X,scalerY):
    Y.reform()
    return Y.scale(scalerY)

def R_postY_fn(Y,X,scalerY):
    Y = ExData(Y)
    return ExData(Y.scale_back(scalerY),n=X.n)

def R_preX_fn_I(X,scalerX,nt): return R_preX_fn(interpolate(X,nt),scalerX)
def R_preY_fn_I(Y,X,scalerY,nt): return R_preY_fn(interpolate(Y,nt,old_time=X[:,:,X.p]),X,scalerY)
def R_postY_fn_I(Y,X,scalerY,nt): return R_postY_fn(interpolate(ExData(Y,n=X.n),X[:,:,X.p],old_time=nt),X,scalerY)


def RecModel(X_T,Y_T,HP = HyperParameters()):

    X_T.reform()
    Y_T.reform()
        
        #Build Model
    model = Sequential()
    model.add(LSTM(HP['layers'][0], input_shape=(None,X_T.f),return_sequences=True))
    model.add(Dropout(HP['dropout_rate']))
    for n in HP['layers'][1:]:
        model.add(LSTM(n, return_sequences=True))
        model.add(Dropout(HP['dropout_rate']))
    model.add(TimeDistributed(Dense(Y_T.f)))

    if HP['loss'] == 'mae': metric = 'mse'
    else: metric = 'mae'
        
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=HP['loss'], metrics = [metric])

    X_T.flatten()
    scalerX = StandardScaler().fit(X_T)
    X_T.reform()

    Y_T.flatten()
    scalerY = StandardScaler().fit(Y_T)
    Y_T.reform()

    if HP['interpolation'] == None:
    
        preprocessX = (R_preX_fn,[scalerX])
        preprocessY = (R_preY_fn,[scalerY])
        postprocessY = (R_postY_fn,[scalerY])

    else:
        n = HP['interpolation']
        new_time = np.linspace(0,np.max(np.asarray(X_T)[0,:,X_T.p]),n)
        preprocessX = (R_preX_fn_I,[scalerX,new_time])
        preprocessY = (R_preY_fn_I,[scalerY,new_time])
        postprocessY = (R_postY_fn_I,[scalerY,new_time])

    summary = f"Recurrent neural network created on {datetime.now().strftime('%d-%m-%Y %H:%M')}.\n\n"
    summary += "Architecture :\n"
    summary += f"Input shape : (None, {str(X_T.f)})\n"
    for i in range(len(HP['layers'])):
        summary += f"Layer {str(2*i+1)} : LSTM, {str(HP['layers'][i])} cells, activation = tanh\n"
        summary += f"Layer {str(2*i+2)} : Dropout, Rate = {str(HP['dropout_rate'])}\n"
    summary += f"Output Layer : LSTM, {Y_T.f} cells, activation = tanh \n"
    summary += "Call obj.model.summary() for more details on architecture (provided by keras)\n\n"
    summary += f"Optimizer : Adam (learning rate = 0.001)\n"
    summary += f"Loss measure : {HP['loss']}\n\n"
    summary += f"Data is scaled before and after prediction using StandardScalers.\n"
    if HP['interpolation'] != None: summary += f"Data is interpolated using {str(HP['interpolation'])} timesteps between 0 and {np.max(np.asarray(X_T)[0,:,X_T.p])} seconds.\n"
    summary += '\n'

    history = []

    return Model(model,X_T,Y_T,preprocessX,preprocessY,postprocessY,summary,history)


####################
###SLIDING WINDOW###
####################

#WORK IN PROGRESS !!!

#A recurrent model, which preprocess the data using a sliding window method

def W_preX_fn(X,scalerX,size):
    W = X.sliding_window(size,strip=10,padded=True)
    return W.scale(scalerX)

def W_preY_fn(Y,X,scalerY):
    W = Y.sliding_window(1,strip=10,padded=False)
    return W.scale(scalerY)

def W_postY_fn(Y,X,scalerY):
    return ExData(Y.scale_back(scalerY),n=X.n)


def SWModel(X_T,Y_T,HP = HyperParameters()):

    X_T.reform()
    Y_T.reform()

    size = HP['window_size']
    '''
    WX = X_T.sliding_window(size,strip=10,padded=True)
    WY = Y_T.sliding_window(size,strip=10,padded=True)

    #self.X_T = ExData(np.concatenate((WX,WY),axis=2),X_T.p)
    X_T = WX
    Y_T = Y_T.sliding_window(1,strip=10,padded=False)'''

    #Test if shapes match

    model = Sequential()
    model.add(LSTM(HP['layers'][0], input_shape=(size,X_T.f),return_sequences=True))
    model.add(Dropout(HP['dropout_rate']))
    for n in HP['layers'][1:-1]:
        model.add(LSTM(n, return_sequences=True))
        model.add(Dropout(HP['dropout_rate']))
    model.add(LSTM(HP['layers'][-1]))
    model.add(Dropout(HP['dropout_rate']))
    model.add(Dense(Y_T.f))
        
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=HP['loss'], metrics = ['mse'])
    
    X_T.flatten()
    scalerX = StandardScaler().fit(X_T)
    X_T.reform()

    Y_T.flatten()
    scalerY = StandardScaler().fit(Y_T)
    Y_T.reform()

    preprocessX = (W_preX_fn,[scalerX,size])
    preprocessY = (W_preY_fn,[scalerY])
    postprocessY = (W_postY_fn,[scalerY])

    return Model(model,X_T,Y_T,preprocessX,preprocessY,postprocessY)


###################
####MEGA MODELS####
###################

#Several models are built using the given method and hyperparameters, trained consecutively ; predicted result
#is the average of all predicted results

class MegaModel():
    def __init__(self,X_T,Y_T,n=10,method='FNN',HP=HyperParameters()):
        if not(method in ['FNN', 'RNN', 'SW']):
            raise NotImplementedError("Please select 'FNN', 'RNN' or 'SW' as method.")

        self.X_T = X_T.copy()
        self.Y_T = Y_T.copy()

        self.HP = HP
        
        if method == 'FNN':
            self.models = [ForwardModel(self.X_T,self.Y_T,HP=HP) for i in range(n)]

        if method == 'RNN':
            self.models = [RecModel(self.X_T,self.Y_T,HP=HP) for i in range(n)]

        if method == 'SW':
            self.models = [SWModel(self.X_T,self.Y_T,HP=HP) for i in range(n)]
        
        
    def __len__(self):
        return len(self.models)

    def __getitem__(self,i):
        return self.models[i]

    def train(self,n_epochs=100,verbose=2):
        for i in range(len(self)):
            L = []
            history = self[i].train(n_epochs,verbose=verbose-1)
            L.append(history.history['loss'][-1])
            if verbose >= 1: print(f"Model {i+1} trained successfully. Training loss: {L[-1]}")
        if verbose >= 1: print(f"All models trained successfully. Average loss: {np.mean(L)}")

    def predict(self,X,return_var=False):
        predictions = np.array([m.predict(X) for m in self.models])
        Y = ExData(np.mean(predictions, axis = 0))
        V = np.var(predictions, axis = 0)
        if return_var: return Y,ExData(V)
        else: return Y

    def evaluate(self,X,Y):
        scalerY = self[0].scalerY
        Yp = self.predict(X).scale(scalerY)
        Ys = Y.scale(scalerY)
        method = self.HP['loss']
        if method == 'mse': loss = np.mean((Yp-Ys)**2)
        if method == 'mae': loss = np.mean(abs(Yp-Ys))
        print(f"Evaluation loss: {loss}")
        return loss

    def best_model(self,X,Y):
        L = [m.evaluate(X,Y) for m in self.models]
        return self[np.argmin(L)]

    def save(self,name):
        os.mkdir(name)
        for i in range(len(self)):
            self[i].save(Path(name,"Model_"+str(i).zfill(1+int(np.log10(len(self))))))

'''
    def query():

    def enrich():

    def active_learning():
    
'''

######################
### Load functions ###
######################

def load(name):
    if str(name)[-4:] == '.pkl':
        return load_single(name)
    else:
        return load_mega(name)

def load_single(name): #Loads a model from a given pickle file
    with open(name,'rb') as f:
        data = pickle.load(f)
    return Model(data['model'],
                 ExData(data['X_T']),ExData(data['Y_T']),
                 data['preprocessX'],data['preprocessY'],data['postprocessY'],
                 data['summary'],data['history'])

def load_mega(name):
    file_list = os.listdir(name)
    models = [load_single(Path(name,x)) for x in file_list]
    MM = MegaModel(models[0].X_T,models[0].Y_T)
    MM.models = models
    return MM
