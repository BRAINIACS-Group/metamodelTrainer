import os
import numpy as np
from explore_param_space import *
from file_process import *
from pathlib import Path
from tensorflow import keras
from keras.models import Sequential
from keras.models import load_model as klm
from keras.layers import Dense, Dropout, LSTM, TimeDistributed
from keras.callbacks import EarlyStopping
import pickle
import scipy.interpolate as spi
from datetime import datetime
import pandas as pd


class StandardScaler:

    '''
    This was meant to emulate the sklearn.preprocessing StandardScaler, because sklearn wasn't installed on the clusters.
    I decided to keep it, as coding custom objects makes the whole pipeline more robust (I think).
    '''

    def __init__(self,mu=0,std=1):
        self.mu = mu
        self.std = std

    def fit(self,X):
        self.mu = np.mean(np.asarray(X),axis=0)
        self.std = np.std(np.asarray(X),axis=0)
        return StandardScaler(self.mu,self.std)

    def transform(self,X):
        return (np.array(X) - self.mu)/self.std
    
    def inverse_transform(self,X):
        return (np.array(X)*self.std) + self.mu

class HyperParameters(dict):
    
    '''
    The HyperParameters object is just a dictionary with the correct keys to adjust a few hyperparameters of the model :
    - 'layers' : a list of integers, corresponding to the number of neurons per layer
    - 'loss' : the loss indicator (either 'mae' or 'mse', usually)
    - 'dropout_rate' : Each hidden layer is followed by a dropout layer. Set to 0 if you want to ignore it ; or a value between 0 and 1 to activate it
    - 'window_size' : Only used for SWModel (sliding windows), define the size of the window (how many timesteps you look back)
    - 'interpolation' : Only used for RecModel (recurrent network), define how many equally spaced timesteps you want to interpolate
                        from the original protocol
    '''

    def __init__(self,layers=[64,64,64],loss='mae',dropout_rate=0,window_size=100,interpolation=None):
        self._keys = set(["layers", "loss","dropout_rate","window_size","interpolation"])
        super().__setitem__("layers",layers)
        super().__setitem__("loss",loss)
        super().__setitem__("dropout_rate",dropout_rate)
        super().__setitem__("window_size",window_size)
        super().__setitem__("interpolation",interpolation)
    
class Summary(dict):

    '''
    A Summary object is not something you should interact with. It serves to keep all (most) relevant information, including :
    - 'method' : The type of Network (feed-forward, recurrent, sliding window, megamodel...)
    - 'date' : date and time of creation
    - 'HP' : HyperParameters use at creation (see above)
    - 'input_shape' : the shape the input should be put in (Note : None = Any)
    - 'output_shape' : the shape the output will be in
    - 'input_col' : names and order of the input columns
    - 'output_col' : names and order of the output columns
    - 'max_inter' : only when interpolation is used, the maximum time for interpolation (results not guaranteed beyond)
    - 'training_history' : list of (e,n) tuples, indicating that the model was trained for e epochs on n samples

    Use model.summary() to print a formatted string containing all the info
    '''

    def __init__(self,
                method='FNN',
                date=datetime.now(),
                HP = HyperParameters(),
                input_shape=(None,1),
                output_shape=(None,1),
                input_col = None,
                output_col = None,
                max_inter=None,
                training_history=[],
                add_info = ""):
        self._keys = set(["method", "date","HP","input_shape","output_shape","input_col","output_col","max_inter","training_history","add_info"])
        super().__setitem__("method",method)
        super().__setitem__("date",date)
        super().__setitem__("HP",HP)
        super().__setitem__("input_shape",input_shape)
        super().__setitem__("output_shape",output_shape)
        super().__setitem__("input_col",input_col)
        super().__setitem__("output_col",output_col)
        super().__setitem__("max_inter",max_inter)
        super().__setitem__("training_history",training_history)
        super().__setitem__("add_info",add_info)

    def to_string(self):
        string = "\n"
        if self['method'] == 'FNN': string += "Forward Neural Network"
        elif self['method'] == 'RNN': string += "Recurrent Neural Network"
        elif self['method'] == 'SW': string += "Recurrent Neural Network"
        elif self['method'][-4:] == 'MFNN' : string += f"Forward Neural Network Mega-Model ({self['method'][:-4]} models)"
        elif self['method'][-4:] == 'MRNN' : string += f"Recurrent Neural Network Mega-Model ({self['method'][:-4]} models)"
        elif self['method'][-3:] == 'MSW' : string += f"Recurrent Neural Network Mega-Model ({self['method'][:-3]} models)"
        string += f" created on {self['date'].strftime('%d-%m-%Y %H:%M')}.\n\n"
        string += f"Input columns : {', '.join(self['input_col'])}\n"
        string += f"Output columns : {', '.join(self['output_col'])}\n"
        string += "Architecture :\n"
        string += f"Input shape : {str(self['input_shape'])}\n"
        if self['method'][-4:] == 'FNN':
            for i in range(len(self['HP']['layers'])):
                string += f"Layer {str(2*i+1)} : Dense, {str(self['HP']['layers'][i])} cells, activation = relu\n"
                string += f"Layer {str(2*i+2)} : Dropout, Rate = {str(self['HP']['dropout_rate'])}\n"
            string += f"Output Layer : Dense, {str(self['output_shape'])} neurons, activation = linear \n"
        elif self['method'][-4:] == 'RNN':
            for i in range(len(self['HP']['layers'])):
                string += f"Layer {str(2*i+1)} : LSTM, {str(self['HP']['layers'][i])} cells, activation = tanh\n"
                string += f"Layer {str(2*i+2)} : Dropout, Rate = {str(self['HP']['dropout_rate'])}\n"
            string += f"Output Layer : LSTM, {str(self['output_shape'])} cells, activation = tanh \n"
        elif self['method'][-3:] == 'SW':
            for i in range(len(self['HP']['layers'])):
                string += f"Layer {str(2*i+1)} : LSTM, {str(self['HP']['layers'][i])} cells, activation = tanh\n"
                string += f"Layer {str(2*i+2)} : Dropout, Rate = {str(self['HP']['dropout_rate'])}\n"
            string += f"Output Layer : LSTM, {str(self['output_shape'])} cells, activation = tanh \n"
        string += "Call obj.model.summary() for more details on architecture (provided by keras)\n\n"
        string += f"Optimizer : Adam (learning rate = 0.001)\n"
        string += f"Loss measure : {self['HP']['loss']}\n\n"
        string += f"Data is scaled before and after prediction using StandardScalers.\n"
        if self['HP']['interpolation'] != None and self['method'][-3:] == 'RNN': 
            string += f"Data is interpolated using {str(self['HP']['interpolation'])} timesteps between 0 and {str(self['max_inter'])} seconds.\n"
        if self['method'][-2:] == 'SW':
            string += f"Data is processed into sliding windows of width {self['HP']['window_size']}.\n"
        string += '\n'
        for i in range(len(self["training_history"])):
            string += f"Trained for {str(self['training_history'][i][0])} epochs on {str(self['training_history'][i][1])} samples.\n"
        string += '\n'    
        string += self['add_info']
        return string
        
#########################
####    MODELS !    #####
#########################


class Model(): 
    
    '''
    Compared to a keras model, this class comes with a few additional info/objects that make the whole thing more user-friendly :
    - 'model' : the actual keras model
    - 'X_T', 'Y_T' : training dataset of the model. Useful for analysis purposes when building, not sure if you will need them
    - 'preprocessX', 'preprocessY', 'postprocessY' : pre- and postprocessing functions ; necessary to get the correct results from the model
                                                     They come as tuples, containing the actual function and the additional arguments
                                                     that need to be passed (scalers, etc)
    - 'sum' : Summary object

    Note : predictions work on ExData objects, not any array, so make sure that the input are converted

    -> Build a model using Forward/Rec/SWModel(X_T,Y_T,HP) : training data and HyperParameters

    -> Use model.train(n_epochs) to train it for a certain number of epochs

    -> Use model.predict(X) to predict results from an ExData input. Add 'return_var = True' if you want to get the uncertainty of the model (only if HP['dropout_rate'] != 0)

    -> Use model.run(X,input_dir) to emulate a FE simulation by giving a set of material parameters X and a directory from which to get input displacement and torque
      Returns the path to the output directory

    -> Use model.drop() to remove the training dataset that is usually saved with the model. Only do this on trained models,
    but it can save a lot of data space.

    -> Use model.save(path) to save it to the given path. See below for more detailed explanations

    -> Use model.comment(str) to add whatever comment you want at the end of the summary

    -> Use model.enrich(X,Y) to append new training data to the existing one (don't think this will be useful)

    -> Use model.summary() to print out the summary.

    '''

    def __init__(self,model,X_T,Y_T,preprocessX,preprocessY,postprocessY,summary):
        self.model = model
        self.X_T = X_T
        self.Y_T = Y_T
        self.preprocessX = preprocessX
        self.preprocessY = preprocessY
        self.postprocessY = postprocessY
        self.sum = summary


    def train(self,n_epochs=100,verbose=1):
        if verbose == 1: verbose = 2 
        elif verbose == 2: verbose = 1
        preX_fn, preX_arg = self.preprocessX
        preY_fn, preY_arg = self.preprocessY
        Xs = preX_fn(self.X_T,*preX_arg)
        Ys = preY_fn(self.Y_T,self.X_T,*preY_arg)
        #cb = EarlyStopping(monitor='val_loss',restore_best_weights=True,patience=max(int(n_epochs/10),50),start_from_epoch=int(n_epochs))
        history = self.model.fit(Xs,Ys,epochs=n_epochs,verbose=verbose,batch_size=int(len(Xs)/16),validation_split=0.1)
        self.sum['training_history'].append((n_epochs,self.X_T.n))
        return history

    def predict(self,X,return_var=False):
        if X.columns != self.sum['input_col']:
            raise ValueError(f"Input columns do not match training columns. Expected {str(self.sum['input_col'])}, got {str(X.columns)} instead.")
        preX_fn, preX_arg = self.preprocessX
        postY_fn, postY_arg = self.postprocessY
        Xs = preX_fn(X,*preX_arg)
        if return_var:
            predictions = np.array([postY_fn(self.model(Xs,training=True),X,*postY_arg) for _ in range(16)])
            Y = np.mean(predictions,axis=0)
            V = np.var(predictions,axis=0)
            return ExData(Y,n=X.n,columns=self.sum['output_col']), ExData(V,n=X.n,columns = self.sum['output_col'])
        else:
            return ExData(postY_fn(self.model.predict(Xs,verbose=0),X,*postY_arg),n=X.n,columns = self.sum['output_col'])

    def run(self,S,input_dir = Path(Path(__file__).resolve().parents[1],'FE','data','input','10.01.2022ALG_5_GEL_5_P2'),
                output_dir = Path(Path(__file__).resolve().parents[1],'out',str(uuid4())[:8]),
                parameter_file = Path(Path(__file__).resolve().parents[1],'FE','data','prm','reference.prm')):
        columns = self.sum['input_col'][self.X_T.p:] + self.sum['output_col']
        dataset = pd.DataFrame(columns=columns)
        default = {'time': 0, 'displacement': 0, 'force': 0, 'angle': 0, 'torque': 0}
        #Get data
        for file in os.listdir(input_dir):
            if file.endswith('.csv'):
                df = pd.read_csv(Path(input_dir, file),dtype=np.float32).rename(columns = {' displacement': 'displacement', ' force':'force', ' angle':'angle', ' torque':'torque'})
                df = pd.DataFrame({**default, **df},dtype=np.float32)
                dataset = pd.concat((dataset, df), ignore_index=True).sort_values(by=["time"])
        #Convert into usable ExData
        P = S
        X = P.spread(np.array(dataset[self.sum['input_col'][self.X_T.p:]]).reshape(len(dataset),len(self.sum['input_col'][self.X_T.p:])),input_columns=self.sum['input_col'][self.X_T.p:])
        Y = self.predict(X)
        res_to_file(X,Y,input_dir,output_dir,parameter_file)
        return output_dir

    def finalize(self,n_epochs = 1000):
        X_T, Y_T, HP = self.X_T, self.Y_T, self.sum['HP']
        HP['dropout_rate'] = 0
        if self.sum['method'] == 'RNN' : model = RecModel(X_T,Y_T,HP)
        if self.sum['method'] == 'FNN' : model = ForwardModel(X_T,Y_T,HP)
        if self.sum['method'] == 'SW' : model = SWModel(X_T,Y_T,HP)
        model.train(n_epochs,1)
        return model

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

    def comment(self,comment):
        self.sum['add_info'] += comment
        self.sum['add_info'] += ""

    def summary(self):
        print(self.sum.to_string())

    def drop(self):
        self.X_T.reform()
        self.Y_T.reform()
        self.X_T = self.X_T[0]
        self.Y_T = self.Y_T[0]

    def save(self,name):

        '''
        Saving a model uses the native keras save method to save the contained model to a 'model.h5' file ; it will save all 
        additional information (training set, processing functions, summary) to a 'aux.pkl' file using pickle, and write out
        the summary string to a 'summary.txt' file. All three are contained within the specified destination folder.

        Use model = load_model(path) to load it back
        '''

        if not(os.path.exists(name)): os.mkdir(name)
        self.model.save(Path(name,"model.h5"))
        self.X_T.reform()
        self.Y_T.reform()
        data = {
                'X_T': self.X_T,
                'Y_T': self.Y_T,
                'preprocessX': self.preprocessX,
                'preprocessY': self.preprocessY,
                'postprocessY': self.postprocessY,
                'summary': self.sum,
            }
        with open(Path(name,'summary.txt'),'w') as f:
            f.write(self.sum.to_string())

        with open(Path(name,"aux.pkl"), 'wb') as f:
            pickle.dump(data, f)


#####################
####FORWARD MODEL####
#####################

#Used for Hyperelastic data : one input = one output. Is very fast

#Pre/post processing functions:

def F_preX_fn(X,scalerX):
        X.flatten()
        return X.scale(scalerX)

def F_preY_fn(Y,X,scalerY):
        Y.flatten()
        return Y.scale(scalerY)

def F_postY_fn(Y,X,scalerY):
        return ExData(scalerY.inverse_transform(Y),n=X.n)

# Model creation

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

    summary = Summary(method = 'FNN',
                      date = datetime.now(),
                      HP = HP,
                      input_shape=(X_T.f,),
                      output_shape = Y_T.f,
                      input_col = X_T.columns,
                      output_col = Y_T.columns,
                      training_history=[])

    return Model(model,X_T,Y_T,preprocessX,preprocessY,postprocessY,summary)


#####################
###RECURRENT MODEL###
#####################

#Used for Viscoelastic data : output depends on history (LSTM model)

#Pre/post processing functions :

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
def R_preY_fn_I(Y,X,scalerY,nt): return R_preY_fn(interpolate(Y,nt,old_time=X[:,:,X.columns.index('time')]),X,scalerY)
def R_postY_fn_I(Y,X,scalerY,nt): return R_postY_fn(interpolate(ExData(Y,n=X.n),X[:,:,X.columns.index('time')],old_time=nt),X,scalerY)


#Interpolates new_time values from a given X, whose values are ordered by old_time
def interpolate(X,new_time,old_time=None):
    res = []
    X.reform()
    for i in range(X.n):
        P,curve = X[i].separate()
        if np.all(old_time) == None:
            time = curve[:,X.columns.index('time')-X.p]
        elif old_time.ndim != 2: time = old_time
        else: time = np.asarray(old_time)[i]
        fn = spi.CubicSpline(time,curve)
        if new_time.ndim != 3:
            res.append(P.spread(fn(new_time),input_columns=X.columns[X.p:])[0])
        else:
            res.append(P.spread(fn(new_time[i]),input_columns=X.columns[X.p:])[0])
    return ExData(res,n=X.n,p=X.p,columns=X.columns)

#Model creation :

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
        new_time = np.linspace(0,np.max(np.asarray(X_T)[0,:,X_T.columns.index('time')]),n)
        preprocessX = (R_preX_fn_I,[scalerX,new_time])
        preprocessY = (R_preY_fn_I,[scalerY,new_time])
        postprocessY = (R_postY_fn_I,[scalerY,new_time])

    summary = Summary(method = 'RNN',
                      date = datetime.now(),
                      HP = HP,
                      input_shape=(None,X_T.f),
                      output_shape = Y_T.f,
                      max_inter=np.max(np.asarray(X_T)[0,:,X_T.columns.index('time')]),
                      training_history=[],
                      input_col = X_T.columns,
                      output_col = Y_T.columns)

    return Model(model,X_T,Y_T,preprocessX,preprocessY,postprocessY,summary)


####################
###SLIDING WINDOW###
####################

#A recurrent model, which preprocess the data using a sliding window method
#Note : DO NOT USE, the results are quite disappointing

#Pre/post processing functions : 

def W_preX_fn(X,scalerX,size):
    W = X.sliding_window(size,padded=True)
    return W.scale(scalerX)

def W_preY_fn(Y,X,scalerY):
    W = Y.sliding_window(1,padded=False)
    return W.scale(scalerY)

def W_postY_fn(Y,X,scalerY):
    Y = ExData(Y,n=X.n)
    return ExData(Y.scale_back(scalerY),n=X.n)

#Model Creation:

def SWModel(X_T,Y_T,HP = HyperParameters()):

    X_T.reform()
    Y_T.reform()

    size = HP['window_size']

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

    summary = Summary(method = 'SW',
                      date = datetime.now(),
                      HP = HP,
                      input_shape=(None,X_T.f),
                      output_shape = Y_T.f,
                      input_col = X_T.columns,
                      training_history=[],
                      output_col = Y_T.columns)

    return Model(model,X_T,Y_T,preprocessX,preprocessY,postprocessY,summary)


###################
####MEGA MODELS####
###################

class MegaModel():

    '''
    A "MegaModel" (unofficial name) is a collection of several models as described above, which are trained/used collectively
    to allow a measure of uncertainty (through variance). They are all created equal, the difference will stem from the training
    routines they each go through

    A MegaModel can be used in the same way as a regular Model, with the exception of its creation :
    -> use MegaModel(X_T,Y_T,n,method,HP) with X_T,Y_T the training set, n the number of models, method the type of models
    (either 'FNN' for Feed-Forward, 'RNN' for Recurrent, 'SW' for Sliding-Window) ; and HP the HyperParameters
    '''

    def __init__(self,X_T,Y_T,n=10,method='RNN',HP=HyperParameters()):
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

        self.sum = self.models[0].sum
        self.sum['method'] = str(n)+'M'+method
        
        
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
        self.sum['training_history'].append((n_epochs,self.X_T.n))

    def predict(self,X,return_var=False):
        predictions = np.array([m.predict(X) for m in self.models])
        Y = ExData(np.mean(predictions, axis = 0),columns=self.sum['output_col'])
        V = np.var(predictions, axis = 0)
        if return_var: return Y,ExData(V,columns=self.sum['output_col'])
        else: return Y

    def run(self,S,input_dir = Path(Path(__file__).resolve().parents[1],'FE','data','input','10.01.2022ALG_5_GEL_5_P2'),
                output_dir = str(uuid4())[:8],
                parameter_file = Path('../FE/data/prm/reference.prm')):
        columns = self.sum['input_col'][self.X_T.p:] + self.sum['output_col']
        dataset = pd.DataFrame(columns=columns)
        default = {'time': 0, 'displacement': 0, 'force': 0, 'angle': 0, 'torque': 0}
        #Get data
        for file in os.listdir(input_dir):
            if file.endswith('.csv'):
                df = pd.read_csv(Path(input_dir, file),dtype=np.float32).rename(columns = {' displacement': 'displacement', ' force':'force', ' angle':'angle', ' torque':'torque'})
                df = pd.DataFrame({**default, **df},dtype=np.float32)
                dataset = pd.concat((dataset, df), ignore_index=True).sort_values(by=["time"])
        #Convert into usable ExData
        P = S
        X = P.spread(np.array(dataset[self.sum['input_col'][self.X_T.p:]]).reshape(len(dataset),len(self.sum['input_col'][self.X_T.p:])),input_columns=self.sum['input_col'][self.X_T.p:])
        Y = self.predict(X)
        res_to_file(X,Y,input_dir,output_dir,parameter_file)
        return output_dir

    def evaluate(self,X,Y):
        scalerY = self[0].scalerY
        Yp = self.predict(X).scale(scalerY)
        Ys = Y.scale(scalerY)
        method = self.HP['loss']
        if method == 'mse': loss = np.mean((Yp-Ys)**2)
        if method == 'mae': loss = np.mean(abs(Yp-Ys))
        print(f"Evaluation loss: {loss}")
        return loss
    
    def summary(self):
        print(self.sum.to_string())

    def best_model(self,X,Y):
        L = [m.evaluate(X,Y) for m in self.models]
        return self[np.argmin(L)]

    def drop(self):
        self.X_T.reform()
        self.Y_T.reform()
        self.X_T = self.X_T[0]
        self.Y_T = self.Y_T[0]

    def save(self,name):

        '''
        Saving a MegaModel is similar to saving a regular Model : it creates the destination folder, then saves each model within
        '''

        if not os.path.exists(name): os.mkdir(name)
        
        for i in range(len(self)):
            self[i].save(Path(name,"Model_"+str(i).zfill(1+int(np.log10(len(self))))))


######################
### Load functions ###
######################

#load_model(path) will (supposedly) detect whether the destination folder contains a MegaModel or a regular Model, and load it accordingly

def load_model(name):
    if os.path.exists(Path(name,"model.h5")):
        return load_single(name)
    else:
        return load_mega(name)

def load_single(name): #Loads a model from a given folder
    model = klm(Path(name,"model.h5"))
    with open(Path(name,'aux.pkl'),'rb') as f:
        data = pickle.load(f)
    return Model(model,
                 ExData(data['X_T'],columns = data['summary']['input_col']),ExData(data['Y_T'],columns = data['summary']['output_col']),
                 data['preprocessX'],data['preprocessY'],data['postprocessY'],
                 data['summary'])

def load_mega(name):
    file_list = os.listdir(name)
    models = [load_single(Path(name,x)) for x in file_list]
    MM = MegaModel(models[0].X_T,models[0].Y_T)
    MM.models = models
    MM.sum['method'] == str(len(MM))+'M'+MM.sum['method']
    return MM

#######################
### Active learning ###
#######################

# The crux of the active learning routine resides in the following improve function. The goal is to generate
# a new and improved version of a given model. It takes 5 inputs :
#  - 'model' : the base model, that needs to be improved
#  - 'label_fn' : the labeling function. It should take a Sample object as inputs (array of parameter sets)
#                 and return TWO ExData objects, X and Y, containing the corresponding inputs and outputs
#                 to be used for the model 
#  - 'PSpace' : a ParameterSpace object, which will inform where to sample the new points
#  - 'k' : the number of new points to be sampled
#  - 'pool_size' : size of the pool of randomly generated points from which the new ones are selected (not necessary)

# The way it works is :
#  1. Generate 'pool_size' points using one of the sampling methods (PoissonDisk is the best it seems)
#  2. Evaluate the uncertainty of the given model on those new points (using the return_var parameter in predict)
#  3. Rank them from most uncertain to least uncertain
#  4. Select the top 'k' most uncertain points that are not too close to the existing training points
#  5. Get the actual targets for them using 'label_fn'
#  6. Using the new, larger training set, train a new model by copying the HyperParameters of the previous one, and its training history
#  7. Return the new, trained model

def improve(model,label_fn,PSpace,k=10,pool_size=None):
    if pool_size == None: pool_size = min(k*50,500)
    X_T = model.X_T
    Y_T = model.Y_T
    P_T,inputs = X_T.separate()
    HP = model.sum['HP']

    P = PDskSample(PSpace,pool_size)
    
    X = P.spread(inputs, input_columns = X_T.columns[X_T.p:])
    Y,V = model.predict(X,return_var=True)
    V = np.mean(V,axis=(1,2))
    I = np.argsort(-V) #Indexes corresponding to V sorted in descending order
    for j in range(k):
        i = 0
        while distance_to_sample(P[I[i]],P_T,PSpace) < min(1/X_T.n,0.5*min_distance(P_T,PSpace)): 
            i += 1
            if i == len(I): 
                i = j
                break
        P_T = P_T.append(P[I[i]])
    print("Start labeling")

    X_A, Y_A = label_fn(P_T[X_T.n:])
    
    X_T = X_T.append(X_A)
    Y_T = Y_T.append(Y_A)

    if model.sum['method'] == 'FNN': new_model = ForwardModel(X_T,Y_T,HP)
    elif model.sum['method'] == 'RNN': new_model = RecModel(X_T,Y_T,HP)
    elif model.sum['method'] == 'SW': new_model = SWModel(X_T,Y_T,HP)
    elif model.sum['method'][-4:] == 'MFNN':  new_model = MegaModel(X_T,Y_T,len(model),'FNN',HP)
    elif model.sum['method'][-4:] == 'MRNN': new_model = MegaModel(X_T,Y_T,len(model),'RNN',HP)
    elif model.sum['method'][-3:] == 'MSW': new_model = MegaModel(X_T,Y_T,len(model),'SW',HP)

    #new_model.comment("This model was generated by an active improvement function.")

    for x in model.sum['training_history']:
        new_model.train(x[0],1)
    
    return new_model

'''improve_random does the same thing, but adds random points'''

def improve_random(model,label_fn,PSpace,k=8,pool_size=None):
    if pool_size == None: pool_size = min(k*50,500)
    X_T = model.X_T
    Y_T = model.Y_T
    P_T,inputs = X_T.separate()
    HP = model.sum['HP']

    P = LHCuSample(PSpace,pool_size)
    P_T,inputs = X_T.separate()
    X = P.spread(inputs)
    Y = model.predict(X,return_var=False)
    for j in range(k):
        X = RandSample(PSpace,1)
        i = 0
        while distance_to_sample(X[0],P_T,PSpace) < min(0.5/X_T.n,0.5*min_distance(P_T,PSpace)): 
            X = RandSample(PSpace,1)
            i += 1
            if i == 10*k:
                i = j
                break
        P_T = P_T.append(X)
    
    X_A, Y_A = label_fn(P_T[X_T.n:])
    
    X_T = X_T.append(X_A)
    Y_T = Y_T.append(Y_A)

    if model.sum['method'] == 'FNN':  new_model = ForwardModel(X_T,Y_T,HP)
    elif model.sum['method'] == 'RNN': new_model = RecModel(X_T,Y_T,HP)
    elif model.sum['method'] == 'SW': new_model = SWModel(X_T,Y_T,HP)
    elif model.sum['method'][-4:] == 'MFNN':  new_model = MegaModel(X_T,Y_T,len(model),'FNN',HP)
    elif model.sum['method'][-4:] == 'MRNN': new_model = MegaModel(X_T,Y_T,len(model),'RNN',HP)
    elif model.sum['method'][-3:] == 'MSW': new_model = MegaModel(X_T,Y_T,len(model),'SW',HP)

    #new_model.comment("This model was generated by a random improvement function.")

    new_model.train(100,1)
    
    return new_model

