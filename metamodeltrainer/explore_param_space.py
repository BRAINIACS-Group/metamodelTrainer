'''Builds the Sample and ExData classes, that are to be handled by the models
The Sample class is used to sample the parameter space : several initialization functions (sample methods) are given
The ExData is used to handle experimental data from FE files
'''

#STL imports
import random
import pickle

#3rd party imports
import matplotlib.pyplot as mpl
import numpy as np
from scipy.stats.qmc import PoissonDisk,LatinHypercube

ref_input = np.linspace(0.80,1.2,50).reshape(50,1)

class ParameterSpace(dict):
    '''
    A parameter space object is a dictionary, with keys corresponding to names of axis (= parameters) 
    and values the corresponding range to explore
    '''
    def __init__(self,**kwargs):
        for k,range in kwargs.items():
            self[k] = range


#########################
### Sample Definition ###
#########################

'''
A Sample is, in essence, a 2D array containing coordinates within the parameter space.
It is only a custom object for practicity's sake when coding, adding a few functions here and there,
as well as a columns list, which names the axis of the coordinates.
Since it is a subclass of np.ndarray, all of the usual functions work
'''


class Sample(np.ndarray):
    def __new__(cls, A=[[]],columns = None): #Creates a Sample object from an array-type object
        obj = np.asarray(A).view(cls)
        if columns == None:
            columns = ["dim_"+str(i).zfill(len(str(obj.shape[-1]))) for i in range(obj.shape[-1])]
        obj.columns = columns
        return obj

    def __array_finalize__(self, obj): #Makes slicing compatible
        if not (hasattr(self,'columns')) and type(obj) != np.ndarray:
            self.columns = obj.columns

    def append(self,X): #Appends a Sample to another (unlike lists, returns the results)
        return Sample(np.vstack((self,X)),self.columns) 

    def plot(self,name = "Sample"): #Scatter plots the first two axis
        mpl.scatter(self[:,0], self[:,1], label = name)
        ax = mpl.gca()

    def draw(self,name = "Sample"): #Same, but displays the result
        self.plot(name)
        mpl.show()

    def copy(self): #Creates a deepcopy
        return Sample(np.copy(self),self.columns)

    def map(self,f): #Apply a function to each sampled point
        res = np.apply_along_axis(lambda x:f(*x),1,self)
        if res.ndim == 1: res = np.vstack(res)
        return res

    def delete(self,X): #Deletes all values that are present in X ; returns the result
        mask = ~np.all(np.isin(self,X),axis=1)
        return self[mask] #Not sure if working

    def scale(self,scaler): #Uses a sklearn.preprocessing Scaler to scale itself
        return scaler.transform(self)

    def save(self,name): #Save the Sample in an pickle file
        data = {
            'type' : "Sample",
            'array' : np.asarray(self),
            'columns' : self.columns
        }
        if str(name)[-4:] != '.pkl':
            name = str(name) + '.pkl'
        with open(name, 'wb') as f:
            pickle.dump(data, f)


    def spread(self,input_data = ref_input, input_columns = None): 

        #Given some input data, creates a 3D ExData object containing, for each parameter set, a 2D array with:
        #- The material parameters being repeated as many times as there are lines in input_data
        #- The input_data
        #Note : input_data is supposed to be in column shape

        if input_columns == None:
            input_columns = ["input_"+str(i).zfill(len(str(input_data.shape[-1]))) for i in range(input_data.shape[-1])]

        if input_data.ndim == 2:
            n, t = len(self),len(input_data)
            p = self.shape[1]
            
            if p == 0: X = np.tile(input_data,(n,1))
            else: X = np.hstack((np.repeat(self,t,axis=0), np.tile(input_data,(n,1))))        

            f = X.shape[1]
            return ExData(X.reshape(n,t,f),p=p,columns=self.columns+input_columns)
        else:
            m, t, k = input_data.shape
            n, p = self.shape
            X = Sample(self[0].reshape((1,-1)),self.columns).spread(input_data[0])
            for i in range(1,n):
                X = X.append(self[i].reshape((1,-1)).spread(input_data[i%k]))
            return X
             
########################
### Sampling Methods ###
########################

def scale_to(X,R0,R1): 
    #Scales a given sampling from R0 ranges to R1 ranges
    for i in range(len(R1)):
        X[:,i] -= R0[i][0]
        X[:,i] *= (R1[i][1]-R1[i][0])/(R0[i][1]-R0[i][0])
        X[:,i] += R1[i][0]       
    
    return X

def RandSample(PSpace = ParameterSpace(dim_0 = (0,1), dim_1 = (0,1)), k = 100): #Samples k points in a n-dim space randomly
    Points = []
    for i in range(k):
        Points.append([random.random() for j in range(len(PSpace))])
    return Sample(scale_to(np.array(Points),
        [(0,1) for _ in range(len(PSpace))],
        list(PSpace.values())),
        columns = list(PSpace.keys()))

def GridSample(PSpace = ParameterSpace(dim_0 = (0,1), dim_1 = (0,1)),p=5): #Samples a random point in each square of a n-dim grid with p subdivisions
    n = len(PSpace)
    Points = []
    for i in range(p**n):
        Points.append([random.random()/p + ((i%(p**(j+1)))//p**j)*1/p for j in range(n)])
    return Sample(scale_to(np.array(Points),[(0,1) for _ in range(len(PSpace))],list(PSpace.values())),columns = list(PSpace.keys()))

def LHCuSample(PSpace = ParameterSpace(dim_0 = (0,1), dim_1 = (0,1)),k=100): #Creates a Latin Hypercube from k equal divisions of the n-dim cube
    #n = len(PSpace)
    # grid = np.empty((k,n))
    # for i in range(n):
    #     grid[:,i] = np.random.permutation(k)
        
    # Points = []
    # for i in range(k):
    #     Points.append([(grid[i,j]+random.random())/k for j in range(n)])
    sampler = LatinHypercube(d=len(PSpace))
    points = sampler.random(k)

    return Sample(scale_to(points,
        [(0,1) for _ in range(len(PSpace))],
        list(PSpace.values())),
        columns = list(PSpace.keys()))


def PDskSample(PSpace = ParameterSpace(dim_0 = (0,1), dim_1 = (0,1)),k=100): #Uses the PoissonDisk method to sample k points with maximum coverage
    n = 0
    r = np.sqrt(len(PSpace))
    while n < k:
        engine = PoissonDisk(d=len(PSpace), radius=r, hypersphere='surface')
        Points = engine.random(k)
        n = len(Points)
        r *= 0.9

    return Sample(
        scale_to(np.array(Points),
            [(0,1) for _ in range(len(PSpace))],
            list(PSpace.values())),
            columns = list(PSpace.keys()))

def distance(point_a,point_b,PSpace): #Returns the normalized distance between two points (when scaled to a unit square)
    ranges = list(PSpace.values())
    point_a_norm = scale_to(np.array([point_a]),ranges,
        [(0,1) for _ in range(len(ranges))])[0]
    point_b_norm = scale_to(np.array([point_b]),ranges,
        [(0,1) for _ in range(len(ranges))])[0]

    #return euclidean distance in normalized space
    return np.sqrt(np.sum((point_a_norm-point_b_norm)**2))

def distance_to_sample(point_a,X:Sample,PSpace):
    #Returns the minimum distance from a point to a set of points
    distances = [distance(point_a,X[i],PSpace) for i in range(len(X))]
    return min(distances)

def avg_distance(X:Sample,PSpace):
    #Returns the average distance between all points of a sample
    distances = [distance(X[i],X[j],PSpace) for j in range(len(X))
        for i in range(len(X)) if i != j]
    return np.mean(distances)

def min_distance(X:Sample,PSpace):
    #Returns the minimal distance between two points of a sample
    distances = [distance(X[i],X[j],PSpace) for j in range(len(X))
        for i in range(len(X)) if i != j]
    return min(distances)

#########################
### ExData definition ###
#########################

class ExData(Sample):
    '''
    The ExData object is basically a numpy array, enriched with some extra information:
    - self.n : the number of experimental simulations it is supposed to contain
    - self.t : the number of time-steps per simulation
    - self.f : the number of features per time-step (typically, material parameters and simulation input data, or simulation output data)
    - self.p : among those features, how many are actually material parameters
    - self.columns : name of the columns

    One important aspect is the use of obj.flatten() and obj.reform(), which give two representations of the data
    Sometimes, it is necessary to have all data (from all parameter sets) in a single 2D array, which can be obtained using flatten
    (this applies to Feed-Forward Network, as well as scaling using scalers)
    Other times, a 3D array is needed, to separate the different samples/parameter sets (for Recurrent Neural Networks)
    Those two functions can switch from one to the other

    The .separate() method acts as inverse to the Sample.spread() : it returns a Sample containing all parameter sets, as well as
    the variable inputs in columns (time, displacement, etc)
    '''

    def __new__(cls,X=np.array([ref_input]),p=None,n=None,columns=None):
        '''Creates a new one from an array
        '''
        X = np.asarray(X)
        obj = X.view(cls)
        if len(X.shape) == 3:
            if n != None:
                obj.n, obj.t, obj.f = n, int(X.shape[0]*X.shape[1]/n), X.shape[2]
                X = X.reshape(obj.n,obj.t,obj.f)
            else: obj.n,obj.t,obj.f = X.shape
        elif X.ndim == 1:
            obj.n, obj.t, obj.f = 1, len(X), 1
        else:
            obj.n = n
            obj.t = int(X.shape[0]/n)
            obj.f = X.shape[1]
        if p == None:
            if len(X.shape) == 3: p = sum(np.apply_along_axis(lambda x: len(np.unique(x)), axis=0, arr=X[0]) == 1)
            else : p = sum(np.apply_along_axis(lambda x: len(np.unique(x)), axis=0, arr=X[:obj.t]) == 1)
        obj.p = p

        if columns == None:
            columns = ["dim_"+str(i).zfill(len(str(p))) for i in range(p)]+["input_"+str(i).zfill(len(str(obj.f-p))) for i in range(obj.f-p)]

        obj.columns = columns

        return obj
    
    def __array_finalize__(self, obj): #Makes slicing compatible
        if not(obj is None) and self.ndim > 0:
            if not(hasattr(self,'n')):
                if self.ndim == 3:
                    self.n,self.t,self.f = self.shape
                elif self.ndim == 2:
                    self.n,self.t,self.f = 1,self.shape[0],self.shape[1]
                else:
                    self.n,self.t,self.f = 1,1,self.shape[0]
            if not(hasattr(self,'p')):
                try:
                    if self.ndim == 3: p = sum(np.apply_along_axis(lambda x: len(np.unique(x)), axis=0, arr=np.asarray(self)[0]) == 1)
                    elif self.ndim == 2 : p = sum(np.apply_along_axis(lambda x: len(np.unique(x)), axis=0, arr=np.asarray(self)[:self.t]) == 1)
                    else: p = 0
                except: p = 0
                if p == 6: p = 5
                self.p = p
            if not (hasattr(self,'columns')) and type(obj) != np.ndarray and hasattr(obj,'columns'):
                self.columns = obj.columns


    def flatten(self): #Goes into 2D shape (used in Forward Neural Networks)
        if len(self.shape) == 3:
            self.shape = (self.n*self.t,self.f)
        
    
    def reform(self): #Goes into 3D shape (used in Recurrent Neural Networks)
        if len(self.shape) == 2:
            self.shape = (self.n,self.t,self.f)

    def separate(self): #Returns the Sample and input curve it supposedly originated from
        self.reform()
        S = np.asarray(self[0,:,self.p:]).reshape(self.t,self.f-self.p)
        P = Sample(self[:,0,:self.p].reshape(self.n,self.p),columns = self.columns[:self.p])
        return P,S

    def plot(self,name='Sample'): #Plot the first two values of the original Sample
        self.reform()
        P,S = self.separate()
        P.plot(name)
    
    def map(self,f): #Maps a function unto all values
        res = []
        if len(self.shape) == 2: #One input, one output
            return ExData(Sample.map(self,f),p=0,n=self.n)
        for i in range(len(self)): #Curve input, curve output
            parameters = self[i][0,:self.p]
            input_curve = self[i][:,self.p:]
            res.append(f(*parameters, input_curve))
        return ExData(np.array(res))

    def append(self,X): #Returns the appended array
        if self.ndim != X.ndim: X.reform()
        if self.ndim != X.ndim: X.flatten()
        A = Sample.append(self,X)
        return ExData(A,p=self.p,n=self.n+X.n,columns=self.columns)

    def save(self,name): #Save to a .npy file
        data = {
            'type' : "ExData",
            'array' : np.asarray(self),
            'columns' : self.columns,
            'ntfp' : (self.n,self.t,self.f,self.p)
        }
        if str(name)[-4:] != '.pkl':
            name = str(name) + '.pkl'
        with open(name, 'wb') as f:
            pickle.dump(data, f)

    def scale(self,scaler): #Given a sklearn.preprocessing Scaler, scales intself
        n = len(self.shape)
        self.flatten()
        res = scaler.transform(self)
        if n == 3: self.reform()
        return res.reshape(self.shape)

    def scale_back(self,scaler): #Same thing, but with the inverted scaler
        n = len(self.shape)
        self.flatten()
        res = scaler.inverse_transform(self)
        if n == 3: self.reform()
        return res.reshape(self.shape)

    def copy(self): #Creates a deepcopy
        return ExData(np.copy(self),p=self.p,n=self.n,columns=self.columns)

    def sliding_window(self,size,strip=1,padded=False): 
    #Creates sliding windows of the given size
    #strip indicates if consecutive windows need to skip timesteps, and diagonally sample all simulations
    #padded indicates if the first windows should be padded with zeros
        windows = []
        for i in range(0,self.n):
            if padded:
                X = np.vstack((np.repeat(self[i,0].reshape(1,self.f),size-1,axis=0),self[i]))
            else:
                X = self[i]
            for j in range(i%strip,len(X)-size+1,strip):
                windows.append(X[j:j+size,:])

        return ExData(np.array(windows),p=self.p,columns=self.columns).copy()
                

def load_data(name): #Loads either a Sample or ExData
    with open(name,'rb') as f:
        data = pickle.load(f)
    if data['type'] == "Sample":
        return Sample(data['array'],data['columns'])
    else:
        n,t,f,p = data['ntfp']
        return ExData(data['array'],p=p,n=n,columns=data['columns'])
   
