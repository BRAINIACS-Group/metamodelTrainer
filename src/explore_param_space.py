#Builds the Sample and ExData classes, that are to be handled by the models
#The Sample class is used to sample the parameter space : several initialization functions (sample method) are given
#The ExData is used to handle experimental data from FE files

import matplotlib.pyplot as mpl
import numpy as np
import random
from scipy.stats.qmc import PoissonDisk

ref_input = np.linspace(0.80,1.2,50).reshape(50,1)

#########################
### Sample Definition ###
#########################

class Sample(np.ndarray):
    def __new__(cls, A=[[]]): #Creates a Sample object from an array-type object
        obj = np.asarray(A).view(cls)
        return obj

    def append(self,X): #Appends a Sample to another (returns the results)
        return Sample(np.vstack((self,X)))  #Not sure if working

    def plot(self,name = "Sample"): #Scatter plots the first two axis
        mpl.scatter(self[:,0], self[:,1], label = name)
        ax = mpl.gca()

    def draw(self,name = "Sample"): #Same, but displays the result
        self.plot(name)
        mpl.show()

    def copy(self): #Creates a deepcopy
        return Sample(np.copy(self))

    def map(self,f): #Apply a function to each sampled point
        res = np.apply_along_axis(lambda x:f(*x),1,self)
        if res.ndim == 1: res = np.vstack(res)
        return res

    def delete(self,X): #Deletes all values that are present in X ; returns the result
        mask = ~np.all(np.isin(self,X),axis=1)
        return self[mask] #Not sure if working

    def scale(self,scaler): #Uses a sklearn.preprocessing Scaler to scale itself
        return scaler.transform(self)

    def save(self,name): #Save the array in an npy file
        np.save(name,self)

    #Given some input data, creates a 3D ExData object containing, for each parameter set, a 2D array with:
    #- The material parameters being repeated as many times as there are lines in input_data
    #- The input_data
    #Note : input_data is supposed to be in column shape
    def spread(self,input_data = ref_input): 
        if input_data.ndim == 2:
            n, t = len(self),len(input_data)
            p = self.shape[1]
            
            if p == 0: X = np.tile(input_data,(n,1))
            else: X = np.hstack((np.repeat(self,t,axis=0), np.tile(input_data,(n,1))))        

            f = X.shape[1]
            
            return ExData(X.reshape(n,t,f),p)
        else:
            m, t, k = input_data.shape
            n, p = self.shape
            X = self[0].reshape((1,-1)).spread(input_data[0])
            for i in range(1,n):
                X = X.append(self[i].reshape((1,-1)).spread(input_data[i%k]))
            return X
             
########################
### Sampling Methods ###
########################

def scale_to(X,R0,R1): #Scales a random sampling from R0 ranges to R1 ranges
    for i in len(R1):
        X[:,i] -= R0[i][0]
        X[:,i] *= (R1[i][1]-R1[i][0])/(R0[i][1]-R0[i][0])
        X[:,i] += R1[i][0]       
    return X

def RandSample(R=[(0,1),(0,1)],k=100): #Samples k points in a n-dim space randomly
    Points = []
    for i in range(k):
        Points.append([random.random() for j in range(len(R))])
    return Sample(scale_to(np.array(Points),[(0,1) for _ in range(len(R))],R))


def GridSample(R=[(0,1),(0,1)],p=5): #Samples a random point in each square of a n-dim grid with p subdivisions
    n = len(R)
    Points = []
    for i in range(p**n):
        Points.append([random.random()/p + ((i%(p**(j+1)))//p**j)*1/p for j in range(n)])
    return Sample(scale_to(np.array(Points),[(0,1) for _ in range(len(R))],R))

def LHCuSample(R=[(0,1),(0,1)],k=100): #Creates a Latin Hypercube from k equal divisions of the n-dim cube
    n = len(R)
    grid = np.empty((k,n))
    for i in range(n):
        grid[:,i] = np.random.permutation(k)
        
    Points = []
    for i in range(k):
        Points.append([(grid[i,j]+random.random())/k for j in range(n)])
    return Sample(scale_to(np.array(Points),[(0,1) for _ in range(len(R))],R))


def PDskSample(R = [(0,1),(0,1)],k=100): #Uses the PoissonDisk method to sample k points (total number not guaranteed)
    n = len(R)
    engine = PoissonDisk(d=n, radius=0.61/np.sqrt(k*np.sqrt(3)/4), hypersphere='surface')
    return Sample(scale_to(engine.random(k),[(0,1) for _ in range(len(R))],R))

def distance(A,B,R): #Returns the normalized distance between two points (when scaled to a unit square)
    A = scale_to(np.array([A]),R,[(0,1) for _ in range(len(R))])[0]
    B = scale_to(np.array([B]),R,[(0,1) for _ in range(len(R))])[0]
    return np.sqrt(np.sum((A-B)**2))

def distance_to_sample(A,X,R): #Returns the minimum distance from a point to a set of points
    d = [distance(A,X[i],R) for i in range(len(X))]
    return min(d)

def avg_distance(X,R): #Returns the average distance between all points of a sample
    d = [distance(X[i],X[j],R) for j in range(len(X)) for i in range(len(X)) if i != j]
    return np.mean(d)

def min_distance(X,R):
    d = [distance(X[i],X[j],R) for j in range(len(X)) for i in range(len(X)) if i != j]
    return min(d)

#########################
### ExData definition ###
#########################

#The ExData object is basically a numpy array, enriched with some extra information:
# self.n : the number of experimental simulations it is supposed to contain
# self.t : the number of time-steps per simulation
# self.f : the number of features per time-step (typically, material parameters and simulation input data, or simulation output data)
# self.p : among those features, how many are actually material parameters

class ExData(Sample):
    def __new__(cls,X=np.array([ref_input]),p=None,n=None): #Creates a new one from an array
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
        return obj
        
    def flatten(self): #Goes into 2D shape (used in Forward Neural Networks)
        if len(self.shape) == 3:
            self.shape = (self.n*self.t,self.f)
        
    
    def reform(self): #Goes into 3D shape (used in Recurrent Neural Networks)
        if len(self.shape) == 2:
            self.shape = (self.n,self.t,self.f)

    def separate(self): #Returns the Sample and input curve it supposedly originated from
        self.reform()
        S = np.asarray(self[0,:,self.p:]).reshape(self.t,self.f-self.p)
        P = Sample(self[:,0,:self.p].reshape(self.n,self.p))
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
        A = Sample.append(self,X)
        return ExData(A,self.p,self.n+X.n)

    def save(self,name): #Save to a .npy file
        self.reform()
        np.save(name,self)

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
        return ExData(np.copy(self),self.p,self.n)


    #Creates sliding windows of the given size
    #strip indicates if consecutive windows need to skip timesteps, and diagonally sample all simulations
    #padded indicates if the first windows should be padded with zeros
    def sliding_window(self,size,strip=1,padded=False): 
        windows = []
        for i in range(0,self.n):
            if padded:
                X = np.vstack((np.repeat(self[i,0].reshape(1,self.f),size-1,axis=0),self[i]))
            else:
                X = self[i]
            for j in range(i%strip,len(X)-size+1,strip):
                windows.append(X[j:j+size,:])

        return ExData(np.array(windows),self.p).copy()
                

def load_data(name): #Loads either a Sample or ExData
    X = np.load(name)
    if X.ndim == 3:
        n,t,f = X.shape
        p = f - np.count_nonzero(X[0,0,:]-X[0,1,:])
        return ExData(X,p,n)
    else:
        return Sample(X)
   
