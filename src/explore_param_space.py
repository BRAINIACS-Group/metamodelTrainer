import matplotlib.pyplot as mpl
import numpy as np
import random
from scipy.optimize import minimize
from scipy.stats.qmc import PoissonDisk
import pandas as pd
import os

#make n,t,f,p into properties
#maybe even flat and volume versions into properties ?

ref_input = np.linspace(0.80,1.2,50).reshape(50,1)

class Sample(np.ndarray):
    def __new__(cls, A=[[]]):
        obj = np.asarray(A).view(cls)
        return obj

    def append(self,X):
        return Sample(np.vstack((self,X)))  #Not sure if working

    def plot(self,name = "Sample"):
        mpl.scatter(self[:,0], self[:,1], label = name)
        ax = mpl.gca()

    def draw(self,name = "Sample"):
        self.plot(name)
        mpl.show()

    def copy(self):
        return Sample(np.copy(self))

    def map(self,f):
        res = np.apply_along_axis(lambda x:f(*x),1,self)
        if res.ndim == 1: res = np.vstack(res)
        return res

    def delete(self,X):
        mask = ~np.all(np.isin(self,X),axis=1)
        self = self[mask] #Not sure if working

    def scale(self,scaler):
        return scaler.transform(self)

    def save(self,name):
        np.save(name,self)

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
             
        

def scale_to(X,R): #Scales a random sampling in [0,1] square to desired ranges
    for i,r in enumerate(R):
        v_min = r[0]
        v_max = r[1]
        X[:,i] *= (v_max-v_min)
        X[:,i] += v_min
    return X

def RandSample(R=[(0,1),(0,1)],k=100): #Samples k points in a n-dim space randomly
    Points = []
    for i in range(k):
        Points.append([random.random() for j in range(len(R))])
    return Sample(scale_to(np.array(Points),R))


def GridSample(R=[(0,1),(0,1)],p=5): #Samples a random point in each square of a n-dim grid with p subdivisions
    n = len(R)
    Points = []
    for i in range(p**n):
        Points.append([random.random()/p + ((i%(p**(j+1)))//p**j)*1/p for j in range(n)])
    return Sample(scale_to(np.array(Points),R))

def LHCuSample(R=[(0,1),(0,1)],k=100): #Creates a Latin Hypercube from k equal divisions of the n-dim cube
    n = len(R)
    grid = np.empty((k,n))
    for i in range(n):
        grid[:,i] = np.random.permutation(k)
        
    Points = []
    for i in range(k):
        Points.append([(grid[i,j]+random.random())/k for j in range(n)])
    return Sample(scale_to(np.array(Points),R))

def PDskSample(R = [(0,1),(0,1)],k=100):
    n = len(R)
    engine = PoissonDisk(d=n, radius=0.61/np.sqrt(k*np.sqrt(3)/4), hypersphere='surface')
    return Sample(scale_to(engine.random(k),R))


class ExData(Sample):
    def __new__(cls,X=np.array([ref_input]),p=None,n=None):
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
            if len(X.shape) == 3: p = obj.f- np.count_nonzero(X[0,0,:] - X[0,1,:])
            else : p = obj.f - np.count_nonzero(X[0,:] - X[1,:])
        obj.p = p
        return obj
        
    def flatten(self):
        if len(self.shape) == 3:
            self.shape = (self.n*self.t,self.f)
        
    
    def reform(self):
        if len(self.shape) == 2:
            self.shape = (self.n,self.t,self.f)

    def separate(self):
        self.reform()
        S = np.asarray(self[0,:,self.p:]).reshape(self.t,self.f-self.p)
        P = Sample(self[:,0,:self.p].reshape(self.n,self.p))
        return P,S

    def plot(self,name='Sample'):
        self.reform()
        P,S = self.separate()
        P.plot(name)
    
    def map(self,f):
        res = []
        if len(self.shape) == 2: #One input, one output
            return ExData(Sample.map(self,f),p=0,n=self.n)
        for i in range(len(self)): #Curve input, curve output
            parameters = self[i][0,:self.p]
            input_curve = self[i][:,self.p:]
            res.append(f(*parameters, input_curve))
        return ExData(np.array(res))

    def append(self,X):
        A = Sample.append(self,X)
        return ExData(A,self.p,self.n+X.n)

    def save(self,name):
        self.reform()
        np.save(name,self)

    def scale(self,scaler):
        n = len(self.shape)
        self.flatten()
        res = scaler.transform(self)
        if n == 3: self.reform()
        return res.reshape(self.shape)

    def scale_back(self,scaler):
        n = len(self.shape)
        self.flatten()
        res = scaler.inverse_transform(self)
        if n == 3: self.reform()
        return res.reshape(self.shape)

    def copy(self):
        return ExData(np.copy(self),self.p,self.n)

    '''def sliding_window(self,size,strip=1,padded=False,skip_first=True):
        windows = []
        for i in range(0,self.n):
            if padded:
                for j in range(skip_first+i%strip,size,strip):
                    windows.append(np.vstack((np.repeat(self[i,0].reshape(1,self.f),size-j,axis=0),self[i][:j,:])))
            for j in range(i%strip,self.t - size + skip_first,strip):
                windows.append(self[i,j:j+size,:])
        return ExData(np.array(windows),self.p).copy()'''

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
                
def load_data(name):
    X = np.load(name)
    if X.ndim == 3:
        n,t,f = X.shape
        p = f - np.count_nonzero(X[0,0,:]-X[0,1,:])
        return ExData(X,p,n)
    else:
        return Sample(X)
   
