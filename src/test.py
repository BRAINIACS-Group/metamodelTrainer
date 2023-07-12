from explore_param_space import *
from models import *
from file_process import *
import matplotlib.pyplot as mpl
from sklearn.metrics import r2_score
import time

#Path definition
cwd = Path(__file__).resolve().parents[1]
data_path = Path(cwd,"data","LHCU_20")
save_path = Path(cwd,"models")


def visualize_sample(name):
    model = load_model(Path(save_path,name))
    X_T = model.X_T
    P,I = X_T.separate()
    mpl.subplot(2,5,1)
    mpl.scatter(P[:,0],P[:,1],c = [i for i in range(len(P))])
    mpl.subplot(2,5,2)
    mpl.scatter(P[:,0],P[:,2],c = [i for i in range(len(P))])
    mpl.subplot(2,5,3)
    mpl.scatter(P[:,0],P[:,3],c = [i for i in range(len(P))])
    mpl.subplot(2,5,4)
    mpl.scatter(P[:,0],P[:,4],c = [i for i in range(len(P))])
    mpl.subplot(2,5,5)
    mpl.scatter(P[:,1],P[:,2],c = [i for i in range(len(P))])
    mpl.subplot(2,5,6)
    mpl.scatter(P[:,1],P[:,3],c = [i for i in range(len(P))])
    mpl.subplot(2,5,7)
    mpl.scatter(P[:,1],P[:,4],c = [i for i in range(len(P))])
    mpl.subplot(2,5,8)
    mpl.scatter(P[:,2],P[:,3],c = [i for i in range(len(P))])
    mpl.subplot(2,5,9)
    mpl.scatter(P[:,2],P[:,4],c = [i for i in range(len(P))])
    mpl.subplot(2,5,10)
    mpl.scatter(P[:,3],P[:,4],c = [i for i in range(len(P))])
    mpl.show()

def evaluate(model1,model2,X,Y):
    Y_P1 = model1.predict(X, return_var=False)
    Y_P2 = model2.predict(X, return_var=False)
    r21 = np.array([r2_score(Y_P1[i],Y[i]) for i in range(len(X))])
    r22 = np.array([r2_score(Y_P2[i],Y[i]) for i in range(len(X))])
    r2 = r21/r22
    print(np.mean(r2))
    '''
    v = np.mean(V,axis=(1,2))
    v /= np.max(v)
    v = 1 - v
    
    mpl.plot(v,r2)
    mpl.xlabel("1 - Variance, normalized")
    mpl.ylabel("R^2")
    mpl.show()
    '''
    fig, axs = mpl.subplots(2, 3, figsize=(10, 6))

    axs[0, 0].scatter(X[:,0,0], r2)
    axs[0, 0].plot(X[:,0,0], np.ones(X[:,0,0].shape))
    axs[0, 0].set_title('alpha_1')

    axs[0, 1].scatter(X[:,0,1], r2)
    axs[0, 1].plot(X[:,0,1], np.ones(X[:,0,1].shape))
    axs[0, 1].set_title('alpha_2')

    axs[0, 2].scatter(X[:,0,2], r2)
    axs[0, 2].plot(X[:,0,2], np.ones(X[:,0,2].shape))
    axs[0, 2].set_title('mu_1')

    axs[1, 0].scatter(X[:,0,3], r2)
    axs[1, 0].plot(X[:,0,3], np.ones(X[:,0,3].shape))
    axs[1, 0].set_title('mu_2')

    axs[1, 1].scatter(X[:,0,4], r2)
    axs[1, 1].plot(X[:,0,4], np.ones(X[:,0,4].shape))
    axs[1, 1].set_title('eta')

    fig.delaxes(axs[1, 2])

    fig.tight_layout()

    mpl.show()

def finalize(name):
    base = load_model(Path(save_path,name))
    X_T, Y_T = base.X_T, base.Y_T
    HP = base.sum['HP']
    HP['dropout_rate'] = 0
    model = RecModel(X_T,Y_T,HP)
    model.train(1000,1)
    model.save(Path(save_path,name+"_final"))

#finalize("model_AL_base")
#finalize("model_RL_base")
#finalize("model_AL_improved_018")
#finalize("model_RL_improved_018")
#finalize("model_AL_improved_043")
#finalize("model_RL_improved_043")


def label(X): #A really inelegant way to get the material parameters in the correct order for the model
    path = data_path
    X_res, Y_res = load_FE(path)
    P,S = X_res.separate()
    inputs = X_res.columns[X_res.p:]
    comp = np.sum(P - X[0],axis = 1)
    k = 0
    for j in range(len(comp)):
        if abs(comp[j]) < abs(comp[k]):
            k = j
    X_cor = Sample([P[k]],columns=X.columns).spread(S,input_columns=inputs)
    Y_cor = Y_res[k]

    for i in range(1,len(X)):
        inputs = X_res.columns[X_res.p:]
        comp = np.sum(P - X[i],axis = 1)
        k = 0
        for j in range(len(comp)):
            if abs(comp[j]) < abs(comp[k]):
                k = j
        X_cor = X_cor.append(Sample([P[k]],columns=X.columns).spread(S,input_columns=inputs))
        Y_cor = Y_cor.append(Y_res[k])
    X_cor.reform()
    Y_cor.reform()
    return X_cor, Y_cor

model = load_model(Path(save_path,"ogden_final"))
model.drop()
model.save(Path(save_path,"model_final"))
input('Done !')

X,Y = load_FE(data_path)
P,I = X.separate()
P_up = Sample(np.hstack((P[:,1].reshape(len(P),1),P[:,3].reshape(len(P),1),P[:,0].reshape(len(P),1),P[:,2].reshape(len(P),1),P[:,4].reshape(len(P),1))),columns=['alpha_inf','mu_inf','alpha_1','mu_1','eta_1'])
X_new,Y_new = label(P_up)
print(np.max(Y_new - Y))

perm = np.random.permutation(len(P_up))
P2 = Sample(P_up[perm],columns = P_up.columns)
X_new2,Y_new2 = label(P2)
Y2 = Y[perm]
print(np.max(Y_new2 - Y2))
