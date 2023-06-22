from explore_param_space import *
from models import *
from file_process import *
import matplotlib.pyplot as mpl
from sklearn.metrics import r2_score
import time

#Path definition
cwd = Path(__file__).resolve().parents[1]
data_path = Path(cwd,"data","all_data")
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

def evaluate(model,X,Y,name):
    Y_P = model.predict(X, return_var=False)
    r2 = np.array([r2_score(Y_P[i],Y[i]) for i in range(len(X))])
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

    mpl.suptitle(name)

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

#X,Y = load_FE(data_path)

names = ["model_AL_improved_"+str(i*5+3).zfill(3) for i in range(23)]

for n in names:
    finalize(n)

#model = load_model(Path(save_path,"model_AL_base_final"))
#evaluate(model,X,Y,"model_AL_base_final")
#model = load_model(Path(save_path,"model_RL_base_final"))
#evaluate(model,X,Y,"model_RL_base_final")
#model = load_model(Path(save_path,"model_AL_improved_018_final"))
#evaluate(model,X,Y,"model_AL_improved_018_final")
#model = load_model(Path(save_path,"model_RL_improved_018_final"))
#evaluate(model,X,Y,"model_RL_improved_018_final")
#model = load_model(Path(save_path,"model_AL_improved_043_final"))
#evaluate(model,X,Y,"model_AL_improved_043_final")
#model = load_model(Path(save_path,"model_RL_improved_043_final"))
#evaluate(model,X,Y,"model_RL_improved_043_final")
if False:
#if __name__ == '__main__':
    X_T,Y_T = load_FE(data_path)

    #Define hyper parameters
    HP = HyperParameters(layers=[64,64],
                        loss='mae',
                        dropout_rate=0,
                        interpolation=1000)

    model = MegaModel(X_T,Y_T,10,'RNN',HP)
    model.train(10,1,n_workers=4)
    model.save("MMtest")
    model = load_model("MMtest")
    Y = model.predict(X_T)
    input(Y.shape)

    #model = load_model(Path(save_path,"Multi"))
    model = RecModel(X_T,Y_T,HP)
    model.train(10,1)

    model.save("test")
    model = load_model("test")

    t0 = time.time()
    Y,V = model.predict(X_T,return_var=True,n_workers=4)
    print(time.time()-t0)

    print(V.shape)
    input('Done!')

