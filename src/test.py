from explore_param_space import *
from models import *
from file_process import *
import matplotlib.pyplot as mpl
from sklearn.metrics import r2_score

#Path definition
cwd = Path(__file__).resolve().parents[1]
data_path = Path(cwd,"data","LHCU_500")
save_path = Path(cwd,"models")

model = load(Path(save_path,"model_5f10e55d.pkl"))

print(model.sum)

X,Y = load_FE(data_path)

def evaluate(model,X,Y):
    Y_P,V = model.predict(X, return_var=True)
    r2 = [r2_score(Y_P[i],Y[i]) for i in range(len(X))]
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

    mpl.show()'''

evaluate(model,X,Y)
