
#STL modules
from pathlib import Path
from typing import List

#3rd party modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#local imports
from run_simulation import load_FE
from explore_param_space import Sample

def get_dataset_from_dir(dirpath:Path,stress:bool=False,
    parameters:List[str]=None):
    ''''''
    X_res = Y_res = None

    for i,dir in enumerate(dirpath.iterdir()):       
        X_res_new, Y_res_new = load_FE(dir,parameters,stress=stress)
        if i == 0:
            X_res = X_res_new
            Y_res = Y_res_new
        else:
            X_res = X_res.append(X_res_new)
            Y_res = Y_res.append(Y_res_new)


    return X_res,Y_res

if __name__ == "__main__":
   
    base_dir = Path(__file__).resolve().parents[1]

    fe_res_dir = base_dir / "data" / "FE_outputs_231220"

    X,Y = get_dataset_from_dir(fe_res_dir,stress=False,
        parameters=['alpha_inf', 'mu_inf', 'alpha_1', 'mu_1', 'eta_1' ],
#        parameters=['alpha_1','alpha_inf','mu_1','mu_inf','eta_1'])
        )

    dataset_dir = base_dir / "data" / "FE_train_force_231220"
    if not dataset_dir.is_dir():
        dataset_dir.mkdir()
    X.save(dataset_dir / 'X')
    Y.save(dataset_dir / 'Y')

    # P,S = X.separate()
    # parameters = pd.DataFrame(P,columns=P.columns)
    # parameters.hist()
    # plt.show()