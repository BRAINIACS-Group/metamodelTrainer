#set up path to local module efiPyLab
#this is meant as a make shift fix and can be removed
#once the pyVlab module is made available thorugh pythons package system
from pathlib import Path
import sys
parent_dir = Path(__file__).parents[2]
pyLabPath = parent_dir / "efiPyVlab/src"
pyPostPath = parent_dir / "efiPostProc/src"
pyOptPath = parent_dir / "efiOpt/src"
sys.path += [str(pyLabPath),
             str(pyPostPath),
             str(pyOptPath)]

import pyVlab

from .explore_param_space import Sample
from .models import load_model
