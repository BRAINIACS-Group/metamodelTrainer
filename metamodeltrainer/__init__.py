#set up path to local module efiPyLab
#this is meant as a make shift fix and can be removed
#once the pyVlab module is made available thorugh pythons package system
from pathlib import Path
import sys
parent_dir = Path(__file__).resolve().parents[2]
pyLabPath = parent_dir / "efiPyVlab/src"
pyPostPath = parent_dir / "efiPostProc/src"
pyOptPath = parent_dir / "efiOpt/src"
sys.path += [str(pyLabPath),
             str(pyPostPath),
             str(pyOptPath)]

import logging
logger = logging.getLogger(__name__)
# create logger
logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)


import pyVlab

from .explore_param_space import Sample
from .models import load_model
