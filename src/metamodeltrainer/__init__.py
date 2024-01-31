#set up path to local module efiPyLab
#this is meant as a make shift fix and can be removed
#once the pyVlab module is made available thorugh pythons package system
from pathlib import Path

from ._git_version import get_git_version

__git_version__ = get_git_version()

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


from .explore_param_space import Sample
from .models import load_model
