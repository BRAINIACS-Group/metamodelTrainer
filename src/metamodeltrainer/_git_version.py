'''
'''

#STL modules
from pathlib import Path
import logging

#3rd party modules
import git
import subprocess

logger = logging.getLogger(__name__)

def get_git_version()->str:
    '''return the current git branch and hash as obtained by 
    git describe --all --long. First the file git_version.txt is looked up,
    created by hatch during the build process of this module
    Args:
    Returns: string containing git describe output
    Raises: FileNotFoundError if git_version.txt is not found
    '''

    try:
        git_str = subprocess.check_output(["git", "describe","--all","--long"])\
            .decode('utf-8').strip()
        return git_str
    except subprocess.CalledProcessError as cpe:
        logger.warning('git command failed (git installed?) and'
            'returned status %s output %s',cpe.returncode,cpe.output)

    git_file = Path(__file__).resolve().parent / "git_version.txt"
    if not git_file.is_file():
        raise FileNotFoundError(f'git command failed and {git_file} not found')
    git_str = git_file.read_text(encoding='utf-8')

    return git_str
