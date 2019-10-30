"""
AMSIMP - An Open Source Implementation to Simulating Atmospheric
Dynamics in the Troposphere and the Stratosphere, written in 
Cython.
"""
import sys

# Ensure tthe
if sys.version_info < (3,):
    raise ImportError(
        """
You are running AMSIMP on Python 2.
        
AMSIMP is not compatible with Python 2, due to the
legacy nature of this version of Python. Please see
the article by the Python Organisation as to why you should
switch to Python 3. You can find this article at:
https://wiki.python.org/moin/Python2orPython3
        
We apologise for any inconvience caused."""
    )

# Backend Module.
from amsimp.backend import Backend

# Wind Module.
from amsimp.wind import Wind

# Precipitable Water Module.
from amsimp.water import Water

# Weather Module.
from amsimp.dynamics import Dynamics

def get_version():
    """
    Get AMSIMP's version.
    """
    from pkg_resources import get_distribution
    
    return get_distribution(__package__).version

__version__ = get_version()
del  get_version