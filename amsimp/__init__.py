"""
AMSIMP - An Open Source Implementation to Simulating Atmospheric
Dynamics in the Troposphere and the Stratosphere, written in
the programming language of Cython.
"""
import sys

# Ensure Python 3 is being utilised.
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

# Moist Module.
from amsimp.moist import Moist

# Dynamics Module.
from amsimp.dynamics import Dynamics

# RNN Module.
from amsimp.dynamics import RNN

def get_version():
    """
    Get AMSIMP's version.
    """
    from pkg_resources import get_distribution

    return get_distribution(__package__).version

# Version of AMSIMP.
try:
    __version__ = get_version()
    del  get_version
except:
    pass

