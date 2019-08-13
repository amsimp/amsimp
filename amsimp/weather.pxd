#cython: language_level=3
cimport numpy as np
from amsimp.water cimport Water

cdef class Weather(Water):
    pass