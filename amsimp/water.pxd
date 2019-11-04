#cython: language_level=3
cimport numpy as np
from amsimp.wind cimport Wind
from cpython cimport bool

cdef class Water(Wind):
    cpdef np.ndarray vapor_pressure(self)
    cpdef np.ndarray precipitable_water(self, sum_altitude=?)