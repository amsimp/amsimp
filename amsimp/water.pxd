cimport numpy as np
from amsimp.wind cimport Wind

cdef class Water(Wind):
    cpdef np.ndarray vapor_pressure(self)
    cpdef np.ndarray precipitable_water(self)