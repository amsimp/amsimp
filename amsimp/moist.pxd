#cython: language_level=3
cimport numpy as np
from amsimp.backend cimport Backend
from cpython cimport bool

cdef class Moist(Backend):
    cpdef np.ndarray vapor_pressure(self)
    cpdef np.ndarray virtual_temperature(self)

    cpdef np.ndarray density(self)
    cpdef np.ndarray exner_function(self)
    cpdef np.ndarray potential_temperature(self, moist=?)
    cpdef np.ndarray precipitable_water(self)
