#cython: language_level=3
cimport numpy as np
from amsimp.backend cimport Backend

cdef class Wind(Backend):
    cpdef np.ndarray zonal_wind(self)
    cpdef np.ndarray meridional_wind(self)