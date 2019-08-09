cimport numpy as np
from amsimp.backend cimport Backend

cdef class Wind(Backend):
    cpdef np.ndarray geostrophic_wind(self)