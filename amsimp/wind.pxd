#cython: language_level=3
cimport numpy as np
from amsimp.moist cimport Moist

cdef class Wind(Moist):
    cpdef np.ndarray zonal_wind(self)
    cpdef np.ndarray meridional_wind(self)
    
    cpdef np.ndarray static_stability(self)
    cpdef tuple q_vector(self)
    cpdef np.ndarray vertical_motion(self)
