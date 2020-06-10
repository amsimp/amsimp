#cython: language_level=3
cimport numpy as np
from amsimp.moist cimport Moist

cdef class Wind(Moist):
    cpdef tuple geostrophic_wind(self)
    cpdef tuple wind(self)
    cpdef tuple ageostrophic_wind(self)
    cpdef np.ndarray vertical_motion(self)
    cpdef np.ndarray static_stability(self)
    cpdef np.ndarray geostophic_vorticity(self)
