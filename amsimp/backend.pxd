#cython: language_level=3
from cpython cimport bool
cimport numpy as np

cdef class Backend:
    cdef float detail_level
    cdef bool future

    cpdef np.ndarray latitude_lines(self)
    cpdef np.ndarray longitude_lines(self)
    cpdef np.ndarray altitude_level(self)

    cpdef np.ndarray coriolis_parameter(self)
    cpdef np.ndarray gravitational_acceleration(self)

    cpdef np.ndarray temperature(self)
    cpdef np.ndarray density(self)
    
    cpdef np.ndarray pressure(self)
    cpdef fit_method(self, x, a, b, c)
    cpdef np.ndarray pressure_thickness(self)
    cpdef np.ndarray potential_temperature(self)
    cpdef np.ndarray exner_function(self)
    cpdef np.ndarray troposphere_boundaryline(self)