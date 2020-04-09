#cython: language_level=3
cimport numpy as np
from amsimp.wind cimport Wind
from cpython cimport bool

cdef class Dynamics(Wind):
    cdef forecast_length
    cdef bool efs, ai
    cdef int models

    cpdef atmospheric_prognostic_method(self, bool save_file=?, p1=?, p2=?)