#cython: language_level=3
cimport numpy as np
from amsimp.wind cimport Wind
from cpython cimport bool

cdef class RNN(Wind):
    pass

cdef class Dynamics(RNN):
    cdef forecast_length
    cdef bool efs
    cdef int models

    cpdef atmospheric_prognostic_method(self, bool save_file=?, p1=?, p2=?)