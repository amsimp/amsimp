#cython: language_level=3
cimport numpy as np
from amsimp.water cimport Water

cdef class Dynamics(Water):
    cdef float forecast_days

    cpdef list forecast_temperature(self)
    cpdef list forecast_pressure(self)
    cpdef fit_method(self, x, a, b, c)
    cpdef list forecast_pressurethickness(self)
    cpdef list forecast_precipitablewater(self)