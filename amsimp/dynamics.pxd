#cython: language_level=3
"""
AMSIMP Recurrent Neural Network and Dynamics Class. For information about
this class is described below.

Copyright (C) 2020 AMSIMP

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# -----------------------------------------------------------------------------------------#

# Importing Dependencies
cimport numpy as np
from amsimp.wind cimport Wind
from cpython cimport bool

# -----------------------------------------------------------------------------------------#

cdef class RNN(Wind):
    pass

cdef class Dynamics(RNN):
    cdef forecast_length, delta_t
    
    cpdef tuple forecast_period(self)
    cpdef simulate(
        self,
        bool save_file=?, 
        perturbations_temperature=?, 
        perturbations_zonalwind=?, 
        perturbations_meridionalwind=?, 
        perturbations_mixingratio=?
    )