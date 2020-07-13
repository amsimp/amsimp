#cython: language_level=3
"""
AMSIMP Backend Class. For information about this class, please see
the file: backend.pyx.

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
from cpython cimport bool
cimport numpy as np
from amsimp.download cimport Download

# -----------------------------------------------------------------------------------------#

cdef class Backend(Download):
    cdef int delta_latitude
    cdef int delta_longitude
    cdef bool remove_files
    cdef bool ai
    cdef int data_size, epochs
    cdef bool input_data
    cdef input_date, date, input_geo, input_rh, input_temp, input_u, input_v

    cpdef np.ndarray latitude_lines(self, bool beta=?)
    cpdef np.ndarray longitude_lines(self)
    cpdef np.ndarray pressure_surfaces(self, dim_3d=?)
    cdef np.ndarray gradient_x(self, parameter=?)
    cdef np.ndarray gradient_y(self, parameter=?)
    cdef np.ndarray gradient_p(self, parameter=?)
    cdef np.ndarray make_3dimensional_array(self, parameter=?, axis=?)

    cpdef np.ndarray coriolis_parameter(self, bool beta=?)
    cpdef np.ndarray rossby_parameter(self)
    cpdef np.ndarray beta_plane(self)

    cpdef np.ndarray geopotential_height(self)
    cpdef np.ndarray relative_humidity(self)
    cpdef np.ndarray surface_temperature(self)
    cpdef np.ndarray temperature(self)
    cpdef remove_all_files(self)
    
    cpdef np.ndarray gravitational_acceleration(self)
    cpdef np.ndarray pressure_thickness(self, p1=?, p2=?)
    cpdef np.ndarray troposphere_boundaryline(self)
