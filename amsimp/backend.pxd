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

# -----------------------------------------------------------------------------------------#

cdef class Backend:
    cdef forecast_length
    cdef historical_data
    cdef date, psurfaces, lat, lon, input_geo, input_rh
    cdef input_temp, input_u, input_v

    cpdef np.ndarray latitude_lines(self)
    cpdef np.ndarray longitude_lines(self)
    cpdef np.ndarray pressure_surfaces(self, dim_3d=?)

    cpdef np.ndarray gradient_longitude(self, parameter=?)
    cpdef np.ndarray gradient_latitude(self, parameter=?)
    cpdef np.ndarray gradient_pressure(self, parameter=?)
    cpdef np.ndarray make_3dimensional_array(self, parameter=?, axis=?)

    cpdef np.ndarray coriolis_parameter(self)

    cpdef geopotential_height(self, bool cube=?)
    cpdef relative_humidity(self, bool cube=?)
    cpdef temperature(self, bool cube=?)
    cpdef exit(self)
    
    cpdef np.ndarray pressure_thickness(self, p1=?, p2=?)
    cpdef np.ndarray troposphere_boundaryline(self)
