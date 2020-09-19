#cython: language_level=3
"""
AMSIMP Wind Class. For information about this class, please see
the file: wind.pyx.

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
from cpython cimport bool
from amsimp.moist cimport Moist

# -----------------------------------------------------------------------------------------#

cdef class Wind(Moist):
    cpdef tuple wind(self, bool cube=?)
    cpdef np.ndarray vertical_motion(self)
    cpdef np.ndarray static_stability(self)
