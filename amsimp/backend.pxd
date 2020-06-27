#cython: language_level=3
from cpython cimport bool
cimport numpy as np

cdef class Backend:
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
