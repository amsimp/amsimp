#cython: linetrace=True
#distutils: define_macros=CYTHON_TRACE_NOGIL=1
#cython: language_level=3
"""
AMSIMP Wind Class. For information about this class is described below.

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
import os
import iris
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from amsimp.moist cimport Moist
from amsimp.moist import Moist
cimport numpy as np

# -----------------------------------------------------------------------------------------#

cdef class Wind(Moist):
    """
    AMSIMP Wind Class - This class is concerned with calculating numerical
    values for wind in the troposphere and the stratosphere.

    Below is a list of the methods included within this class, with a short
    description of their intended purpose. Please see the relevant class methods
    for more information.

    wind ~ generates a NumPy array of the components of wind. The unit of
    measurement is metres per second.
    static_stability ~ generates a NumPy array of the the static
    stability within a vertical profile.
    verical_motion ~ generates a NumPy array of the vertical motion
    with the atmosphere, by integrating the isobaric version of the
    mass continunity equation.
    wind_contourf ~ generates a vertical profile of geostrophic wind,
    and outputs this as a contour plot.

    globe ~ generates a geostrophic wind contour plot, adds wind vectors to
    that said plot, and overlays both on a Nearside Projection of the Earth.
    """

    cpdef np.ndarray static_stability(self):
        """
        Generates a NumPy array of the the static stability within
        a vertical profile. 
        
        Static stability is defined as the stability of the
        atmosphere in hydrostatic equilibrium with respect to vertical
        displacements.

        Equation:
            sigma = - frac{R T}{p theta} frac{partial theta}{partial p}
        """
        cdef np.ndarray theta = self.potential_temperature(moist=False)
        cdef np.ndarray temperature = self.temperature()
        cdef np.ndarray pressure = self.pressure_surfaces(dim_3d=True)

        static_stability = - self.R * (
                temperature / (pressure * theta)
            ) * self.gradient_p(
                parameter=theta
        )
        return static_stability

    cpdef tuple wind(self):
        """
        This method imports wind data from a GRIB file, which is
        located in the AMSIMP Initial Conditions Data Repository on GitHub.
        The data stored within this repo is updated every six hours by amsimp-bot.
        Following which, it outputs a NumPy array in the shape of
        (len(pressure_surfaces), len(latitude_lines), len(longitude_lines)).
        
        Wind is the flow of gases on a large scale. Wind consists of
        the bulk movement of air.
        
        I strongly recommend storing the output into a variable in order to
        prevent the need to repeatly download the file. For more information,
        visit https://github.com/amsimp/initial-conditions.

        For geostrophic wind, please see the method, geostrophic_wind.
        """
        if not self.input_data:
            folder = "https://github.com/amsimp/initial-conditions/raw/master/initial_conditions/"
            
            # Define date.
            year = self.date.year
            month = self.date.month
            day = self.date.day
            hour = self.date.hour

            # Adds zero before single digit numbers.
            if day < 10:
                day = "0" + str(day)

            if month < 10:
                month =  "0" + str(month)

            if hour < 10:
                hour = "0" + str(hour)

            # Converts integers to strings.
            day = str(day)
            month = str(month)
            year = str(year)
            hour = str(hour)

            folder += (
                year + "/" + month + "/" + day + "/" + hour + "/"
            )
            # The url to the NumPy pressure surfaces file stored on the AMSIMP
            # Initial Conditions Data repository.
            url = folder + "initial_conditions.nc"

            # Download the NumPy file and store the NumPy array into a variable.
            try:
                wind_cube = iris.load("initial_conditions.nc")
                u = np.asarray(wind_cube[4].data)
                v = np.asarray(wind_cube[5].data)
            except OSError:  
                wind_file = self.download(url)
                wind_cube = iris.load(wind_file)
                u = np.asarray(wind_cube[4].data)
                v = np.asarray(wind_cube[5].data)

            # Ensure that the correct data was downloaded (wind).
            if wind_cube[4].units != (self.units.m / self.units.s):
                raise Exception("Unable to determine the zonal wind"
                + " at this time. Please contact the developer for futher"
                + " assistance.")
            elif wind_cube[4].metadata[0] != 'x_wind':
                raise Exception("Unable to determine the zonal wind"
                + " at this time. Please contact the developer for futher"
                + " assistance.")
            
            if wind_cube[5].units != (self.units.m / self.units.s):
                raise Exception("Unable to determine the meridional wind"
                + " at this time. Please contact the developer for futher"
                + " assistance.")
            elif wind_cube[5].metadata[0] != 'y_wind':
                raise Exception("Unable to determine the meridional wind"
                + " at this time. Please contact the developer for futher"
                + " assistance.")
        else:
            u = self.input_u.value
            v = self.input_v.value
        
        if np.shape(u) == (31, 181, 360):
            # Reshape the data in such a way that it matches the pressure surfaces defined in
            # the software.
            u = np.flip(u, axis=0)
            v = np.flip(v, axis=0)

            # Reshape the data in such a way that it matches the latitude lines defined in
            # the software.
            u = np.transpose(u, (1, 2, 0))
            u = u[1:-1]
            u = np.delete(u, [89], axis=0)
            nh_u, sh_u = np.split(u, 2)
            nh_lat, sh_lat = np.split(self.latitude_lines().value, 2)
            nh_u = nh_u[::self.delta_latitude]
            sh_startindex = int((nh_lat[-1] * -1) - 1)
            sh_u = sh_u[sh_startindex::self.delta_latitude]
            u = np.concatenate((nh_u, sh_u))

            v = np.transpose(v, (1, 2, 0))
            v = v[1:-1]
            v = np.delete(v, [89], axis=0)
            nh_v, sh_v = np.split(v, 2)
            nh_lat, sh_lat = np.split(self.latitude_lines().value, 2)
            nh_v = nh_v[::self.delta_latitude]
            sh_startindex = int((nh_lat[-1] * -1) - 1)
            sh_v = sh_v[sh_startindex::self.delta_latitude]
            v = np.concatenate((nh_v, sh_v))

            # Reshape the data in such a way that it matches the longitude lines defined in
            # the software.
            u = np.transpose(u, (2, 0, 1))
            u = u[:, ::-1, ::self.delta_longitude]
            v = np.transpose(v, (2, 0, 1))
            v = v[:, ::-1, ::self.delta_longitude]
            
            if self.remove_files and not self.input_data:
                os.remove("initial_conditions.nc")

            # Define the unit of measurement for wind.
            u *= self.units.m / self.units.s
            v *= self.units.m / self.units.s
        else:
            u *= self.units.m / self.units.s
            v *= self.units.m / self.units.s

        return u, v

    cpdef np.ndarray vertical_motion(self):
        """
        Generates a NumPy array of the vertical motion within
        the atmosphere, by integrating the isobaric version of
        the mass continunity equation. 
        
        Typical large-scale vertical motions in the atmosphere
        are of the order of 0.01 – 0.1 m s−1.

        Equation:
            frac{\partial u}{\partial x} + frac{\partial v}{\partial y} = - frac{\partial \omega}{\partial p}
        """
        # Define variables.
        cdef np.ndarray latitude = np.radians(self.latitude_lines()).value
        cdef np.ndarray longitude = np.radians(self.longitude_lines()).value
        cdef np.ndarray pressure = self.pressure_surfaces(dim_3d=True)
        cdef np.ndarray u, v
        u, v = self.wind()
        
        # Determine the LHS pf the equation by calculating the derivatives.
        # Change in meridional wind with respect to latitude.
        v_dy = self.gradient_y(parameter=v)

        # Change in zonal wind with respect to longitude.
        u_dx = np.gradient(u, longitude, axis=2)
        u_dx = self.gradient_x(parameter=u_dx)

        LHS = u_dx + v_dy
        LHS *= -1

        # Integrate the continunity equation.
        cdef list vertical_motion_list = []
        cdef int n = 0
        cdef int len_pressure = len(pressure)
        cdef omega, omega_unit
        while n < len_pressure:
            p1 = n
            p2 = n+2
            y = LHS[p1:p2, :, :]
            p = pressure[p1:p2, :, :]
            omega = np.trapz(
                y=y, x=p, axis=0
            )
            omega_unit = omega.unit
            vertical_motion_list.append(omega.value)

            n += 1

        # Convert list to NumPy array.
        vertical_motion = np.asarray(vertical_motion_list)

        # Add units.
        vertical_motion *= omega_unit

        # Ensure the shape is correct.
        if np.shape(vertical_motion) != np.shape(u):
            raise Exception("Unable to determine vertical motion"
                + " at this time. Please contact the developer for futher"
                + " assistance.")

        return vertical_motion

# -----------------------------------------------------------------------------------------#

    def globe(self, central_lat=53.1424, central_long=352.3079, psurface=1000, which_wind=0):
        """
        This particular method adds wind vectors to a contour plot. It also
        overlays both of these elements onto a Nearside Perspective projection
        of the Earth.

        By default, the perspective view is looking directly down at the city
        of Dublin in the country of Ireland. If which_wind is set to 0, it will
        plot the non-geostophic wind data, otherwise, if it is set to 1, it will
        plot the geostrophic wind data.

        Note:
        The NumPy method, seterr, is used to suppress a weird RunTime warning
        error.
        """
        # Ensure central_lat is between -90 and 90.
        if central_lat < -90 or central_lat > 90:
            raise Exception(
                "central_lat must be a real number between -90 and 90. The value of central_lat was: {}".format(
                    central_lat
                )
            )

        # Ensure central_long is between 0 and 359.
        if central_long < 0 or central_long > 359:
            raise Exception(
                "central_long must be a real number between 0 and 359. The value of central_long was: {}".format(
                    central_long
                )
            )
        
        # Ensure psurface is between 1000 and 1 hPa above sea level.
        if psurface < 1 or psurface > 1000:
            raise Exception(
                "psurface must be a real number between 1 and 1,000. The value of psurface was: {}".format(
                    psurface
                )
            )
        
        # Index of the nearest alt in amsimp.Backend.altitude_level()
        indx_psurface = (np.abs(self.pressure_surfaces().value - psurface)).argmin()

        # Ignore NumPy errors.
        np.seterr(all="ignore")

        # Define the axes, and the data.
        latitude = self.latitude_lines().value
        longitude = self.longitude_lines().value

        wind = self.wind()
        u = wind[0][indx_psurface, :, :]
        v = wind[1][indx_psurface, :, :]
        title = "Wind"

        u_norm = u / np.sqrt(u ** 2 + v ** 2)
        v_norm = v / np.sqrt(u ** 2 + v ** 2)

        geostrophic_wind = np.sqrt(u ** 2 + v ** 2)

        ax = plt.axes(
            projection=ccrs.NearsidePerspective(
                central_longitude=central_long, central_latitude=central_lat
            )
        )

        # Add latitudinal and longitudinal grid lines, as well as, 
        # coastlines to the globe.
        ax.set_global()
        ax.coastlines()
        ax.gridlines()

        # Contour plotting.
        minimum = geostrophic_wind.min()
        maximum = geostrophic_wind.max()
        levels = np.linspace(minimum, maximum, 21)
        geostrophic_wind, lon = add_cyclic_point(geostrophic_wind, coord=longitude)
        contourf = plt.contourf(
            lon,
            latitude,
            geostrophic_wind,
            transform=ccrs.PlateCarree(),
            levels=levels,
        )
        
        # Reduce density of wind vectors.
        skip_lat = latitude.shape[0] / 23
        skip_lat = int(np.round(skip_lat))
        skip_lon = longitude.shape[0] / 23
        skip_lon = int(np.round(skip_lat))

        # Add geostrophic wind vectors.
        plt.quiver(
            longitude[::skip_lon],
            latitude[::skip_lat],
            u_norm.value[::skip_lat, ::skip_lon],
            v_norm.value[::skip_lat, ::skip_lon],
            transform=ccrs.PlateCarree(),   
        )

        # Add SALT.
        plt.xlabel("Latitude ($\phi$)")
        plt.ylabel("Longitude ($\lambda$)")
        if not self.input_data:
            plt.title(title + " ("
                + str(self.date.year) + '-' + str(self.date.month) + '-'
                + str(self.date.day) + " " + str(self.date.hour)
                + ":00 h, Pressure Surface = "
                + str(self.pressure_surfaces()[indx_psurface]) +")"
            )
        else:
            plt.title(title + " ("
                + "Pressure Surface = "
                + str(self.pressure_surfaces()[indx_psurface]) +")"
            )

        # Add colorbar.
        colorbar = plt.colorbar(contourf)
        tick_locator = ticker.MaxNLocator(nbins=15)
        colorbar.locator = tick_locator
        colorbar.update_ticks()
        colorbar.set_label("Velocity ($\\frac{m}{s}$)")

        # Footnote
        if which_wind == 1:
            plt.figtext(
                0.99,
                0.01,
                "Note: Geostrophic balance does not hold near the equator.",
                horizontalalignment="right",
            )

        plt.show()
        plt.close()
