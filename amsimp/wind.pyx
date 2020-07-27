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

        For geostrophic wind, please see the method, wind.
        """
        if not self.input_data:
            # Input data.
            # Zonal.
            u = self.input_u
            # Meridional.
            v = self.input_v

            pressure = self.pressure_surfaces().to(self.units.Pa)
            # Grid.
            grid_points = [
                ('pressure',  pressure.value),
                ('latitude',  self.latitude_lines().value),
                ('longitude', self.longitude_lines().value),                
            ]

            # Interpolation
            # Zonal.
            u = u.interpolate(
                grid_points, iris.analysis.Linear()
            )
            # Meridional.
            v = v.interpolate(
                grid_points, iris.analysis.Linear()
            )

            # Get data.
            # Zonal.
            u = u.data
            u = np.asarray(u.tolist())
            # Meridional.
            v = v.data
            v = np.asarray(v.tolist())

            u *= self.units.m / self.units.s
            v *= self.units.m / self.units.s
        else:
            # Zonal.
            u = self.input_u
            # Meridional.
            v = self.input_v

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
        v_dy = self.gradient_latitude(parameter=v)

        # Change in zonal wind with respect to longitude.
        u_dx = self.gradient_longitude(parameter=u)

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
        if self.planet == "Earth":
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

            wind = np.sqrt(u ** 2 + v ** 2)

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
            minimum = wind.min()
            maximum = wind.max()
            levels = np.linspace(minimum, maximum, 21)
            wind, lon = add_cyclic_point(wind, coord=longitude)
            wind, lat = add_cyclic_point(np.transpose(wind), coord=latitude)
            wind = np.transpose(wind)
            contourf = plt.contourf(
                lon,
                lat,
                wind,
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
        else:
            raise NotImplementedError(
                "Visualisations for planetary bodies other than Earth is not currently implemented."
            )
