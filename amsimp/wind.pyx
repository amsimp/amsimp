#cython: linetrace=True
#distutils: define_macros=CYTHON_TRACE_NOGIL=1
#cython: language_level=3
# cython: embedsignature=True, binding=True
"""
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
along with this program.  If not, see https://www.gnu.org/licenses/.
"""

# ------------------------------------------------------------------------------#

# Importing Dependencies
import os
import iris
from astropy import units
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.widgets import Slider, Button
import numpy as np
from amsimp.moist cimport Moist
from amsimp.moist import Moist
cimport numpy as np
from cpython cimport bool

# ------------------------------------------------------------------------------#

cdef class Wind(Moist):
    """
    This class is concerned with calculating numerical values for wind.
    """

    cpdef np.ndarray static_stability(self):
        r"""Generates an array of the the static stability.

        .. math:: \sigma = - \frac{R T}{p \theta} \frac{\partial \theta}{\partial p}

        Returns
        -------
        `astropy.units.quantity.Quantity`
            Static stability

        Notes
        -----
        Static stability is defined as the stability of the
        atmosphere in hydrostatic equilibrium with respect to vertical
        displacements.
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

    cpdef tuple wind(self, bool cube=False):
        r"""Generates an arrray of wind.
        
        Returns
        -------
        `tuple`
            Wind (zonal and meridional components)

        Notes
        -----
        If the user did not define initial conditions on initialisation
        of the class, this data is retrieved from the AMSIMP Initial
        Conditions Data Repository on GitHub. Wind is the flow of gases
        on a large scale. Wind consists of the bulk movement of air.

        See Also
        --------
        vertical_motion
        """
        if not self.input_data:
            # Input data.
            # Zonal.
            u = self.input_u
            # Meridional.
            v = self.input_v

            pressure = self.pressure_surfaces().to(units.Pa)
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

            if not cube:
                # Get data.
                # Zonal.
                u = u.data
                u = np.asarray(u.tolist())
                # Meridional.
                v = v.data
                v = np.asarray(v.tolist())

                u *= units.m / units.s
                v *= units.m / units.s
        else:
            # Zonal.
            u = self.input_u
            # Meridional.
            v = self.input_v

        return u, v

    cpdef np.ndarray vertical_motion(self):
        r"""Generates an array of the vertical motion.
        
        .. math:: \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} \
            = - \frac{\partial \omega}{\partial p}
        
        Returns
        -------
        `astropy.units.quantity.Quantity`
            Vertical motion

        Notes
        -----
        Typical large-scale vertical motions in the atmosphere
        are of the order of 0.01 – 0.1 :math:`\frac{m}{s}`.

        See Also
        --------
        wind
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

# ------------------------------------------------------------------------------#

    def globe(
            self,
            which="air_temperature",
            central_latitude = 53.1424, 
            central_longitude = 352.3079, 
            psurface=1000
        ):
        r"""Generates a wind vector plot over a contour plot.

        Parameters
        ----------
        which: `str`
            Desired atmospheric parameter
        central_latitude: `float`
            The line of latitude at which the view is looking directly down at
        central_longitude: `float`
            The line of longitude at which the view is looking directly down at
        psurface: `int`
            Pressure at which the contour plot is generated
        
        Notes
        -----
        It overlays both of these elements onto a Nearside Perspective projection
        of the Earth. By default, the perspective view is looking directly 
        down at the city of Dublin in the country of Ireland. If you would 
        like a particular atmospheric parameter to be added to this method, 
        either create an issue on GitHub or send an email to support@amsimp.com.

        See Also
        --------
        wind
        """
        if self.planet == "Earth":
            # Ensure psurface is between 1000 and 1 hPa above sea level.
            if psurface < 1 or psurface > 1000:
                raise Exception(
                    "psurface must be a real number between 1 and 1,000."
                    + " The value of psurface was: {}".format(
                        psurface
                    )
                )
            
            # Index of the nearest alt in amsimp.Backend.altitude_level()
            indx_psurface = (
                np.abs(self.pressure_surfaces().value - psurface)
            ).argmin()

            # Define latitude and longitude variables.
            latitude = self.latitude_lines().value
            longitude = self.longitude_lines().value

            # Remove NumPy runtime warning.
            np.seterr(all='ignore')

            # Vector plot.
            # Define zonal and meridional wind.
            wind = self.wind()
            u = wind[0][indx_psurface, :, :]
            v = wind[1][indx_psurface, :, :]

            # Reduce density of wind vectors.
            skip_lat = latitude.shape[0] / 23
            skip_lat = int(np.round(skip_lat))
            skip_lon = longitude.shape[0] / 23
            skip_lon = int(np.round(skip_lat))

            # Contour plotting.
            # Define atmospheric parameter of interest.
            if which == "temperature" or which == "air_temperature":
                data = self.temperature()[indx_psurface, :, :]
                data_type = "Air Temperature"
                unit = " (K)"
            elif which == "geopotential_height" or which == "height":
                data = self.geopotential_height()[indx_psurface, :, :]
                data_type = "Geopotential Height"
                unit = " (m)"
            elif which == "density" or which == "atmospheric_density":
                data = self.density()[indx_psurface, :, :]
                data_type = "Atmospheric Density"
                unit = " ($\\frac{kg}{m^3}$)"
            elif which == "humidity" or which == "relative_humidity":
                data = self.relative_humidity()[indx_psurface, :, :]
                data_type = "Relative Humidity"
                unit = " (%)"
            elif which == "virtual_temperature":
                data = self.virtual_temperature()[indx_psurface, :, :]
                data_type = "Virtual Temperature"
                unit = " (K)"
            elif which == "vapor_pressure":
                data = self.vapor_pressure()[indx_psurface, :, :]
                data_type = "Vapor Pressure"
                unit = " (hPa)"
            elif which == "potential_temperature":
                data = self.potential_temperature()[indx_psurface, :, :]
                data_type = "Potential Temperature"
                unit = " (K)"
            elif which == "precipitable_water" or which == "precipitable_water_vapor":
                data = self.precipitable_water()
                data_type = "Precipitable Water Vapor"
                unit = " (mm)"
            elif which == "zonal_wind":
                data = self.wind()[0][indx_psurface, :, :]
                data_type = "Zonal Wind"
                unit = " ($\\frac{m}{s}$)"
            elif which == "meridional_wind":
                data = self.wind()[1][indx_psurface, :, :]
                data_type = "Meridional Wind"
                unit = " ($\\frac{m}{s}$)"
            elif which == "wind":
                data = self.wind()
                data = np.sqrt(data[0]**2 + data[1]**2)[indx_psurface, :, :]
                data_type = "Wind"
                unit = " ($\\frac{m}{s}$)"
            elif which == "mixing_ratio":
                data = self.mixing_ratio()[indx_psurface, :, :]
                data_type = "Mixing Ratio"
                unit = " (Dimensionless)"
            else:
                raise Exception(
                    "Invalid keyword. which must be a string of an atmospheric "
                    + "parameter included with AMSIMP."
                )

            # Define graphing parameters.
            minimum_val = data.min().value
            maximum_val = data.max().value
            levels = np.linspace(minimum_val, maximum_val, 21)
            data, lon = add_cyclic_point(data, coord=longitude)
            data, lat = add_cyclic_point(np.transpose(data), coord=latitude)
            data = np.transpose(data)

            def update(val):
                try:
                    central_lon = slon.val
                    central_lat = slat.val
                except:
                    central_lat = central_latitude
                    central_lon = central_longitude

                # Define the axes.
                crs = ccrs.NearsidePerspective(
                    central_longitude=central_lon, 
                    central_latitude=central_lat
                )
                ax = plt.axes(
                    projection=crs
                )
                plt.subplots_adjust(left=0.25, bottom=0.25)
                
                # Add latitudinal and longitudinal grid lines, as well as, 
                # coastlines to the globe.
                ax.set_global()
                ax.coastlines()
                ax.gridlines()

                # Contour plot.
                contourf = plt.contourf(
                    lon,
                    lat,
                    data,
                    transform=ccrs.PlateCarree(),
                    levels=levels,
                )

                # Add wind vectors.
                plt.quiver(
                    longitude[::skip_lon],
                    latitude[::skip_lat],
                    u.value[::skip_lat, ::skip_lon],
                    v.value[::skip_lat, ::skip_lon],
                    transform=ccrs.PlateCarree(),   
                )

                # Colorbar creation.
                colorbar = plt.colorbar(contourf)
                tick_locator = ticker.MaxNLocator(nbins=15)
                colorbar.locator = tick_locator
                colorbar.update_ticks()
                colorbar.set_label(
                    data_type + unit
                )

                # Add SALT.
                if not self.input_data:
                    title = (
                        data_type + " ("
                        + str(self.date.year) + '-' + str(self.date.month) + '-'
                        + str(self.date.day) + " " + str(self.date.hour)
                        + ":00 h)"
                    )
                else:
                    title = data_type
            
                if which != "precipitable_water" and which != "precipitable_water_vapor":
                    title = (
                        title + " (" + "Pressure Surface = " + str(
                            self.pressure_surfaces()[indx_psurface]
                        ) + ")"
                    )

                plt.title(title)
            
            # First plot.
            update(None)

            # Sliders.
            axcolor = 'lightgoldenrodyellow'
            axlon = plt.axes(
                [0.25, 0.15, 0.65, 0.03],
                facecolor=axcolor, 
                projection=ccrs.PlateCarree()
            )
            axlat = plt.axes(
                [0.25, 0.2, 0.65, 0.03],
                facecolor=axcolor, 
                projection=ccrs.PlateCarree()
            )

            slon = Slider(
                axlon, 'Longitude', 0, 359, valinit=central_longitude
            )
            slat = Slider(
                axlat, 'Latitude', -89, 89, valinit=central_latitude
            )

            slon.on_changed(update)
            slat.on_changed(update)

            resetax = plt.axes(
                [0.8, 0.025, 0.1, 0.04], projection=ccrs.PlateCarree()
            )
            button = Button(
                resetax, 
                'Reset', 
                color=axcolor, 
                hovercolor='0.975'
            )

            def reset(event):
                slon.reset()
                slat.reset()
            button.on_clicked(reset)

            # Show plot.
            plt.show()
        else:
            raise NotImplementedError(
                "Visualisations for planetary bodies other than Earth"
                + " is not currently implemented."
            )
