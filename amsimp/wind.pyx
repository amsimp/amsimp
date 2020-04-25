#cython: linetrace=True
#distutils: define_macros=CYTHON_TRACE_NOGIL=1
#cython: language_level=3
"""
AMSIMP Wind Class. For information about this class is described below.
"""

# -----------------------------------------------------------------------------------------#

# Importing Dependencies
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
    values for wind, specifically geostrophic wind, in the troposphere and the
    stratosphere. It also contain two methods for the visualisation of these
    numerical values.

    Below is a list of the methods included within this class, with a short
    description of their intended purpose. Please see the relevant class methods
    for more information.

    zonal_wind ~ outputs the zonal component of geostrophic wind.
    meridional_wind ~ outputs the meridional component of geostrophic wind.
    wind_contourf ~ generates a geostrophic wind contour plot.
    globe ~ generates a geostrophic wind contour plot, adds wind vectors to
    that said plot, and overlays both on a Nearside Projection of the Earth.
    """

    cpdef np.ndarray zonal_wind(self):
        """
        This method outputs the zonal component of geostrophic wind, in
        the shape of (len(pressure_surfaces), len(latitude_lines),
        len(longitude_lines)). 
        
        Geostrophic wind is a theoretical wind that is a result of a perfect
        balance between the Coriolis force and the pressure gradient force.
        This balance is known as geostrophic balance. 

        Equation:
            u_g = -frac{g}{f} \* frac{1}{a} \* frac{\partial \Phi}{\partial y}
        
        Note: Geostrophic balance does not hold near the equator.
        """
        # Gradient of geopotential height over latitudinal distance.
        cdef np.ndarray latitude = np.radians(self.latitude_lines())
        cdef height_gradient = np.gradient(
            self.geopotential_height(),
            self.a * latitude.value,
            axis=1,
        )

        # Defining a 3D coriolis parameter NumPy array.
        cdef np.ndarray coriolis_parameter = self.coriolis_parameter().value
        coriolis_parameter /= self.units.s
        coriolis_parameter = self.make_3dimensional_array(
            parameter=coriolis_parameter, axis=1
        )

        # Geostrophic wind calculation (zonal component).
        zonal_wind = (
            -(self.gravitational_acceleration() / coriolis_parameter)
            * height_gradient
        )

        # Convert zonal wind to metres per second.
        zonal_wind = zonal_wind.si

        return zonal_wind

    cpdef np.ndarray meridional_wind(self):
        """
        This method outputs the meridional component of geostrophic wind, in
        the shape of (len(pressure_surfaces), len(latitude_lines),
        len(longitude_lines)).
        
        Geostrophic wind is a theoretical wind that is a result of a perfect
        balance between the Coriolis force and the pressure gradient force.
        This balance is known as geostrophic balance.

        Equation:
            v_g = frac{g}{f} \* frac{\partial \Phi}{\partial y} 
        
        Note: Geostrophic balance does not hold near the equator.
        """
        # Gradient of geopotential height over longitudinal distance.
        cdef np.ndarray longitude = np.radians(self.longitude_lines())
        cdef np.ndarray height_gradient = np.gradient(
            self.geopotential_height().value,
            longitude.value,
            axis=2,
        )
        height_gradient *= self.units.m
        height_gradient = self.gradient_x(parameter=height_gradient)

        # Defining a 3D Coriolis parameter NumPy array.
        cdef np.ndarray coriolis_parameter = self.coriolis_parameter().value
        cdef list f_list = []
        for f in coriolis_parameter:
            f = f + np.zeros(
                (len(self.pressure_surfaces()), len(self.longitude_lines()))
            )
            f = f.tolist()
            f_list.append(f)
        coriolis_parameter = np.asarray(f_list)
        coriolis_parameter = np.transpose(coriolis_parameter, (1, 0, 2))
        coriolis_parameter /= self.units.s

        meridional_wind = (
            (self.gravitational_acceleration() / coriolis_parameter)
            * height_gradient
        )
        meridional_wind = meridional_wind.si

        return meridional_wind

    cpdef np.ndarray static_stability(self):
        """
        Explain here.
        """
        cdef np.ndarray theta = self.potential_temperature(moist=False)
        cdef np.ndarray temperature = self.temperature()
        cdef np.ndarray pressure = self.pressure_surfaces(dim_3d=True)

        static_stability = - self.R * (
                temperature / (pressure * theta)
            ) * np.gradient(
                theta, self.pressure_surfaces(), axis=0
        )
        return static_stability

    cpdef tuple q_vector(self):
        """
        Explain here.
        """
        cdef np.ndarray latitude = np.radians(self.latitude_lines()).value
        cdef np.ndarray longitude = np.radians(self.longitude_lines()).value
        cdef np.ndarray zonal_wind = self.zonal_wind()
        cdef np.ndarray meridional_wind = self.meridional_wind()
        cdef np.ndarray temperature = self.temperature()
        cdef np.ndarray pressure = self.pressure_surfaces(dim_3d=True)
        cdef np.ndarray static_stability = self.static_stability()

        # Geostrophic Wind.
        cdef np.ndarray du_dx = np.gradient(
            zonal_wind.value, longitude, axis=2
        ) * zonal_wind.unit
        cdef np.ndarray du_dy = np.gradient(
            zonal_wind.value, latitude, axis=1
        ) * zonal_wind.unit
        cdef np.ndarray dv_dx = np.gradient(
            meridional_wind.value, longitude, axis=2
        ) * meridional_wind.unit
        cdef np.ndarray dv_dy = np.gradient(
            meridional_wind.value, latitude, axis=1
        ) * meridional_wind.unit

        # Temperature.
        cdef np.ndarray dT_dx = np.gradient(
            temperature.value, longitude, axis=2
        ) * temperature.unit
        cdef np.ndarray dT_dy = np.gradient(
            temperature.value, latitude, axis=1
        ) * temperature.unit

        # Delta x.
        du_dx = self.gradient_x(parameter=du_dx)
        dv_dx = self.gradient_x(parameter=dv_dx)
        dT_dx = self.gradient_x(parameter=dT_dx)
        
        # Delta y
        du_dy = (1 / self.a) * du_dy
        dv_dy = (1 / self.a) * dv_dy
        dT_dy = (1 / self.a) * dT_dy

        # q_vector
        q1 = (-self.R / (pressure * static_stability)) * (du_dx * dT_dx + dv_dx * dT_dy)
        q2 = (-self.R / (pressure * static_stability)) * (du_dy * dT_dx + dv_dy * dT_dy)

        q_vector = (q1, q2)
        return q_vector

    cpdef np.ndarray vertical_motion(self):
        """
        Explain here.
        """
        cdef np.ndarray latitude = np.radians(self.latitude_lines()).value
        cdef np.ndarray longitude = np.radians(self.longitude_lines()).value
        cdef np.ndarray pressure = self.pressure_surfaces(dim_3d=True)
        cdef np.ndarray q1, q2
        q1, q2 = self.q_vector()
        
        q2_dy = np.gradient(q2, self.a * latitude, axis=1)

        q1_dx = np.gradient(q1, longitude, axis=2)
        q1_dx = self.gradient_x(parameter=q1_dx)

        vertical_motion = -2 * (q1_dx + q2_dy)
        vertical_motion *= -1
        vertical_motion = vertical_motion.value * (self.units.hPa / self.units.s)
        return vertical_motion

# -----------------------------------------------------------------------------------------#

    def wind_contourf(self, which_wind=0, central_long=-7.6921):
        """
        For zonal (geostrophic) wind, set which_wind to 0.
        For meridional (geostrophic) wind, set which_wind to 1.
        
        Generates a geostrophic wind contour plot, with the axes being
        latitude, and longitude.
        
        For the raw data, please use the
        amsimp.Wind.geostrophic_wind() method.
        """
        # Ensure central_long is between -180 and 180.
        if central_long < -180 or central_long > 180:
            raise Exception(
                "central_long must be a real number between -180 and 180. The value of central_long was: {}".format(
                    central_long
                )
            )
        
        # Index of the nearest central_long in amsimp.Backend.longtitude_lines()
        indx_long = (np.abs(self.longitude_lines().value - central_long)).argmin()
        
        # Set geostrophic wind to either the zonal, or meridional component.
        if which_wind == 0:
            geostrophic_wind = self.zonal_wind()[:, :, indx_long]
            wind_type = "Zonal"
        elif which_wind == 1:
            geostrophic_wind = self.meridional_wind()[:, :, indx_long]
            wind_type = "Meridional"
        else:
            raise Exception(
                "which_wind must be equal to either zero or one. The value of which_wind was: {}".format(
                    which_wind
                )
            )

        # Defines the axes, and the data.
        latitude, pressure_surfaces = np.meshgrid(self.latitude_lines(), self.pressure_surfaces())

        # Specifies the contour levels
        minimum = geostrophic_wind.min()
        maximum = geostrophic_wind.max()
        levels = np.linspace(minimum, maximum, 21)

        # Contour plotting.
        plt.contourf(latitude, pressure_surfaces, geostrophic_wind, levels=levels)

        plt.set_cmap("jet")

        # Add SALT.
        plt.xlabel("Latitude ($\phi$)")
        plt.ylabel("Pressure (hPa)")
        plt.yscale('log')
        plt.gca().invert_yaxis()
        if not self.input_data:
            plt.title(wind_type + " Wind Contour Plot ("
                + str(self.date.year) + '-' + str(self.date.month) + '-'
                + str(self.date.day) + " " + str(self.date.hour)
                + ":00 h, Longitude = "
                + str(np.round(self.longitude_lines()[indx_long], 2)) + ")"
            )
        else:
            plt.title(wind_type + " Wind Contour Plot ("
                + "Longitude = "
                + str(np.round(self.longitude_lines()[indx_long], 2)) + ")"
            )

        # Footnote
        plt.figtext(
            0.99,
            0.01,
            "Note: Geostrophic balance does not hold near the equator.",
            horizontalalignment="right",
        )
        plt.subplots_adjust(bottom=0.135)

        # Colorbar creation.
        colorbar = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=15)
        colorbar.locator = tick_locator
        colorbar.update_ticks()
        colorbar.set_label("Velocity ($\\frac{m}{s}$)")

        # Average boundary line between the troposphere and the stratosphere.
        troposphere_boundaryline = self.troposphere_boundaryline()
        avg_tropstratline = np.mean(troposphere_boundaryline) + np.zeros(
            len(troposphere_boundaryline)
        )

        # Plot average boundary line on the contour plot.
        plt.plot(
            latitude[1],
            avg_tropstratline,
            color="black",
            linestyle="dashed",
            label="Troposphere - Stratosphere Boundary Line",
        )
        plt.legend(loc=0)

        plt.show()
        plt.close()

    def globe(self, central_lat=53.1424, central_long=-7.6921, psurface=1000):
        """
        Similiar to amsimp.Wind.wind_contourf(), however, this particular method
        adds wind vectors to the contour plot. It also overlays both of these
        elements onto a Nearside Perspective projection of the Earth.

        By default, the perspective view is looking directly down at the city
        of Dublin in the country of Ireland.

        Note:
        The NumPy method, seterr, is used to suppress a weird RunTime warning
        error that occurs on certain detail_level values in certain months.
        """
        # Ensure central_lat is between -90 and 90.
        if central_lat < -90 or central_lat > 90:
            raise Exception(
                "central_lat must be a real number between -90 and 90. The value of central_lat was: {}".format(
                    central_lat
                )
            )

        # Ensure central_long is between -180 and 180.
        if central_long < -180 or central_long > 180:
            raise Exception(
                "central_long must be a real number between -180 and 180. The value of central_long was: {}".format(
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
        latitude = self.latitude_lines()
        longitude = self.longitude_lines()

        u = self.zonal_wind()[indx_psurface, :, :]
        v = self.meridional_wind()[indx_psurface, :, :]

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

        # Add geostrophic wind vectors.
        plt.quiver(
            longitude.value,
            latitude.value,
            u_norm.value,
            v_norm.value,
            transform=ccrs.PlateCarree(),   
        )

        # Add SALT.
        plt.xlabel("Latitude ($\phi$)")
        plt.ylabel("Longitude ($\lambda$)")
        if not self.input_data:
            plt.title("Geostrophic Wind ("
                + str(self.date.year) + '-' + str(self.date.month) + '-'
                + str(self.date.day) + " " + str(self.date.hour)
                + ":00 h, Pressure Surface = "
                + str(self.pressure_surfaces()[indx_psurface]) +")"
            )
        else:
            plt.title("Geostrophic Wind ("
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
        plt.figtext(
            0.99,
            0.01,
            "Note: Geostrophic balance does not hold near the equator.",
            horizontalalignment="right",
        )

        plt.show()
        plt.close()
