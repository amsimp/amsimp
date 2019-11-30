#cython: linetrace=True
#distutils: define_macros=CYTHON_TRACE_NOGIL=1
#cython: language_level=3
"""
AMSIMP Wind Class. For information about this class is described below.
"""

# -----------------------------------------------------------------------------------------#

# Importing Dependencies
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from amsimp.backend cimport Backend
from amsimp.backend import Backend
cimport numpy as np

# -----------------------------------------------------------------------------------------#

cdef class Wind(Backend):
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
        the shape of (len(longitude_lines), len(latitude_lines),
        len(altitude_level)). 
        
        Geostrophic wind is a theoretical wind that is a result of a perfect
        balance between the Coriolis force and the pressure gradient force.
        This balance is known as geostrophic balance. 

        Equation:
            u_g = -frac{1}{rho * f} \* frac{\partial p}{\partial y}
        
        Note: Geostrophic balance does not hold near the equator.
        """
        # Ensures that the detail_level must be higher than 2 in order to utilise this method.
        if self.detail_level < 3:
            raise Exception(
                "detail_level must be greater than 2 in order to utilise this method."
            )

        # Distance of one degree of latitude (e.g. 0N - 1N/1S), measured in metres.
        cdef lat_d = (2 * np.pi * self.a) / 360
        # Distance between latitude lines in the class method, 
        # amsimp.Backend.latitude_lines().
        cdef delta_y = (
            self.latitude_lines()[-1].value - self.latitude_lines()[-2].value
        ) * lat_d
        delta_y *= 2

        # Gradient of geopotential height over latitudinal distance.
        pressure_gradient = np.gradient(self.pressure())
        pressure_gradient = pressure_gradient[1]
        cdef np.ndarray pressuregradient_deltay = pressure_gradient / delta_y

        # Defining a 3D coriolis parameter NumPy array.
        cdef np.ndarray coriolis_parameter = self.coriolis_parameter().value
        cdef list f_alt = []
        cdef int len_altitude = len(self.altitude_level())
        for f in coriolis_parameter:
            f = f + np.zeros(len_altitude)
            f = list(f)
            f_alt.append(f)
        
        cdef list list_coriolisparameter = []
        cdef int len_longitudelines = len(self.longitude_lines())
        cdef int n = 0
        while n < len_longitudelines:
            list_coriolisparameter.append(f_alt)
            n += 1
        coriolis_parameter = np.asarray(list_coriolisparameter)
        coriolis_parameter /= self.units.s

        # Geostrophic wind calculation (zonal component).
        zonal_wind = (
            -(1 / (self.density() * coriolis_parameter))
            * pressuregradient_deltay
        )

        # Convert zonal wind to metres per second.
        zonal_wind = zonal_wind.si

        return zonal_wind

    cpdef np.ndarray meridional_wind(self):
        """
        This method outputs the meridional component of geostrophic wind, in
        the shape of (len(longitude_lines), len(latitude_lines),
        len(altitude_level)). 
        
        Geostrophic wind is a theoretical wind that is a result of a perfect
        balance between the Coriolis force and the pressure gradient force.
        This balance is known as geostrophic balance.

        Equation:
            v_g = frac{1}{rho * f} \* frac{\partial p}{\partial y} 
        
        Note: Geostrophic balance does not hold near the equator.
        """
        # Ensures that the detail_level must be higher than 2 in order to utilise this method.
        if self.detail_level < 3:
            raise Exception(
                "detail_level must be greater than 2 in order to utilise this method."
            )

        # Distance between longitude lines at the equator.
        cdef eq_longd = 111.19 * self.units.km
        eq_longd = eq_longd.to(self.units.m)
        # Distance of one degree of longitude (e.g. 0W - 1W/1E), measured in metres.
        # The distance between two lines of longitude is not constant.
        cdef np.ndarray long_d = np.cos(self.latitude_lines()) * eq_longd
        # Distance between latitude lines in the class method, amsimp.Backend.latitude_lines().
        cdef np.ndarray delta_x = (
            self.longitude_lines()[-1].value - self.longitude_lines()[-2].value
        ) * long_d
        
        # Defining a 3D longitudinal distance NumPy array.
        delta_x = delta_x.value
        cdef list long_alt = []
        cdef int len_altitude = len(self.altitude_level())
        for x in delta_x:
            x = x + np.zeros(len_altitude)
            x = list(x)
            long_alt.append(x)

        cdef list list_deltax = []
        cdef int len_longitudelines = len(self.longitude_lines())
        cdef int n = 0
        while n < len_longitudelines:
            list_deltax.append(long_alt)
            n += 1
        delta_x = np.asarray(list_deltax)
        delta_x *= self.units.m
        delta_x *= 2

        # Gradient of geopotential height over latitudinal distance.
        pressure_gradient = np.gradient(self.pressure())
        pressure_gradient = pressure_gradient[0]
        cdef np.ndarray pressuregradient_deltax = pressure_gradient / delta_x

        # Defining a 3D coriolis parameter NumPy array.
        cdef np.ndarray coriolis_parameter = self.coriolis_parameter().value
        cdef list f_alt = []
        for f in coriolis_parameter:
            f = f + np.zeros(len_altitude)
            f = list(f)
            f_alt.append(f)
        
        cdef list list_coriolisparameter = []      
        n = 0
        while n < len_longitudelines:
            list_coriolisparameter.append(f_alt)
            n += 1
        coriolis_parameter = np.asarray(list_coriolisparameter)
        coriolis_parameter /= self.units.s

        # Geostrophic wind calculation (meridional component).
        meridional_wind = (
            (1 / (self.density() * coriolis_parameter))
            * pressuregradient_deltax
        )

        # Convert meridional wind to metres per second.
        meridional_wind = meridional_wind.si

        return meridional_wind

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
            geostrophic_wind = self.zonal_wind()[indx_long, :, :]
            wind_type = "Zonal"
        elif which_wind == 1:
            geostrophic_wind = self.meridional_wind()[indx_long, :, :]
            wind_type = "Meridional"
        else:
            raise Exception(
                "which_wind must be equal to either zero or one. The value of which_wind was: {}".format(
                    which_wind
                )
            )

        # Defines the axes, and the data.
        latitude, altitude = np.meshgrid(self.latitude_lines(), self.altitude_level())
        geostrophic_wind = np.transpose(geostrophic_wind)

        # Specifies the contour levels
        minimum = geostrophic_wind.min()
        maximum = geostrophic_wind.max()
        levels = np.linspace(minimum, maximum, 21)

        # Contour plotting.
        plt.contourf(latitude, altitude, geostrophic_wind, levels=levels)

        plt.set_cmap("jet")

        # Add SALT.
        plt.xlabel("Latitude ($\phi$)")
        plt.ylabel("Altitude (m)")
        plt.title(wind_type + " Wind Contour Plot ("
         + self.date.strftime("%d-%m-%Y") + ", Longitude = "
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
            len(troposphere_boundaryline[0])
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

    def globe(self, central_lat=53.1424, central_long=-7.6921, alt=0):
        """
        Similiar to amsimp.Wind.wind_contourf(), however, this particular method
        adds wind vectors to the contour plot. It also overlays both of these
        elements onto a Nearside Perspective projection of the Earth.

        By default, the perspective view is looking directly down at the city
        of Dublin in the country of Ireland.

        Known bug(s):
        Some vectors appear to float to the side of the projection.

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
        
        # Ensure alt is between 0 and 50000 metres above sea level.
        if alt < 0 or alt > 50000:
            raise Exception(
                "alt must be a real number between 0 and 50,000. The value of alt was: {}".format(
                    alt
                )
            )
        
        # Index of the nearest alt in amsimp.Backend.altitude_level()
        indx_alt = (np.abs(self.altitude_level().value - alt)).argmin()

        # Ignore NumPy errors.
        np.seterr(all="ignore")

        # Define the axes, and the data.
        latitude, longitude = np.meshgrid(self.latitude_lines().value,
         self.longitude_lines().value
        )
        u = self.zonal_wind()[:, :, indx_alt].value
        v = self.meridional_wind()[:, :, indx_alt].value

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

        contourf = plt.contourf(
            longitude,
            latitude,
            geostrophic_wind,
            transform=ccrs.PlateCarree(),
            levels=levels,
        )

        # Add geostrophic wind vectors.
        plt.quiver(longitude, latitude, u_norm, v_norm, transform=ccrs.PlateCarree())

        # Add SALT.
        plt.xlabel("Latitude ($\phi$)")
        plt.ylabel("Longitude ($\lambda$)")
        plt.title("Geostrophic Wind ("
         + self.date.strftime("%d-%m-%Y") + ", Altitude = "
         + str(self.altitude_level()[indx_alt]) +")"
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
