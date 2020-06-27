#cython: linetrace=True
#distutils: define_macros=CYTHON_TRACE_NOGIL=1
#cython: language_level=3
"""
AMSIMP Wind Class. For information about this class is described below.
"""

# -----------------------------------------------------------------------------------------#

# Importing Dependencies
import os
import wget
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
    values for wind, specifically geostrophic wind, in the troposphere and the
    stratosphere. It also contain two methods for the visualisation of these
    numerical values.

    Below is a list of the methods included within this class, with a short
    description of their intended purpose. Please see the relevant class methods
    for more information.

    geostrophic_wind ~ generates a NumPy array of the components of
    geostrophic wind. The unit of measurement is metres per second.
    wind ~ generates a NumPy array of the components of wind (non-geostrophic).
    The unit of measurement is metres per second.
    ageostrophic_wind ~ generates a NumPy array of the components of the
    ageostrophic wind. The unit of measurement is metres per second.
    static_stability ~ generates a NumPy array of the the static
    stability within a vertical profile.
    verical_motion ~ generates a NumPy array of the vertical motion
    with the atmosphere, by integrating the quasi-geostrophic
    mass continunity equation.
    wind_contourf ~ generates a vertical profile of geostrophic wind,
    and outputs this as a contour plot.
    globe ~ generates a geostrophic wind contour plot, adds wind vectors to
    that said plot, and overlays both on a Nearside Projection of the Earth.
    """

    cpdef tuple geostrophic_wind(self):
        """
        This method outputs the components of geostrophic wind, in
        the shape of (len(pressure_surfaces), len(latitude_lines),
        len(longitude_lines)). 
        
        Geostrophic wind is a theoretical wind that is a result of a perfect
        balance between the Coriolis force and the pressure gradient force.
        This balance is known as geostrophic balance. 

        Equation:
            u_g = -frac{g}{f} \* frac{1}{a} \* frac{\partial \Phi}{\partial y}
            v_g = frac{g}{f} \* frac{1}{a} \* frac{\partial \Phi}{\partial x}
        
        Note: Geostrophic balance does not hold near the equator. For
        non-geostrophic wind, please see the method, wind.
        """
        # Gradient of geopotential height over latitudinal distance.
        cdef np.ndarray latitude = np.radians(self.latitude_lines().value)
        cdef np.ndarray height_gradienty = np.gradient(
            self.geopotential_height(),
            self.a * latitude,
            axis=1,
        )

        # Gradient of geopotential height over longitudinal distance.
        cdef np.ndarray longitude = np.radians(self.longitude_lines().value)
        cdef np.ndarray height_gradientx = np.gradient(
            self.geopotential_height().value,
            longitude,
            axis=2,
        )
        height_gradientx *= self.units.m
        height_gradientx = self.gradient_x(parameter=height_gradientx)

        # Defining a 3D coriolis parameter NumPy array.
        cdef np.ndarray coriolis_parameter = self.coriolis_parameter().value
        coriolis_parameter = np.resize(
            coriolis_parameter, (
                height_gradienty.shape[0],
                height_gradienty.shape[2],
                height_gradienty.shape[1],
            )
        ) / self.units.s
        coriolis_parameter = np.transpose(coriolis_parameter, (0, 2, 1))

        # Geostrophic wind calculation (zonal component).
        u_g = (
            -(self.gravitational_acceleration() / coriolis_parameter)
            * height_gradienty
        )

        # Convert zonal wind to metres per second.
        u_g = u_g.si

        # Geostrophic wind calculation (meridional component).
        v_g = (
            (self.gravitational_acceleration() / coriolis_parameter)
            * height_gradientx
        )
        # Convert meridional wind to metres per second.
        v_g = v_g.si

        return u_g, v_g

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
            ) * np.gradient(
                theta, self.pressure_surfaces(), axis=0
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
                wind_file = wget.download(url)
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
        the atmosphere, by integrating the quasi-geostrophic
        mass continunity equation. 
        
        The basic idea of quasi-geostrophic theory is that
        it reveals how hydrostatic balance and geostrophic
        balance constrain and simply atmospheric dynamics, but,
        in a realistic manner. Typical large-scale
        vertical motions in the atmosphere are of the order
        of 0.01 – 0.1 m s−1.

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
        v_dy = np.gradient(v, self.a * latitude, axis=1)

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

    def wind_contourf(self, which_wind=0, central_long=-7.6921):
        """
        For zonal wind, set which_wind to 0.
        For meridional wind, set which_wind to 1.
        For zonal (geostrophic) wind, set which_wind to 2.
        For meridional (geostrophic) wind, set which_wind to 3.
        
        Generates a geostrophic wind, or non-geostrophic wind contour
        plot, with the axes being latitude, and pressure surfaces.
        
        For the raw data, please use the
        amsimp.Wind.geostrophic_wind() or the amsimp.Wind.wind() method.
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
        
        # Set wind to either the zonal, or meridional component.
        if which_wind == 0:
            wind = self.wind()[0][:, :, indx_long]
            wind_type = "Zonal"
        elif which_wind == 1:
            wind = self.wind()[1][:, :, indx_long]
            wind_type = "Meridional"
        elif which_wind == 2:
            wind = self.geostrophic_wind()[0][:, :, indx_long]
            wind_type = "Zonal Geostrophic"
        elif which_wind == 3:
            wind = self.geostrophic_wind()[1][:, :, indx_long]
            wind_type = "Meridional Geostrophic"
        else:
            raise Exception(
                "which_wind must be equal to between 0 and 3. The value of which_wind was: {}".format(
                    which_wind
                )
            )

        # Defines the axes, and the data.
        latitude, pressure_surfaces = np.meshgrid(self.latitude_lines(), self.pressure_surfaces())

        # Specifies the contour levels
        minimum = wind.min()
        maximum = wind.max()
        levels = np.linspace(minimum, maximum, 21)

        # Contour plotting.
        plt.contourf(latitude, pressure_surfaces, wind, levels=levels)

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
        if which_wind > 1:
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

    def globe(self, central_lat=53.1424, central_long=-7.6921, psurface=1000, which_wind=0):
        """
        Similiar to amsimp.Wind.wind_contourf(), however, this particular method
        adds wind vectors to the contour plot. It also overlays both of these
        elements onto a Nearside Perspective projection of the Earth.

        By default, the perspective view is looking directly down at the city
        of Dublin in the country of Ireland. If which_wind is set to 0, it will
        plot the non-geostophic wind data, otherwise, if it is set to 1, it will
        plot the geostrophic wind data.

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

        if which_wind == 0:
            wind = self.wind()
            u = wind[0][indx_psurface, :, :]
            v = wind[1][indx_psurface, :, :]
            title = "Wind"
        elif which_wind == 1:
            geostrophic_wind = self.geostrophic_wind()
            u = geostrophic_wind[0][indx_psurface, :, :]
            v = geostrophic_wind[1][indx_psurface, :, :]
            title = "Geostrophic Wind"
        else:
            raise Exception(
                "which_wind must be equal to between 0 and 1. The value of which_wind was: {}".format(
                    which_wind
                )
            )

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
        plt.figtext(
            0.99,
            0.01,
            "Note: Geostrophic balance does not hold near the equator.",
            horizontalalignment="right",
        )

        plt.show()
        plt.close()
