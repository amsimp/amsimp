#cython: linetrace=True
#distutils: define_macros=CYTHON_TRACE_NOGIL=1
#cython: language_level=3
"""
AMSIMP Dynamics Class. For information about this class is described below.
"""

# -----------------------------------------------------------------------------------------#

# Importing Dependencies
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.gridspec as gridspec
from matplotlib import ticker
import numpy as np
from pynverse import inversefunc
from scipy.optimize import curve_fit
import cartopy.crs as ccrs
from amsimp.wind cimport Wind
from amsimp.wind import Wind
from amsimp.water cimport Water
from amsimp.water import Water

# -----------------------------------------------------------------------------------------#


cdef class Dynamics(Water):
    """
    AMSIMP Dynamics Class - Also, known as Motus Aeris @ AMSIMP. This class
    generates rudimentary simulation of tropospheric and stratsopheric
    dynamics on a synoptic scale. Predictions are made by calculating the
    derivative of each element using finite-difference. The initial conditions
    are defined in the class methods of Water, Wind, and Backend. For more
    information on the initial conditions, please see those classes.

    Below is a list of the methods included within this class, with a short
    description of their intended purpose. Please see the relevant class methods
    for more information.

    forecast_temperature ~ this method outputs the forecasted temperature
    for the specified number of forecast days. Every single day is divided
    into 6, meaning the length of the outputted list is six times the number
    of forecast days specified.
    forecast_pressure ~ 
    forecast_pressurethickness ~ 
    forecast_precipitablewater ~ this method outputs the forecasted precipitable
    water for the specified number of forecast days. Every single day is divided
    into 6, meaning the length of the outputted list is six times the number
    of forecast days specified.

    simulate ~ this method outputs a visualisation of how temperature, pressure
    thickness, geostrophic wind, and precipitable water vapor will evolve for
    the specified number of forecast days.
    """

    def __cinit__(self, int detail_level=3, int forecast_days=3):
        """
        Defines the number of days that will be included within the simulation.
        This value must be greater than 0, and less than 5 in order
        to ensure that the simulation methods function correctly. Defaults to
        a value of 3.

        For more information, please refer to amsimp.Backend.__cinit__()
        method.
        """
        # Declare class variables.
        super().__init__(detail_level)
        self.forecast_days = forecast_days

        # Ensure self.forecast_days is between 0 and 10.
        if self.forecast_days > 5 or self.forecast_days <= 0:
            raise Exception(
                "forecast_days must be a positive integer between 1 and 5. The value of forecast_days was: {}".format(
                    self.forecast_days
                )
            )

    def delta_xyz(self):
        """
        Defines delta_x (the distance in metres between lines
        of latitude), delta_y (the distance in metres between
        lines of longitude), and delta_z (the distance between
        altitude levels). Please do not interact with
        the method directly.
        """
        # delta_x
        # Distance between longitude lines at the equator.
        cdef eq_longd = 111.19 * self.units.km
        eq_longd = eq_longd.to(self.units.m)
        # Distance of one degree of longitude (e.g. 0W - 1W/1E), measured in metres.
        # The distance between two lines of longitude is not constant.
        cdef np.ndarray long_d = np.cos(self.latitude_lines()) * eq_longd
        # Distance between latitude lines in the class method,
        # amsimp.Backend.latitude_lines().
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

        # delta_y
        # Distance of one degree of latitude (e.g. 0N - 1N/1S), measured in metres.
        cdef lat_d = (2 * np.pi * self.a) / 360
        # Distance between latitude lines in the class method, 
        # amsimp.Backend.latitude_lines().
        cdef delta_y = (
            self.latitude_lines()[-1].value - self.latitude_lines()[-2].value
        ) * lat_d

        # delta_z
        cdef delta_z = self.altitude_level()[1]

        return delta_x, delta_y, delta_z

    cpdef list forecast_temperature(self):
        """
        Description is placed here.

        Equation:
        \frac{\partial T}{\partial t} = u \* \frac{\partial T}{\partial x} +
                                        v \* \frac{\partial T}{\partial y} +
                                        w \* \frac{\partial T}{\partial z}
        """
        # Define the forecast period.
        forecast_days = int(self.forecast_days)
        cdef np.ndarray time = np.linspace(
            0, forecast_days, (forecast_days * 1440)
        )

        # Define the initial temperature condition.
        cdef np.ndarray temperature = self.temperature()

        # Define delta_x, delta_y, and delta_y.
        delta_xyz = self.delta_xyz()
        cdef np.ndarray delta_x = delta_xyz[0]
        cdef delta_y = delta_xyz[1]
        cdef delta_z = delta_xyz[2]

        # Define delta_t
        delta_t = time[1] - time[0]
        delta_xval = delta_x.value
        delta_t = delta_t + np.zeros(
            (len(delta_xval), len(delta_xval[0]), len(delta_xval[0][0]))
        )
        delta_t *= self.units.day
        delta_t = delta_t.to(self.units.s)

        # Define the zonal, meridional, and vertical wind velocities.
        cdef np.ndarray u = self.zonal_wind()
        cdef np.ndarray v = self.meridional_wind()
        w = 0 * (self.units.m / self.units.s)

        # Calculate how temperature will evolve over time.
        forecast_temperature = []
        cdef list temperature_gradient
        cdef np.ndarray temperature_gradientx, temperature_gradienty
        cdef np.ndarray temperature_gradientz
        for t in time:
            # Define the temperature gradient.
            temperature_gradient = np.gradient(temperature)

            # Define the temperature gradient in the longitude, latitude, and
            # altitude directions.
            temperature_gradientx = temperature_gradient[0]
            temperature_gradienty = temperature_gradient[1]
            temperature_gradientz = temperature_gradient[2]

            # Calculate how temperature at a particular point in time.
            temp = temperature + (
                u * (delta_t / delta_x) * temperature_gradientx
            ) + (
                v * (delta_t / delta_y) * temperature_gradienty
            ) + (
                w * (delta_t / delta_z) * temperature_gradientz
            )
            temperature = temp.copy()

            # Store the data within a list.
            forecast_temperature.append(temp)
        
        forecast_temperature = forecast_temperature[::60]
        
        # Remove extreme values from the forecast.
        one_percentile = np.percentile(forecast_temperature, 1)
        temp = []
        for forecast_temp in forecast_temperature:
            forecast_temp = forecast_temp.value
            forecast_temp[forecast_temp < one_percentile] = one_percentile
            forecast_temp *= self.units.K
            temp.append(forecast_temp)

        return forecast_temperature
    
    cpdef list forecast_pressure(self):
        """
        Description is place here.
        """
        # Store the forecasted temperature in a variable. Atmospheric
        # density remains constant as described by the Conservation
        # of Mass Equation (Primitive Equations).
        cdef list forecast_temperature = self.forecast_temperature()
        cdef np.ndarray density = self.density()
        # Universal gas constant (J * kg^-1 * K^-1)
        cdef R = 287 * (self.units.J / (self.units.kg * self.units.K))

        # Generate a forecast for pressure for the specified number of
        # days.
        cdef np.ndarray forecast_temp, forecast_p
        forecast_pressure = []
        for forecast_temp in forecast_temperature:
            forecast_p = density * R * forecast_temp
            forecast_p = forecast_p.to(self.units.hPa)

            forecast_pressure.append(forecast_p)

        return forecast_pressure

    cpdef fit_method(self, x, a, b, c):
        """
        This method is solely utilised for non-linear regression in the
        amsimp.Dynamics.forecast_pressurethickness() method. Please do not
        interact with the method directly.
        """
        return a - (b / c) * (1 - np.exp(-c * x))

    cpdef list forecast_pressurethickness(self):
        """
        Description is place here.
        """
        # Store the forecasted temperature in a variable.
        cdef list forecast_pressure = self.forecast_pressure()
        cdef np.ndarray altitude = self.altitude_level()[:20].value
        
        forecast_pressurethickness = []
        cdef np.ndarray forecast_p, pressure_thickness
        cdef np.ndarray p, p_lat, abc
        cdef list list_pressurethickness = []
        cdef list guess = [1000, 0.12, 0.00010]
        cdef list pthickness_lat
        cdef tuple c
        cdef float hPa1000_height, hPa500_height, pthickness
        for forecast_p in forecast_pressure:
            forecast_p = forecast_p.value

            list_pressurethickness = []
            for p in forecast_p:
                pthickness_lat = []
                for p_lat in p:
                    p_lat = p_lat[:20]

                    c = curve_fit(self.fit_method, altitude, p_lat, guess)
                    abc = c[0]

                    inverse_fitmethod = inversefunc(self.fit_method,
                        args=(abc[0], abc[1], abc[2])
                    )

                    hPa1000_height = inverse_fitmethod(1000)
                    hPa500_height = inverse_fitmethod(500)
                    pthickness = hPa500_height - hPa1000_height

                    pthickness_lat.append(pthickness)
                list_pressurethickness.append(pthickness_lat)
            
            pressure_thickness = np.asarray(list_pressurethickness)
            pressure_thickness *= self.units.m
            forecast_pressurethickness.append(pressure_thickness)

        return forecast_pressurethickness

    cpdef list forecast_precipitablewater(self):
        """
        Description is placed here.

        Equation:
        \frac{\partial W}{\partial t} = u \* \frac{\partial W}{\partial x} +
                                        v \* \frac{\partial W}{\partial y} +
                                        w \* \frac{\partial W}{\partial z}
        """
        # Define the forecast period.
        forecast_days = int(self.forecast_days)
        cdef np.ndarray time = np.linspace(
            0, forecast_days, (forecast_days * 1440)
        )

        # Define the initial precipitable water condition.
        cdef np.ndarray precipitable_water = self.precipitable_water(
            sum_altitude=False
        )

        # Define delta_x, delta_y, and delta_y.
        delta_xyz = self.delta_xyz()
        cdef np.ndarray delta_x = delta_xyz[0][:, :, :40]
        cdef delta_y = delta_xyz[1]
        cdef delta_z = delta_xyz[2]

        # Define delta_t
        delta_t = time[1] - time[0]
        delta_xval = delta_x.value
        delta_t = delta_t + np.zeros(
            (len(delta_xval), len(delta_xval[0]), len(delta_xval[0][0]))
        )
        delta_t *= self.units.day
        delta_t = delta_t.to(self.units.s)

        # Define the zonal, meridional, and vertical wind velocities.
        cdef np.ndarray u = self.zonal_wind()[:, :, :40]
        cdef np.ndarray v = self.meridional_wind()[:, :, :40]
        w = 0 * (self.units.m / self.units.s)

        # Calculate how precipitable water will evolve over time.
        forecast_precipitablewater = []
        cdef list pwv_gradient
        cdef np.ndarray pwv_gradientx, pwv_gradienty, pwv_gradientz
        for t in time:
            # Define the precipitable water gradient.
            pwv_gradient = np.gradient(precipitable_water)

            # Define the precipitable water gradient in the longitude,
            # latitude, and altitude directions.
            pwv_gradientx = pwv_gradient[0]
            pwv_gradienty = pwv_gradient[1]
            pwv_gradientz = pwv_gradient[2]

            # Calculate how temperature at a particular point in time.
            pwv = precipitable_water + (
                u * (delta_t / delta_x) * pwv_gradientx
            ) + (
                v * (delta_t / delta_y) * pwv_gradienty
            ) + (
                w * (delta_t / delta_z) * pwv_gradientz
            )
            precipitable_water = pwv.copy()

            # Sum by the altitude component.
            pwv_output = pwv.value
            list_pwv = []
            for pwv_long in pwv_output:
                list_pwvlat = []
                for pwv_lat in pwv_long:
                    pwv_alt = np.sum(pwv_lat)
                    list_pwvlat.append(pwv_alt)
                list_pwv.append(list_pwvlat)
            pwv_output = np.asarray(list_pwv) * self.units.mm

            # Store the predicted amount of precipitable water into a list.
            forecast_precipitablewater.append(pwv_output)

        forecast_precipitablewater = forecast_precipitablewater[::60]

        # Remove extreme values from the forecast.
        one_percentile = np.percentile(forecast_precipitablewater, 1)
        ninetynine_percentile = np.percentile(
            forecast_precipitablewater, 99
        )
        pwv = []
        for forecast_pwv in forecast_precipitablewater:
            forecast_pwv = forecast_pwv.value
            forecast_pwv[forecast_pwv < one_percentile] = one_percentile
            forecast_pwv[
                forecast_pwv > ninetynine_percentile
            ] = ninetynine_percentile
            forecast_pwv *= self.units.mm
            pwv.append(forecast_pwv)

        forecast_precipitablewater = pwv
        return forecast_precipitablewater

    def simulate(self):
        """
        This method outputs a visualisation of how temperature, pressure
        thickness, atmospheric pressure, and precipitable water vapor will evolve.
        The atmospheric pressure and precipitable water elements of this 
        visualisation operate similarly to the method, 
        amsimp.Backend.longitude_contourf(), so, please refer to this method for a
        detailed description of the aforementioned elements. Likewise, please refer
        to amsimp.Water.water_contourf() for more information on the visualisation
        element of precipitable water vapor, or to
        amsimp.Backend.altitude_contourf() for more information on the visualisation
        element of temperature.

        For a visualisation of geostrophic wind (zonal and meridional components),
        please refer to the amsimp.Wind.wind_contourf(), or amsimp.Wind.globe()
        methods.
        """
        # Define the forecast period.
        forecast_days = int(self.forecast_days)
        cdef np.ndarray time = np.linspace(
            0, forecast_days, (forecast_days * 24)
        )

        # Style of graph.
        style.use("fivethirtyeight")

        # Define layout.
        gs = gridspec.GridSpec(2, 2)
        fig = plt.figure(figsize=(18.5, 7.5))
        fig.subplots_adjust(hspace=0.340, bottom=0.105, top=0.905)
        plt.ion()

        # Temperature
        indx_long = (np.abs(self.longitude_lines().value - 0)).argmin()
        ax1 = plt.subplot(gs[0, 0])
        forecast_temp = self.forecast_temperature()
        # Atmospheric Pressure
        ax2 = plt.subplot(gs[1, 0], projection=ccrs.EckertIII())
        forecast_pressure = self.forecast_pressure()
        # Precipitable Water
        ax3 = plt.subplot(gs[0, 1], projection=ccrs.EckertIII())
        forecast_Pwv = self.forecast_precipitablewater()
        # Pressure Thickness (1000hPa - 500hPa).
        ax4 = plt.subplot(gs[1, 1], projection=ccrs.EckertIII())
        forecast_pthickness = self.forecast_pressurethickness()

        # Troposphere - Stratosphere Boundary Line
        trop_strat_line = self.troposphere_boundaryline()
        trop_strat_line = (
            np.zeros(len(trop_strat_line.value)) + np.mean(trop_strat_line)
        )

        t = 0
        while t < len(time):
            # Defines the axes.
            # For the temperature contour plot.
            latitude, altitude = np.meshgrid(
                self.latitude_lines(), self.altitude_level()
            )
            # For the pressure, precipitable water, and pressure 
            # thickness countour plot
            lat, long = np.meshgrid(
                self.latitude_lines(), self.longitude_lines()
            )
            
            # Temperature contour plot.
            # Temperature data.
            temperature = forecast_temp[t]
            temperature = temperature[indx_long, :, :]
            temperature = np.transpose(temperature)

            # Contour plotting.
            cmap1 = plt.get_cmap("hot")
            min1 = np.min(forecast_temp)
            max1 = np.max(forecast_temp)
            level1 = np.linspace(min1, max1, 21)
            temp = ax1.contourf(
                latitude, altitude, temperature, cmap=cmap1, levels=level1
            )

            # Checks for a colorbar.
            if t == 0:
                cb1 = fig.colorbar(temp, ax=ax1)
                tick_locator = ticker.MaxNLocator(nbins=10)
                cb1.locator = tick_locator
                cb1.update_ticks()
                cb1.set_label("Temperature (K)")

            # Add SALT to the graph.
            ax1.set_xlabel("Latitude ($\phi$)")
            ax1.set_ylabel("Altitude (m)")
            ax1.set_title("Temperature")

            # Atmospheric pressure contour.
            # Pressure data.
            pressure = forecast_pressure[t]
            pressure = pressure[:, :, 0]

            # EckertIII projection details.
            ax2.set_global()
            ax2.coastlines()
            ax2.gridlines()

            # Contourf plotting.
            pressure_sealevel = np.asarray(forecast_pressure)
            pressure_sealevel = pressure_sealevel[:, :, :, 0]
            cmap2 = plt.get_cmap("jet")
            min2 = np.min(pressure_sealevel)
            max2 = np.max(pressure_sealevel)
            level2 = np.linspace(min2, max2, 21)
            atmospheric_pressure = ax2.contourf(
                long,
                lat,
                pressure,
                cmap=cmap2,
                levels=level2,
                transform=ccrs.PlateCarree(),
            )

            # Checks for a colorbar.
            if t == 0:
                cb2 = fig.colorbar(atmospheric_pressure, ax=ax2)
                tick_locator = ticker.MaxNLocator(nbins=10)
                cb2.locator = tick_locator
                cb2.update_ticks()
                cb2.set_label("Pressure (hPa)")

            # Add SALT to the graph.
            ax2.set_xlabel("Longitude ($\lambda$)")
            ax2.set_ylabel("Latitude ($\phi$)")
            ax2.set_title(
                "Atmospheric Pressure (Alt = " 
                + str(self.altitude_level()[0])
                + ")"
            )

            # Precipitable water contour.
            # Precipitable water data.
            precipitable_water = forecast_Pwv[t]

            # EckertIII projection details.
            ax3.set_global()
            ax3.coastlines()
            ax3.gridlines()

            # Contourf plotting.
            cmap3 = plt.get_cmap("seismic")
            min3 = np.min(forecast_Pwv)
            max3 = np.max(forecast_Pwv)
            level3 = np.linspace(min3, max3, 21)
            precipitable_watervapour = ax3.contourf(
                long,
                lat,
                precipitable_water,
                cmap=cmap3,
                levels=level3,
                transform=ccrs.PlateCarree(),
            )

            # Checks for a colorbar.
            if t == 0:
                cb3 = fig.colorbar(precipitable_watervapour, ax=ax3)
                tick_locator = ticker.MaxNLocator(nbins=10)
                cb3.locator = tick_locator
                cb3.update_ticks()
                cb3.set_label("Precipitable Water (mm)")

            # Add SALT to the graph.
            ax3.set_xlabel("Longitude ($\lambda$)")
            ax3.set_ylabel("Latitude ($\phi$)")
            ax3.set_title("Precipitable Water")

            # Pressure thickness scatter plot.
            # Pressure thickness data.
            pressure_thickness = forecast_pthickness[t]

            # EckertIII projection details.
            ax4.set_global()
            ax4.coastlines()
            ax4.gridlines()

            # Contourf plotting.
            min4 = np.min(forecast_pthickness)
            max4 = np.max(forecast_pthickness)
            level4 = np.linspace(min4, max4, 21)
            pressure_h = ax4.contourf(
                long,
                lat,
                pressure_thickness,
                transform=ccrs.PlateCarree(),
                levels=level4,
            )

            # Index of the rain / snow line
            indx_snowline = (np.abs(level4 - 5400)).argmin()
            pressure_h.collections[indx_snowline].set_color('black')
            pressure_h.collections[indx_snowline].set_linewidth(1) 

            # Add SALT.
            ax4.set_xlabel("Latitude ($\phi$)")
            ax4.set_ylabel("Longitude ($\lambda$)")
            ax4.set_title("Thickness (1000 hPa - 500 hPa)")

            # Checks for a colorbar.
            if t == 0:
                cb4 = fig.colorbar(pressure_h, ax=ax4)
                tick_locator = ticker.MaxNLocator(nbins=10)
                cb4.locator = tick_locator
                cb4.update_ticks()
                cb4.set_label("Pressure Thickness (m)")

            # Plots the average boundary line on two contourfs.
            # Temperature contourf.
            ax1.plot(
                latitude[1],
                trop_strat_line,
                color="black",
                linestyle="dashed",
                label="Troposphere - Stratosphere Boundary Line",
            )
            ax1.legend(loc=0)

            # Title of Simulation.
            now = self.date + timedelta(hours=+t)
            hour = now.hour
            minute = now.minute

            fig.suptitle(
                "Motus Aeris @ AMSIMP (" + str(hour)
                + ":" + str(minute) + " "
                + now.strftime("%d-%m-%Y") + ")"
            )

            # Displaying simualtion.
            plt.show()
            plt.pause(0.01)
            if t < (len(time) - 1):
                ax1.clear()
                ax2.clear()
                ax3.clear()
                ax4.clear()
            else:
                plt.pause(10)

            note = (
                "Note: Geostrophic balance does not hold near the equator."
                + " Rain / Snow Line is marked on the Pressure Thickness" 
                + " contour plot by the black line (5,400 m)."
            )

            # Footnote
            plt.figtext(
                0.99,
                0.01,
                note,
                horizontalalignment="right",
                fontsize=10,
            )

            t += 1
