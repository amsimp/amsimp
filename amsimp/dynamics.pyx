#cython: language_level=3
"""
AMSIMP Dynamics Class. For information about this class is described below.
"""

# -----------------------------------------------------------------------------------------#

# Importing Dependencies
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.gridspec as gridspec
from matplotlib import ticker
import numpy as np
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
    predict_pressurethickness ~ 
    forecast_precipitablewater ~ this method outputs the forecasted precipitable
    water for the specified number of forecast days. Every single day is divided
    into 6, meaning the length of the outputted list is six times the number
    of forecast days specified.

    simulate ~ this method outputs a visualisation of how temperature, pressure
    thickness, geostrophic wind, and precipitable water vapor will evolve for
    the specified number of forecast days.
    """

    def __cinit__(self, detail_level, int forecast_days=3):
        """
        Defines the number of days that will be included within the simulation.
        This value must be greater than 0, and less than 5 in order
        to ensure that the simulation methods function correctly.

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

    def forecast_temperature(self):
        """
        Description is placed here.
        
        Known bug(s):
        For some unknown reason, it seems to generate a few 
        temperature extremities, i.e. really low Kelvin values (143 K),
        or really high Kelvin values (316 K). However, the majority of
        values are in line with expectations.
        """
        forecast_days = int(self.forecast_days)
        time = np.linspace(
            0, forecast_days, (forecast_days * 6)
        )

        # Define the initial temperature condition.
        temperature = self.temperature()

        # Define delta_x, delta_y, and delta_y.
        delta_xyz = self.delta_xyz()
        delta_x = delta_xyz[0]
        delta_y = delta_xyz[1]
        delta_z = delta_xyz[2]

        # Define the zonal, meridional, and vertical wind velocities.
        u = self.zonal_wind()
        v = self.meridional_wind()
        w = 0 * (self.units.m / self.units.s)

        # Calculate how temperature will evolve over time.
        forecast_temperature = []
        for t in time:
            # Change time increment from days to seconds.
            t = t * self.units.day
            t = t.to(self.units.s)

            # Define the temperature gradient.
            temperature_gradient = np.gradient(temperature)

            # Define the temperature gradient in the longitude, latitude, and
            # altitude directions.
            temperature_gradientx = temperature_gradient[0]
            temperature_gradienty = temperature_gradient[1]
            temperature_gradientz = temperature_gradient[2]

            # Calculate how temperature will change over time.
            delta_T_over_delta_t = (
                (u * (temperature_gradientx / delta_x))
                + (v * (temperature_gradienty / delta_y))
                + (w * (temperature_gradientz / delta_z))
            )

            # Predict the temperature on a given day.
            # y = mx + c
            temp = temperature + (delta_T_over_delta_t * t)

            # Store the predicted temperature into a list
            forecast_temperature.append(temp)

        return forecast_temperature

    def predict_pressurethickness(self):
        """
        This is the pressure thickness variation of the method,
        predict_temperature. Please refer to
        amsimp.Backend.predict_temperature() for a general description of this
        method.
        """
        

    def forecast_precipitablewater(self):
        """
        Description is placed here.

        Known bug(s):
        For some unknown reason, it seems to generate a few 
        precipitable water extremities, i.e. really high amounts of
        precipitable water (93 mm). However, the majority of
        values are in line with expectations.
        """
        forecast_days = int(self.forecast_days)
        time = np.linspace(
            0, forecast_days, (forecast_days * 6)
        )

        # Define the initial precipitable water condition.
        precipitable_water = self.precipitable_water(sum_altitude=False)

        # Define delta_x, delta_y, and delta_y.
        delta_xyz = self.delta_xyz()
        delta_x = delta_xyz[0]
        delta_x = delta_x[:, :, :40]
        delta_y = delta_xyz[1]
        delta_z = delta_xyz[2]


        # Define the zonal, meridional, and vertical wind velocities.
        u = self.zonal_wind()[:, :, :40]
        v = self.meridional_wind()[:, :, :40]
        w = 0 * (self.units.m / self.units.s)

        # Calculate how precipitable water will evolve over time.
        forecast_precipitablewater = []
        for t in time:
            # Change time increment from days to seconds.
            t = t * self.units.day
            t = t.to(self.units.s)

            # Define the precipitable water gradient.
            pwv_gradient = np.gradient(precipitable_water)

            # Define the precipitable water gradient in the longitude,
            # latitude, and altitude directions.
            pwv_gradientx = pwv_gradient[0]
            pwv_gradienty = pwv_gradient[1]
            pwv_gradientz = pwv_gradient[2]

            # Calculate how precipitable water will change over time.
            delta_W_over_delta_t = (
                (u * (pwv_gradientx / delta_x))
                + (v * (pwv_gradienty / delta_y))
                + (w * (pwv_gradientz / delta_z))
            )

            # Predict the amount of precipitable water on a given day.
            # y = mx + c
            pwv = precipitable_water + (delta_W_over_delta_t * t)

            # Sum by the altitude component.
            pwv = pwv.value
            list_pwv = []
            for pwv_long in pwv:
                list_pwvlat = []
                for pwv_lat in pwv_long:
                    pwv_alt = np.sum(pwv_lat)
                    list_pwvlat.append(pwv_alt)
                list_pwv.append(list_pwvlat)
            pwv = np.asarray(list_pwv) * self.units.mm

            # Store the predicted amount of precipitable water into a list.
            forecast_precipitablewater.append(pwv)

        return forecast_precipitablewater

    def simulate(self):
        """
        This method outputs a visualisation of how temperature, pressure
        thickness, geostrophic wind, and precipitable water vapor will evolve.
        The geostrophic wind and temperature elements of this visualisation
        operate similarly to the method, amsimp.Wind.wind_contourf(), so, please
        refer to this method for a detailed description of the aforementioned
        elements. Likewise, please refer to amsimp.Water.water_contourf() for
        more information on the visualisation element of precipitable water
        vapor.
        """
        # Time (Unit: day).
        time = np.linspace(0, (self.number_of_days - 1), (self.number_of_days * 2))

        # Style of graph.
        style.use("fivethirtyeight")

        # Define layout.
        gs = gridspec.GridSpec(2, 2)
        fig = plt.figure(figsize=(18.5, 7.5))
        fig.subplots_adjust(hspace=0.340, bottom=0.105, top=0.905)
        plt.ion()

        # Geostrophic Wind
        ax1 = plt.subplot(gs[0, 0])
        predict_u = self.zonal_wind()
        # Temperature
        ax2 = plt.subplot(gs[1, 0])
        predict_t = self.predict_temperature()
        # Precipitable Water
        ax3 = plt.subplot(gs[0, 1], projection=ccrs.EckertIII())
        predict_Pwv = np.asarray(self.predict_precipitablewater())
        # Pressure Thickness
        ax4 = plt.subplot(gs[1, 1])
        predict_pthickness = self.predict_pressurethickness()

        t = 0
        while t < len(time):
            # Defines the axes.
            # For temperature, geostrophic wind, and pressure thickness plots.
            latitude, altitude = np.meshgrid(
                self.latitude_lines(), self.altitude_level()
            )
            # For recipitable water plot
            long = self.latitude_lines() * 2
            lat_pw, longitude = np.meshgrid(self.latitude_lines(), long)

            # Geostrophic wind contourf.
            # Geostrophic wind data.
            geostrophic_wind = (
                predict_u[0] * time[t]
            ) + predict_u[1]

            # Contouf plotting.
            cmap1 = plt.get_cmap("jet")
            min1 = predict_u[1].min()
            max1 = predict_u[1].max()
            if min1 > -100 and max1 < 100:
                level1 = np.linspace(-60, max1, 21)
            else:
                level1 = np.linspace(-70, 120, 21)
            v_g = ax1.contourf(
                latitude, altitude, geostrophic_wind, cmap=cmap1, levels=level1
            )

            # Checks for a colorbar.
            if t == 0:
                cb1 = fig.colorbar(v_g, ax=ax1)
                tick_locator = ticker.MaxNLocator(nbins=10)
                cb1.locator = tick_locator
                cb1.update_ticks()
                cb1.set_label("Velocity ($\\frac{m}{s}$)")
                cb1

            # Add SALT to the graph.
            ax1.set_xlabel("Latitude ($\phi$)")
            ax1.set_ylabel("Altitude (m)")
            ax1.set_title("Geostrophic Wind")

            # Temperature contouf.
            # Temperature data.
            temperature = (
                predict_t[0] * time[t]
            ) + predict_t[1]

            # Contouf plotting.
            cmap2 = plt.get_cmap("hot")
            min2 = temperature.min()
            max2 = temperature.max()
            level2 = np.linspace(min2, max2, 21)
            temp = ax2.contourf(
                latitude, altitude, temperature, cmap=cmap2, levels=level2
            )

            # Checks for a colorbar.
            if t == 0:
                cb2 = fig.colorbar(temp, ax=ax2)
                tick_locator = ticker.MaxNLocator(nbins=10)
                cb2.locator = tick_locator
                cb2.update_ticks()
                cb2.set_label("Temperature (K)")

            # Add SALT to the graph.
            ax2.set_xlabel("Latitude ($\phi$)")
            ax2.set_ylabel("Altitude (m)")
            ax2.set_title("Temperature")

            # Precipitable water contourf.
            # Precipitable water data.
            P_w = (
                predict_Pwv[0] * time[t]
            ) + predict_Pwv[1]
            precipitable_water = []
            n = 0
            while n < len(longitude):
                precipitable_water.append(list(P_w))

                n += 1
            precipitable_water = np.asarray(precipitable_water)

            # EckertIII projection details.
            ax3.set_global()
            ax3.coastlines()
            ax3.gridlines()

            # Contourf plotting.
            cmap3 = plt.get_cmap("seismic")
            min3 = predict_Pwv[1].min()
            level3 = np.linspace(min3, 100, 21)
            precipitable_watervapour = ax3.contourf(
                longitude,
                lat_pw,
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
            pressure_thickness = (
                predict_pthickness[0] * time[t]
            ) + predict_pthickness[1]

            # Define snow line, and plot.
            snow_line = np.zeros(len(pressure_thickness))
            snow_line += 5400
            ax4.plot(snow_line, self.latitude_lines(), "m--", label="Snow Line")

            # Scatter plotting.
            ax4.scatter(pressure_thickness, self.latitude_lines(), color="b")

            # Add SALT to the graph.
            ax4.set_xlabel("Pressure Thickness (m)")
            ax4.set_ylabel("Latitude ($\phi$)")
            ax4.set_title("Pressure Thickness (1000hPa - 500hPa)")
            ax4.set_xlim(5100, 6300)
            ax4.legend(loc=0)

            # Troposphere - Stratosphere Boundary Line
            # Determines the true boundary line.
            trop_strat_line = []
            for temp_ in np.transpose(temperature):
                n = 0
                while n < len(temp_):
                    y1 = temp_[n]
                    y2 = temp_[n + 1]

                    if (y2 - y1) > 0 and self.altitude_level()[n] >= 10000:
                        alt = self.altitude_level()[n]
                        trop_strat_line.append(alt)
                        n = len(temp_)

                    n += 1

            # Generates the average boundary line as a numpy array.
            trop_strat_line = np.asarray(trop_strat_line)
            trop_strat_line = np.mean(trop_strat_line) + np.zeros(len(trop_strat_line))

            # Plots the average boundary line on two contourfs.
            # Geostrophic wind contourf.
            ax1.plot(
                latitude[1],
                trop_strat_line,
                color="black",
                linestyle="dashed",
                label="Troposphere - Stratosphere Boundary Line",
            )

            # Temperature contourf.
            ax2.plot(
                latitude[1],
                trop_strat_line,
                color="black",
                linestyle="dashed",
                label="Troposphere - Stratosphere Boundary Line",
            )

            # Title of Simulation.
            day = int(np.floor((time[t] + 1)))
            if day < 10:
                day = "0" + str(day)
            else:
                day = str(day)

            fig.suptitle(
                "Motus Aeris @ AMSIMP ("
                + day
                + " of "
                + self.month.title()
                + ", "
                + str(self.date.year)
                + ")"
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

            # Footnote
            plt.figtext(
                0.99,
                0.01,
                "Note: Geostrophic balance does not hold near the equator.",
                horizontalalignment="right",
            )

            t += 1
