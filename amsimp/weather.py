"""
AMSIMP Weather Class. For information about this class is described below.
"""

# -----------------------------------------------------------------------------------------#

# Importing Dependencies
from amsimp.wind import Wind
from amsimp.water import Water
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.gridspec as gridspec
from matplotlib import ticker
import numpy as np
import cartopy.crs as ccrs

# -----------------------------------------------------------------------------------------#


class Weather(Wind, Water):
    """
	AMSIMP Weather Class - Also, known as Tempestas Praenuntientur @ AMSIMP.
	"""

    def __init__(self, detail_level):
        """
        Explain code here.
        """

        super().__init__(detail_level)

        self.future = True

    def predict_temperature(self):
        """
        Explain code here.
        """
        future_temperature = self.temperature()
        self.future = False
        init_temperature = self.temperature()
        self.future = True

        n = self.number_of_days - 1

        gradient = (future_temperature - init_temperature) / n

        return gradient, init_temperature

    def predict_pressure(self):
        """
        Explain code here.
        """
        future_pressure = self.pressure()
        self.future = False
        init_pressure = self.pressure()
        self.future = True

        n = self.number_of_days - 1

        gradient = (future_pressure - init_pressure) / n

        return gradient, init_pressure

    def predict_pressurethickness(self):
        """
        Explain code here.
        """
        future_pressurethickness = self.pressure_thickness()
        self.future = False
        init_pressurethickness = self.pressure_thickness()
        self.future = True

        n = self.number_of_days - 1

        gradient = (future_pressurethickness - init_pressurethickness) / n

        return gradient, init_pressurethickness

    def predict_geostrophicwind(self):
        """
        Explain code here.
        """
        future_geostrophicwind = self.geostrophic_wind()
        self.future = False
        init_geostrophicwind = self.geostrophic_wind()
        self.future = True

        n = self.number_of_days - 1

        gradient = (future_geostrophicwind - init_geostrophicwind) / n

        return gradient, init_geostrophicwind

    def predict_precipitablewater(self):
        """
        Explain code here.
        """
        future_precipitablewater = self.precipitable_water()
        self.future = False
        init_precipitablewater = self.precipitable_water()
        self.future = True

        n = self.number_of_days - 1

        gradient = (future_precipitablewater - init_precipitablewater) / n

        return gradient, init_precipitablewater

    def simulate(self):
        """
        Explain code here.
        """
        # Time (Unit: day).
        time = np.linspace(0, (self.number_of_days - 1), (self.number_of_days * 2))

        # Style of graph.
        style.use("fivethirtyeight")

        # Define layout.
        gs = gridspec.GridSpec(2, 2)
        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5)
        fig.subplots_adjust(hspace=0.340, bottom=0.105, top=0.905)
        plt.ion()

        # Geostrophic Wind
        ax1 = plt.subplot(gs[0, 0])
        # Temperature
        ax2 = plt.subplot(gs[1, 0])
        # Precipitable Water
        ax3 = plt.subplot(gs[0, 1], projection=ccrs.EckertIII())
        # Pressure Thickness
        ax4 = plt.subplot(gs[1, 1])

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
                self.predict_geostrophicwind()[0] * time[t]
            ) + self.predict_geostrophicwind()[1]

            # Contouf plotting.
            cmap1 = plt.get_cmap("jet")
            min1 = self.predict_geostrophicwind()[1].min()
            max1 = self.predict_geostrophicwind()[1].max()
            if min1 > -100 and max1 < 100:
                level1 = np.linspace(min1, 60, 21)
            else:
                level1 = np.linspace(-120, 70, 21)
            v_g = ax1.contourf(
                latitude, altitude, geostrophic_wind, cmap=cmap1, levels=level1
            )

            # Checks for a colorbar.
            try:
                cb1
            except NameError:
                cb1 = fig.colorbar(v_g, ax=ax1)
                tick_locator = ticker.MaxNLocator(nbins=10)
                cb1.locator = tick_locator
                cb1.update_ticks()
                cb1.set_label("Velocity ($\\frac{m}{s}$)")

            # Add SALT to the graph.
            ax1.set_xlabel("Latitude ($\phi$)")
            ax1.set_ylabel("Altitude (m)")
            ax1.set_title("Geostrophic Wind")

            # Temperature contouf.
            # Temperature data.
            temperature = (
                self.predict_temperature()[0] * time[t]
            ) + self.predict_temperature()[1]

            # Contouf plotting.
            cmap2 = plt.get_cmap("hot")
            min2 = temperature.min()
            max2 = temperature.max()
            level2 = np.linspace(min2, max2, 21)
            temp = ax2.contourf(
                latitude, altitude, temperature, cmap=cmap2, levels=level2
            )

            # Checks for a colorbar.
            try:
                cb2
            except NameError:
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
                self.predict_precipitablewater()[0] * time[t]
            ) + self.predict_precipitablewater()[1]
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
            min3 = self.predict_precipitablewater()[1].min()
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
            try:
                cb3
            except NameError:
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
                self.predict_pressurethickness()[0] * time[t]
            ) + self.predict_pressurethickness()[1]

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

            # Title of Weather Forecast.
            day = int(np.floor((time[t] + 1)))
            if day < 10:
                day = "0" + str(day)
            else:
                day = str(day)

            fig.suptitle(
                "Tempestas Praenuntientur @ AMSIMP ("
                + day
                + " of "
                + self.month.title()
                + ", "
                + str(self.date.year)
                + ")"
            )

            # Displaying weather forecast.
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
