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

# -----------------------------------------------------------------------------------------#


class Weather(Wind, Water):
    """
	AMSIMP Weather Class - Explain code here.
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

        style.use("fivethirtyeight")

        # Define layout.
        gs = gridspec.GridSpec(3, 2)
        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5)
        fig.subplots_adjust(hspace=0.325, bottom=0.075)
        plt.ion()
        cmap = plt.get_cmap('jet')

        # Geostrophic Wind
        ax1 = plt.subplot(gs[0, 0])
        # Temperature
        ax2 = plt.subplot(gs[1:, 0])
        # Pressure
        ax3 = plt.subplot(gs[0, 1])
        # Precipitable Water
        ax4 = plt.subplot(gs[1, 1])
        # Pressure Thickness
        ax5 = plt.subplot(gs[2, 1])

        t = 0
        while t < len(time):
            # Temperature Contouf
            latitude, altitude = np.meshgrid(self.latitude_lines(), self.altitude_level())
            temperature = (self.predict_temperature()[0] * time[t]) + self.predict_temperature()[1]

            cmap = plt.get_cmap('jet')
            temp = ax2.contourf(latitude, altitude, temperature, cmap=cmap)
            
            colorbar = fig.colorbar(temp, ax=ax2, anchor=(0.0, 0.5), panchor=(1.0, 0.5))
            tick_locator = ticker.MaxNLocator(nbins=15)
            colorbar.locator = tick_locator
            colorbar.update_ticks()
            colorbar.set_label("Temperature (K)")
            ax2.set_xlabel("Latitude ($\phi$)")
            ax2.set_ylabel("Altitude (m)")
            ax2.set_title("Temperature")

            # Title of Weather Forecast.
            day = int(np.floor((time[t] + 1)))
            if day < 10:
                day = '0' + str(day)
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

            plt.show()
            plt.pause(0.01)
            if t < (len(time) - 1):
                ax2.clear()
                colorbar.remove()

            t += 1
