"""
AMSIMP Wind Class. For information about this class is described below.
"""

# -----------------------------------------------------------------------------------------#

# Importing Dependencies
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from amsimp.backend import Backend

# -----------------------------------------------------------------------------------------#


class Wind(Backend):
    """
	AMSIMP Wind Class - Explain code here.
	"""

    def geostrophic_wind(self):
        """
        Explain code here.
        """
        # Ensures that the detail_level must be higher than 2 in order to utilise this method.
        if self.detail_level < (5 ** (3 - 1)):
            raise Exception(
                "detail_level must be greater than 2 in order to utilise this method."
            )

        # Distance of one degree of latitude (e.g. 0N - 1N/1S), measured in metres.
        lat_d = (2 * np.pi * self.a) / 360
        # Distance between latitude lines in the class method, Backend.latitude_lines().
        delta_y = (self.latitude_lines()[-1] - self.latitude_lines()[-2]) * lat_d

        # Gradient of geopotential height over latitudinal distance.
        gradient_geopotentialheight = []
        for Z in self.geopotential_height():
            southpole, northpole = np.split(Z, 2)

            southpole = np.flip(southpole)
            southpole = np.gradient(southpole)
            southpole = np.flip(southpole)

            northpole = np.gradient(northpole)

            delta_geoheight = np.concatenate((southpole, northpole))

            grad_geoheight = delta_geoheight / delta_y
            grad_geoheight = list(grad_geoheight)
            gradient_geopotentialheight.append(grad_geoheight)
        gradient_geopotentialheight = np.asarray(gradient_geopotentialheight)

        # Geostrophic wind calculation.
        geostrophic_wind = (
            -(self.gravitational_acceleration() / self.coriolis_force())
            * gradient_geopotentialheight
        )

        return geostrophic_wind

    # -----------------------------------------------------------------------------------------#

    def contourf(self):
        """
		Explain code here.
		"""
        # Defines the axes, and the data.
        latitude, altitude = np.meshgrid(self.latitude_lines(), self.altitude_level())
        geostrophic_wind = self.geostrophic_wind()

        # Specifies the contour levels
        minimum = geostrophic_wind.min()
        maximum = geostrophic_wind.max()
        if minimum > -100 and maximum < 100:
            levels = np.linspace(minimum, maximum, 21)
        else:
            levels = np.linspace(-120, 70, 21)

        # Contourf plotting.
        plt.contourf(latitude, altitude, geostrophic_wind, levels=levels)

        plt.set_cmap("jet")

        # Adds SALT to the graph.
        plt.xlabel("Latitude ($\phi$)")
        plt.ylabel("Altitude (m)")
        plt.title("Geostrophic Wind in the Month of " + self.month.title())

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

        # Defines the boundary line between the troposphere and the stratosphere.
        temperature = np.transpose(self.temperature())

        trop_strat_line = []
        for temp in temperature:
            n = 0
            while n < (len(temp)):
                y1 = temp[n]
                y2 = temp[n + 1]

                if (y2 - y1) > 0 and self.altitude_level()[n] >= 10000:
                    alt = self.altitude_level()[n]
                    trop_strat_line.append(alt)
                    n = len(temp)

                n += 1
        trop_strat_line = np.asarray(trop_strat_line)

        # Average boundary line between the troposphere and the stratosphere.
        avg_tropstratline = np.mean(trop_strat_line) + np.zeros(len(trop_strat_line))

        # Plot average boundary line on the contourf plot.
        plt.plot(
            latitude[1],
            avg_tropstratline,
            color="black",
            linestyle="dashed",
            label="Troposphere - Stratosphere Boundary Line",
        )
        plt.legend(loc=0)

        plt.show()
