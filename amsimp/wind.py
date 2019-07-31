"""
AMSIMP Wind Class. For information about this class is described below.
"""

# -----------------------------------------------------------------------------------------#

# Importing Dependencies
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from amsimp.backend import Backend

# -----------------------------------------------------------------------------------------#


class Wind(Backend):
    """
    AMSIMP Wind Class - This class is concerned with calculating numerical
    values for wind, specifically geostrophic wind, in the troposphere and the
    stratosphere. It also contain two methods for the visualisation of these
    numerical values.

    Below is a list of the methods included within this class, with a short
    description of their intended purpose. Please see the relevant class methods
    for more information.

    geostrophic_wind ~ outputs geostrophic wind values.
    wind_contourf ~ generates a geostrophic wind contour plot.
    globe ~ generates a geostrophic wind contour plot, adds wind vectors to
    that said plot, and overlays both on a Nearside Projection of the Earth.
	"""

    def geostrophic_wind(self):
        """
        This method outputs geostrophic wind values. Geostrophic wind is a
        theoretical wind that is a result of a perfect balance between the
        Coriolis force and the pressure gradient force. This balance is known as
        geostrophic balance. Note: Geostrophic balance does not hold near the
        equator.
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
            delta_geoheight = np.gradient(Z)

            grad_geoheight = delta_geoheight / delta_y
            grad_geoheight = list(grad_geoheight)
            gradient_geopotentialheight.append(grad_geoheight)
        gradient_geopotentialheight = np.asarray(gradient_geopotentialheight)

        # Geostrophic wind calculation.
        geostrophic_wind = (
            -(self.gravitational_acceleration() / self.coriolis_parameter())
            * gradient_geopotentialheight
        )

        return geostrophic_wind

    # -----------------------------------------------------------------------------------------#

    def wind_contourf(self):
        """
        Generates a geostrophic wind contour plot, with the axes being
        latitude, and longitude. For the raw data, please use the
        amsimp.Wind.geostrophic_wind() method.
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
            levels = np.linspace(-70, 120, 21)

        # Contour plotting.
        plt.contourf(latitude, altitude, geostrophic_wind, levels=levels)

        plt.set_cmap("jet")

        # Adds SALT to the graph.
        if self.future:
            month = self.next_month.title()
        else:
            month = self.month.title()

        plt.xlabel("Latitude ($\phi$)")
        plt.ylabel("Altitude (m)")
        plt.title("Geostrophic Wind in the Month of " + month)

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
        avg_tropstratline = self.troposphere_boundaryline()

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

    def globe(self, central_lat=53.1424, central_long=-7.6921):
        """
        Similiar to amsimp.Wind.wind_contourf(), however, this particular method
        adds wind vectors to the contour plot. It also overlays both of these
        elements onto a Nearside Perspective projection of the Earth.

        By default, the perspective view is looking directly down at the city
        of Dublin in the country of Ireland.

        Known bug(s):
        When setting the parameter, central_lat, to a value greater than 45
        in certain months (August is known thus far) results in the contour
        plot having zero variation in colour.

        Note:
        The NumPy method, seterr, is used to suppress a weird RunTime warning
        error that occurs on certain detail_level values in certain months.
        """
        # Ensure central_lat is between -90 and 90.
        if central_lat < -90 or central_lat > 90:
            raise Exception(
                "central_lat must be a real number between -90 and 90. The value of detail_level was: {}".format(
                    self.detail_level
                )
            )

        # Ensure central_lat is between -180 and 180.
        if central_lat < -180 or central_lat > 180:
            raise Exception(
                "central_long must be a real number between -180 and 180. The value of detail_level was: {}".format(
                    self.detail_level
                )
            )

        # Ignore NumPy errors.
        np.seterr(all="ignore")

        # Define the axes, and the data.
        long = self.latitude_lines() * 2
        latitude, longitude = np.meshgrid(self.latitude_lines(), long)
        geo_wind = self.geostrophic_wind()[0]

        geostrophic_wind = []
        n = 0
        while n < len(long):
            geostrophic_wind.append(list(geo_wind))

            n += 1
        geostrophic_wind = np.asarray(geostrophic_wind)

        ax = plt.axes(
            projection=ccrs.NearsidePerspective(
                central_longitude=central_long, central_latitude=central_lat
            )
        )

        # Add latitudinal and longitudinal grid lines, as well as, coastlines to the globe.
        ax.set_global()
        ax.coastlines()
        ax.gridlines()

        # Contour plotting.
        minimum = geostrophic_wind.min()
        maximum = np.percentile(geostrophic_wind, 97)
        levels = np.linspace(minimum, maximum, 21)

        contourf = plt.contourf(
            longitude,
            latitude,
            geostrophic_wind,
            transform=ccrs.PlateCarree(),
            levels=levels,
        )

        # Wind vector plotting.
        if self.detail_level == (5 ** (4 - 1)):
            skip = 6
        elif self.detail_level == (5 ** (5 - 1)):
            skip = 26
        else:
            skip = 1

        long = longitude[::skip]
        lat = latitude[::skip]

        u = geostrophic_wind[::skip]
        v = u * 0

        u_norm = u / np.sqrt(u ** 2 + v ** 2)

        plt.quiver(long, lat, u_norm, v, transform=ccrs.PlateCarree())

        # Add SALT to the graph.
        if self.future:
            month = self.next_month.title()
        else:
            month = self.month.title()

        plt.xlabel("Latitude ($\phi$)")
        plt.ylabel("Longitude ($\lambda$)")
        plt.title("Geostrophic Wind in the Month of " + month + " (Altitude: 0 metres)")

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
