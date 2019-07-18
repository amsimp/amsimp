"""
AMSIMP Wind Class. For information about this class is described below.
"""

# -----------------------------------------------------------------------------------------#

# Importing Dependencies
import cartopy
import cartopy.crs as ccrs
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import pandas as pd
from amsimp.backend import Backend
from amsimp.derivatives import dev_geopotentialheight, verticalvelocity_component

# -----------------------------------------------------------------------------------------#


class Wind(Backend):
    """
	AMSIMP Wind Class - This class inherits the AMSIMP Backend Class.
	Calculates the Zonal and Meridional Wind. It will also create the vectors needed for projection
	onto the simulated globe.
	"""

    def zonal_velocity(self):
        """
        Explain code here.
        """
        zonal_velocity = []
        
        return zonal_velocity

    def meridional_velocity(self):
        """
        Similar to zonal velocity, this generates a numpy of the quasi-geostrophic
        approximation of meridional wind / velocity.

        Equation: v = v' (v′ = 0)
        """
        meridional_velocity = self.zonal_velocity() * 0

        return meridional_velocity

    def vertical_velocity(self):
        """
		Generates a numpy of vertical velocity (omega), under the  f-plane approximation,
        by utilizing the derivative of the pressure equation (pressure() function).

		Since pressure decreases upward, a negative omega means rising motion, while
		a positive omega means subsiding motion. 
		"""
        vertical_velocity = -self.density() * self.gravitational_acceleration()

        return vertical_velocity

    # -----------------------------------------------------------------------------------------#

    def simulate(self):
        """
		Plots the vector field, vector_creation() (of Zonal and Meridional Winds),
		onto a globe.
		"""
        longitude = self.longitude_lines()
        latitude = self.latitude_lines()

        u = self.zonal_velocity()[0]
        v = self.meridional_velocity()[0]

        if self.detail_level == (5 ** (1 - 1)):
            skip = 1
        elif self.detail_level == (5 ** (2 - 1)):
            skip = 1
        elif self.detail_level == (5 ** (3 - 1)):
            skip = 1
        elif self.detail_level == (5 ** (4 - 1)):
            skip = 3
        elif self.detail_level == (5 ** (5 - 1)):
            skip = 12

        latitude = latitude[::skip]
        longitude = longitude[::skip]

        u = u[::skip]
        v = v[::skip]

        points = ccrs.NearsidePerspective().transform_points(
            ccrs.Geodetic(), longitude, latitude
        )

        x, y = np.meshgrid(points[:, 0], points[:, 1])

        plt.figure()

        ax = plt.axes(
            projection=ccrs.NearsidePerspective(
                central_latitude=45, central_longitude=-0, satellite_height=10000000.0
            )
        )

        if self.benchmark:
            plt.ion()

        ax.add_feature(cartopy.feature.OCEAN, zorder=0)
        ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor="black")

        ax.set_global()
        ax.gridlines()
        ax.coastlines(resolution="110m")
        ax.stock_img()

        u_norm = u / np.sqrt(u ** 2.0 + v ** 2.0)
        v_norm = v / np.sqrt(u ** 2.0 + v ** 2.0)

        geostrophic_wind = np.sqrt((u ** 2) + (v ** 2))
        norm = PowerNorm(gamma=0.5)
        norm.autoscale(geostrophic_wind)
        colormap = cm.gist_rainbow

        ax.quiver(y, x, u_norm, v_norm, color=colormap(norm(geostrophic_wind)))

        cax, _ = colorbar.make_axes(plt.gca())
        colour_bar = colorbar.ColorbarBase(
            cax,
            cmap=cm.gist_rainbow,
            norm=norm,
            extend="min",
            boundaries=np.linspace(geostrophic_wind.min(), geostrophic_wind.max(), 1000),
        )
        colour_bar.set_label("Geostrophic Wind (m/s)")

        plt.show()
