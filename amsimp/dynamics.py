"""
AMSIMP Dynamics Class. For information about this class is described below.
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
from amsimp.backend import Backend

# -----------------------------------------------------------------------------------------#


class Dynamics(Backend):
    """
	AMSIMP Dynamics Class - This class inherits the AMSIMP Backend Class.
	Calculates the Zonal and Meridional Wind. It will also create the vectors needed for projection
	onto the simulated globe.
	"""

    def zonal_wind(self):
        """
        Generates a numpy of the quasi-geostrophic approximation of geostrophic wind / velocity.
        The Rossby number at synoptic scales is small, which implies that the
        velocities are nearly geostrophic.

        Equation: u = u_bar + u' (u_bar = β / (k ^ 2 + l ^ 2)) (u′≈ − g_0/f * 1 / r * ∂/dphi(Φ))
        """
        zonal_wind = []
        latitude = np.radians(self.latitude_lines())

        a = 6378137
        b = 6356752.3142
        epsilon = np.sqrt((a ** 2) - (b ** 2)) / a
        N_lat = a / np.sqrt(1 - epsilon * np.sin(latitude) ** 2)
        R = []
        for z in self.altitude_level():
            var = (N_lat + z) * np.cos(latitude)
            var = var.tolist()
            R.append(var)
        R = np.asarray(R)

        derivative_geopotential = []
        count = 0
        for var in R:
            z = self.altitude_level()
            z = z.tolist()
            dev_g = (self.Upomega ** 2) * z[count] * var * np.sin(2 * latitude)
            dev_g = dev_g.tolist()
            derivative_geopotential.append(dev_g)
            count += 1
        derivative_geopotential = np.asarray(derivative_geopotential)

        derivative_geopotential_height = derivative_geopotential / self.g

        for geo in derivative_geopotential_height:
            u = -(self.g / self.coriolis_force()) * (1 / self.a) * geo
            u = u.tolist()
            zonal_wind.append(u)

        zonal_wind = np.asarray(zonal_wind)

        return zonal_wind

    def meridional_wind(self):
        """
        Similar to zonal velocity, this generates a numpy of the quasi-geostrophic
        approximation of meridional wind / velocity.

        Equation: v = v' (v′ = 0)
        """
        meridional_wind = self.zonal_wind() * 0

        return meridional_wind

    # -----------------------------------------------------------------------------------------#

    def simulate(self):
        """
		Plots the vector field, vector_creation() (of Zonal and Meridional Winds),
		onto a globe.
		"""
        longitude = self.longitude_lines()
        latitude = self.latitude_lines()

        u = self.zonal_wind()[-1]
        v = self.meridional_wind()[-1]

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
                central_latitude=0, central_longitude=-0, satellite_height=10000000.0
            )
        )

        if self.benchmark:
            plt.ion()

        ax.add_feature(cartopy.feature.OCEAN, zorder=0)
        ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor="black")

        ax.set_global()
        ax.gridlines()
        ax.coastlines(resolution="110m")

        geostrophic_wind = np.sqrt((u ** 2) + (v ** 2))
        norm = PowerNorm(vmin=2.5, vmax=geostrophic_wind.max(), gamma=10.0)
        norm.autoscale(geostrophic_wind)
        colormap = cm.gist_rainbow

        ax.quiver(y, x, u, v, color=colormap(norm(geostrophic_wind)))

        cax, _ = colorbar.make_axes(plt.gca())
        colour_bar = colorbar.ColorbarBase(
            cax,
            cmap=cm.gist_rainbow,
            norm=norm,
            extend="min",
            boundaries=np.linspace(2.5, geostrophic_wind.max(), 20),
        )
        colour_bar.set_label("Geostrophic Wind (m/s)")

        plt.show()
