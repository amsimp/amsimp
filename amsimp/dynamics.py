#-----------------------------------------------------------------------------------------#

#Importing Dependencies
import math
from amsimp.backend import Backend
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import statistics as stats

#-----------------------------------------------------------------------------------------#

class Dynamics(Backend):
	"""
	AMSIMP Dynamics Class - This class inherits the AMSIMP Backend Class.
	Calculates the Zonal and Meridional Wind, which is defined as the derivative of zonal
	and meridional velocity respectfully. It will also create the vectors needed for projection
	onto the simulated globe.
	"""

	def zonal_wind(self):
		"""
		Zonal wind as mentioned previously is the derivative of zonal velocity. This
		method generates a numpy array of zonal wind.

		Equation: vector{u} = du/dt 
		"""
		zonal_wind = []

		derivative_geopotential = 3.98712e14 / ((self.altitude_level() + 6378000) ** 2)

		for geopotential in derivative_geopotential:
			vector_u = geopotential / (self.coriolis_force() ** 2)
			vector_u = vector_u.tolist()
			zonal_wind.append(vector_u)

		zonal_wind = np.asarray(zonal_wind)
		return zonal_wind

	def meridional_wind(self):
		"""
		Similar to zonal velocity, this generates a numpy array of zonal wind.

		Equation: vector{u} = du/dt 
		"""
		meridional_wind = []

		derivative_geopotential = 3.98712e14 / ((self.altitude_level() + 6378000) ** 2)

		for geopotential in derivative_geopotential:
			vector_v = (- geopotential / (self.coriolis_force() ** 2))
			vector_v = vector_v.tolist()
			meridional_wind.append(vector_v)

		meridional_wind = np.asarray(meridional_wind)
		return meridional_wind

#-----------------------------------------------------------------------------------------#

	def simulate(self):
		"""
		Plots the vector field, vector_creation() (of Zonal and Meridional Winds),
		onto a globe. Defaults to a view of the Earth at a longitude and latitude
		of 0.
		"""
		longitude = self.longitude_lines()
		latitude = self.latitude_lines()

		points = ccrs.Orthographic().transform_points(ccrs.Geodetic(), longitude, latitude)

		x, y = np.meshgrid(points[:, 0], points[:, 1])
		
		zonal_wind = self.zonal_wind()[0]
		meridional_wind = self.meridional_wind()[0]

		if self.detail_level != 100:
			u_split = np.split(zonal_wind, 2)
			v_split = np.split(meridional_wind, 2)
		else:
			zonal_wind = zonal_wind.tolist()
			meridional_wind = meridional_wind.tolist()
			x = x.tolist()
			y = y.tolist()
			u_median = stats.median(zonal_wind)
			v_median = stats.median(meridional_wind)
			x_median = stats.median(x)
			y_median = stats.median(y)
			zonal_wind.remove(u_median)
			meridional_wind.remove(v_median)
			x.remove(x_median)
			y.remove(y_median)
			x = np.asarray(x)
			y = np.asarray(x)
			zonal_wind = np.asarray(zonal_wind)
			meridional_wind = np.asarray(meridional_wind)
			u_split = np.split(zonal_wind, 2)
			v_split = np.split(meridional_wind, 2)

		u_northern_hemisphere = u_split[0]
		v_northern_hemisphere = v_split[0]
		u_northern_hemisphere *= -1
		v_northern_hemisphere *= -1

		u_southern_hemisphere = u_split[1]
		v_southern_hemisphere = v_split[1]

		u = np.concatenate([u_northern_hemisphere, u_southern_hemisphere])
		v = np.concatenate([v_northern_hemisphere, v_southern_hemisphere])

		u_norm = u / np.sqrt(u ** 2.0 + v ** 2.0)
		v_norm = v / np.sqrt(u ** 2.0 + v ** 2.0)

		ax = plt.axes(projection = ccrs.Orthographic())

		ax.add_feature(cartopy.feature.OCEAN, zorder=0)
		ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black')

		ax.set_global()
		ax.gridlines()
		ax.stock_img()

		ax.quiver(y, x, v_norm, u_norm, np.arctan2(v, u), color = 'r')

		plt.show()			