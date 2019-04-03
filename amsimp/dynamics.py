#-----------------------------------------------------------------------------------------#

#Importing Dependencies
import math
from amsimp.backend import Backend
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs

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
			vector_v = -1 * (geopotential / (self.coriolis_force() ** 2))
			vector_v = vector_v.tolist()
			meridional_wind.append(vector_v)

		meridional_wind = np.asarray(meridional_wind)
		return meridional_wind

	def vector_creation(self):
		"""
		Create the vectors needed for the projection onto the simulated globe.
		"""
		vectors = []

		if len(self.zonal_wind()) == len(self.meridional_wind()):
			n = 0
			while n < len(self.zonal_wind()):
				vector_u = np.asarray(self.zonal_wind()[n])
				vector_v = np.asarray(self.meridional_wind()[n])

				vector = np.sqrt((vector_u ** 2) + (vector_v ** 2))

				vector = vector.tolist()
				vectors.append(vector)

				n += 1

		vectors = np.asarray(vectors)
		return vectors

#-----------------------------------------------------------------------------------------#

	def simulate(self):
		"""
		Plots the vector field, vector_creation() (of Zonal and Meridional Winds),
		onto a globe. 
		"""
		latitude = self.latitude_lines()
		longitude = self.longitude_lines()

		x, y = np.meshgrid(latitude, longitude)

		u = self.zonal_wind()[0]
		v = self.meridional_wind()[0]

		ax = plt.axes(projection = ccrs.Orthographic(0, 0))

		ax.add_feature(cartopy.feature.OCEAN, zorder = 0)
		ax.add_feature(cartopy.feature.LAND, zorder = 0, edgecolor = 'black')

		ax.set_global()
		ax.gridlines()

		ax.quiver(x, y, u, v)

		plt.title('Vector Field of Zonal and Meridional Winds (At Sea Level)')

		plt.show() 