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
			vector_v = (- geopotential / (self.coriolis_force() ** 2))
			vector_v = vector_v.tolist()
			meridional_wind.append(vector_v)

		meridional_wind = np.asarray(meridional_wind)
		return meridional_wind

#-----------------------------------------------------------------------------------------#

	def simulate(self):
		"""
		Plots the vector field, vector_creation() (of Zonal and Meridional Winds),
		onto a globe. 
		"""
		latitude = self.latitude_lines()
		longitude = self.longitude_lines()
		
		zonal_wind = self.zonal_wind()[0]
		meridional_wind = self.meridional_wind()[0]