#-----------------------------------------------------------------------------------------#

#Importing Dependencies
import math
from amsimp.backend import Backend
import numpy as np

#-----------------------------------------------------------------------------------------#

class Dynamics(Backend):
	"""
	AMSIMP Dynamics Class - This class inherits the AMSIMP Backend Class.
	Calculates the Zonal and Meridional Wind, which is defined as the derivative of zonal
	and meridional velocity respectfully. It will also create the vectors, and vector 
	angles needed for projection on the simulated globe.
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
			meridional_wind = np.asarray(meridional_wind)

		meridional_wind = np.asarray(meridional_wind)
		return meridional_wind