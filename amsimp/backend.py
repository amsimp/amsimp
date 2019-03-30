#-----------------------------------------------------------------------------------------#

#Importing Dependencies
import math
from scipy.constants import gas_constant
from astropy import constants as const
import numpy as np

#-----------------------------------------------------------------------------------------#

class Backend:
	"""
	AMSIMP Backend Module - Contains / calculates all the variables needed to utilize the 
	Primitive Equations. 
	"""

	#Predefined Constants.
	#Angular rotation rate of Earth.
	Upomega = (2 * math.pi) / 24
	#Ideal Gas Constant
	R = gas_constant
	#Radius of the Earth.
	a = const.R_earth
	a = a.value
	#Big G.
	G = const.G
	G = G.value
	#Mass of the Earth.
	m = const.M_earth
	m = m.value

	def __init__(self, detail_level = 3):
		"""
		Numerical value for the level of detail that will be used in the mathematical 
		calculations for the computation detail. This value is between 0 and 4. 
		""" 
		self.detail_level = detail_level

		if not isinstance(self.detail_level, int):
			raise Exception("detail_level must be an integer. The value of detail_level was: {}".format(self.detail_level))

		if self.detail_level >= 5 or self.detail_level < 0:
			raise Exception("detail_level must be a positive integer between 0 and 4. The value of detail_level was: {}".format(self.detail_level))

		self.detail_level = 10 ** self.detail_level

	def longitude_lines(self):
		#Generates a list of longitude lines.
		num_of_longitude_lines = 360

		longitude_lines = [i for i in np.arange(-180, 181, (360 / self.detail_level)) if i >= -180 and i <= 180 and i !=0]

		longitude_lines = np.asarray(longitude_lines)
		return longitude_lines

	def latitude_lines(self):
		#Generates a list of latitude lines.
		num_of_latitude_lines = 180

		latitude_lines = [i for i in np.arange(-90, 91, (180 / self.detail_level)) if i >= -90 and i <= 90 and i != 0]

		latitude_lines = np.asarray(latitude_lines)
		return latitude_lines

	def altitude_level(self):
		#Setting the maximum height above sea level (in metres).
		max_height = 50000
		mim_height_detail = max_height / 5

		"""
		Generates a list which will be used in calculations relating to height above 
		sea level (array in metres). 
		"""
		
		altitude_level = list(n for n in range(0, max_height + 1) if n % (mim_height_detail / self.detail_level) == 0)

		altitude_level = np.asarray(altitude_level)
		return altitude_level

	def coriolis_force(self):
		"""
		Generates a list of Coriolis force based on the relevant mathematical formula. 
		As such, it utilizes the constant, angular rotation of the Earth, and the latitude.
		"""
		coriolis_force = []

		for latitude in self.latitude_lines():
			latitude = 2 * self.Upomega * math.sin(math.radians(latitude))
			coriolis_force.append(latitude)

		coriolis_force = np.asarray(coriolis_force)
		return coriolis_force

	def geopotential(self):
		"""
		Generates a list of geopotential based on the relevant mathematical formula.
		As such, it utilizes gravitational constant, the mass and radius of the Earth,
		and height in metres above sea level.
		"""
		geopotential = []
		
		for altitude in self.altitude_level():
			altitude = (self.G * self.m * ((1 / self.a) - (1 / (self.a + altitude))))
			geopotential.append(altitude)

		geopotential = np.asarray(geopotential)
		return geopotential

	def pressure(self):
		"""
		Generates a list of atmospheric pressure by utilizing the Barometric Formula
		for Pressure. As such, it was generated from a height above sea level list.
		Such a list was created in the function, altitude_level. 
		"""
		pressure = []

		for altitude in self.altitude_level():
			altitude /= 1000
			altitude = -832.6777 + (101323.6 + 832.6777) / (1 + (altitude / 6.527821) ** 2.313703)
			pressure.append(altitude)

		pressure = np.asarray(pressure)
		return pressure

#-----------------------------------------------------------------------------------------#
	
	def temperature(self):
		"""
		The equation for temperature, T, assuming temperature is hydrostatic: T = -p/R * ∂Φ/∂p.     
		"""
		return false

	def zonal_velocity(self):
		"""
		Generates a list of the quasi-geostrophic approximation of zonal velocity.
		
		The Rossby number at synoptic scales is small, which implies that the
		velocities are nearly geostrophic.

		Equation: u′≈ −1/f * ∂/dy(Φ)
		"""
		zonal_velocity = []

		for force in self.coriolis_force():
			force = ((-1 / force) * 9.7221235)
			zonal_velocity.append(force)

		zonal_velocity = np.asarray(zonal_velocity)	
		return zonal_velocity

	def meridional_velocity(self):
		"""
		Similar to zonal velocity, this generates a list of the quasi-geostrophic
		approximation of meridional velocity.

		Equation: v′≈ 1/f * ∂/dx(Φ)
		"""
		meridional_velocity = []

		for force in self.coriolis_force():
			force = ((1 / force) * 9.7221235)
			meridional_velocity.append(force)

		meridional_velocity = np.asarray(meridional_velocity)
		return meridional_velocity

	def vertical_velocity(self):
		"""
		Generates a list of vertical velocity (omega) by utilizing the derivative of 
		the pressure equation (pressure() function).

		Since pressure decreases upward, a negative omega means rising motion, while
		a positive omega means subsiding motion.
		"""
		vertical_velocity = []

		for pressure in self.pressure():
			pressure = -832.6777 + 102156.2777 / (1.49196723444642e-9 * pressure ** 2.313703 + 1)
			vertical_velocity.append(pressure)

		vertical_velocity = np.asarray(vertical_velocity)	
		return vertical_velocity

#-----------------------------------------------------------------------------------------#

	def absolute_vorticity(self):
		"""
		The vorticity is the curl of the velocity. It has components in x, y and z.
		But at synoptic scales, the vertical component is by far the most important.

		Absolute vorticity is the sum of the relative vorticity, ζ, and the planetary 
		vorticity, f.
		"""
		absolute_vorticity = []

		for force in self.coriolis_force():
			force = -1 * force
			absolute_vorticity.append(force)

		absolute_vorticity = np.asarray(absolute_vorticity)
		return absolute_vorticity