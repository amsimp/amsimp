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
	a_in_km = a / 1000
	#Big G.
	G = const.G
	G = G.value
	#Mass of the Earth.
	m = const.M_earth
	m = m.value

	def __init__(self, detail_level = 3):
		"""
		Numerical value for the level of detail that will be used in the mathematical 
		calculations for the computation detail. This value is between 0 and 5. 
		""" 
		self.detail_level = detail_level
		self.detail_level = 10 ** self.detail_level

		if not isinstance(self.detail_level, int):
			raise ValueError("detail_level must be an integer.")
		if self.detail_level <= 0 and self.detail_level > 5:
			raise ValueError("detail_level must be a positive integer between 0 and 5.")

	def longitude_lines(self):
		#Generates a list of longitude lines.
		num_of_longitude_lines = 360

		longitude_lines = [i for i in np.arange(-180, 181, (360 / self.detail_level)) if i >= -180 and i <= 180]

		return longitude_lines

	def latitude_lines(self):
		#Generates a list of latitude lines.
		num_of_latitude_lines = 180

		latitude_lines = [i for i in np.arange(-90, 91, (180 / self.detail_level)) if i >= -90 and i <= 90]

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

		return coriolis_force

	def geopotential(self):
		"""
		Generates a list of geopotential based on the relevant equation.
		As such, it utilizes gravitational constant, the mass and radius of the Earth,
		and height in metres above sea level.
		
		Equation: 
		Integral_0^z{g dz}
		"""
		geopotential = []
		
		for altitude in self.altitude_level():
			altitude = ()
			geopotential.append(altitude)

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

		return pressure

	def temperature(self):
		"""
		Equation for Temperature: T = 
		"""
		return false

	def zonal_velocity(self):
		return false

	def meridional_velocity(self):
		return false

	def vertical_velocity(self):
		return false