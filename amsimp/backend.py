#-----------------------------------------------------------------------------------------#

#Importing Dependencies
import math
from scipy.constants import gas_constant
from astropy import constants as const
import sympy as cal

#-----------------------------------------------------------------------------------------#

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

#-----------------------------------------------------------------------------------------#

#Zonal velocity calculations.
#Equation to calculate zonal velocity based on the latitude from the equator.
def u(phi):
	return a_in_km * Upomega * ((math.sin(math.radians(phi)) ** 2 ) / math.cos(math.radians(phi)))

#Generates a latitude list (Negatiive numbers describe the Southern Hemisphere. 
#It increments by one degree. Excludes the poles!
latitude = list(range(-89, 90))

#Creates a zonal velocity list based on the aforementioned equation.
zonal_velocity = []
for i in latitude:
	i = u(i)
	zonal_velocity.append(i)

#Creates a dictionary that combines the zonal velocity, and latitude lists.
zonalvelocity_latitude = dict(zip(latitude, zonal_velocity))

#THESE CALCULATIONS COULD BE WRONG!!!

#-----------------------------------------------------------------------------------------#

#Meridional velocity calculations.
#Atmospheric pressure calculations.
#Equation to calculate atmospheric pressure based on height above sea level.
def p(z):
	z /= 1000
	return -832.6777 + (101323.6 - -832.6777) / (1 + (z / 6.527821) ** 2.313703)

#Creates a list with heights above sea level in kilometres between 0 & 50 (Increments by 100 metres).
height = []
height_sea_level = 0
while height_sea_level < 50100:
	height.append(height_sea_level)
	height_sea_level += 100

#Creates a list of atmospheric pressure based on the height list.
pressure = []
for i in height:
	i = p(i)
	pressure.append(i)

#Combines the height above sea level, and the atmospheric pressure lists into a dictionary.
pressure_height = dict(zip(height, pressure))




#-----------------------------------------------------------------------------------------#

#Vertical velocity calculations.
#The equation for vertical velocity (derivative of the pressure equation Dp/ Dt).
def omega(p):
	return -832.6777 + 102156.2777 / (1.49196723444642e-9* p ** 2.313703 + 1)

#Creates a list of vertical velocities based on atmospheric pressure.
vertical_velocity = []
for i in pressure:
	i = omega(i)
	vertical_velocity.append(i)

#Combines the height above sea level, and the vertical velocity lists into a dictionary.
verticalvelocity_height = dict(zip(height, vertical_velocity))

#-----------------------------------------------------------------------------------------#

#Temperature calculations.




#-----------------------------------------------------------------------------------------#

#Geopotential calculations.
#Equation to calculate geopotential.
def Upphi(z):
	return G * m * ((1 / a) - (1 / (a + z)))

#Creates a geopotential list based on the height list.
geopotential = []
for i in height:
	i = Upphi(i)
	geopotential.append(i)

#Combines the geopotential, and the height lists into a dictionary.
geopotential_height = dict(zip(height, geopotential))

#-----------------------------------------------------------------------------------------#

#Coriolis Force
#Equation to calculate coriolis force.
def f(phi):
	return 2 * Upomega * math.sin(math.radians(phi))

#Creates a coriolis force list.
coriolis_force = []
for i in latitude:
	i = f(i)
	coriolis_force.append(i)


#Combines the coriolis force, and the latitude list into a dictionary.
coriolisforce_latitude = dict(zip(latitude, coriolis_force))

#-----------------------------------------------------------------------------------------#