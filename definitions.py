#-----------------------------------------------------------------------------------------#

#Importing Dependencies
import math

#-----------------------------------------------------------------------------------------#

#Predefined Constants.
#Angular rotation rate of Earth.
omega = (2 * math.pi) / 24  

#-----------------------------------------------------------------------------------------#

#Zonal velocity calculations.
#Equation to calculate zonal velocity.
def u(phi):
	a = 6378100.0
	return a * omega * ((math.sin(phi) ** 2) / math.cos(phi))

#Creates a list of latitudes (Increments by one degree).
latitude = list(range(-90, 91))

#Calculates zonal velocity at a particular latitude, and adds it to a list.
zonal_velocity = []
for i in latitude:
	i = u(i)
	zonal_velocity.append(i)

#Combines zonal velocity, and the latitude lists into a dictionary.
zonalvelocity_latitude = dict(zip(latitude, zonal_velocity))

#-----------------------------------------------------------------------------------------#

#Meridional velocity calculations.

#-----------------------------------------------------------------------------------------#

#Vertical velocity calculations.
#Atmospheric density calculations.
#Equation to calculate atmospheric density based on height above sea level.
def rho(h):
	h /= 1000 
	return -0.00666837 + (1.224977 - -0.00666837) / (1 + (h / 8.150575) ** 2.807024)

#Creates a list with heights above sea level in kilometres between 0 & 86 (Increments by 100 metres).
height = []
height_sea_level = 0
while height_sea_level < 50100:
	height.append(height_sea_level)
	height_sea_level += 100

#Creates a list of atmospheric densities based on the height list.
density = []
for i in height:
	i = rho(i)
	density.append(i)

#Combines the height above sea level, and the atmospheric density lists into a dictionary.
density_height = dict(zip(height, density))

#-----------------------------------------------------------------------------------------#