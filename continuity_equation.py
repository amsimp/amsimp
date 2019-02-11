#Continuity equation in Fluid Dynamics.
#Modules
import matplotlib.pyplot as plt
import sympy as sy

#Fluid density calculations.
#Equation to calculate fluid density based on height above sea level.
def density_cal(h):
	h /= 1000 
	return -0.00666837 + (1.224977 - -0.00666837) / (1 + (h / 8.150575) ** 2.807024)

#Creates a list with heights above sea level in kilometres between 0 & 86 (Increments every 100 metres).
height = []
height_sea_level = 0
while height_sea_level < 86100:
	height.append(height_sea_level)
	height_sea_level += 100
print(height)

#Creates a list of fluid densities based on the height list.
density = []
for i in height:
	i = density_cal(i)
	density.append(i)
print(density)

#Plots fluid density against height above sea level in kilometres.
plt.scatter(density, height)
plt.title("Density-Height Scatter Plot")
plt.xlabel("Fluid Density (kg/m^3)")
plt.ylabel("Height Above Sea Level (m)")
plt.show()

#Differentiated fluid density function.
def diff_density(p):
	return -0.00957204041467721 * p ** 1.807024 / (0.00276867971716256 * p ** 2.807024 + 1) ** 2

#Differentiates the density list.
differentiated_density = []
for i in density:
	i = diff_density(i)
	differentiated_density.append(i)