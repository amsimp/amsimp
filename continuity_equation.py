#Continuity equation in Fluid Dynamics.
#Modules
import matplotlib.pyplot as plt
import sympy as sy

#Fluid density calculations.
#Equation to calculate fluid density based on height above sea level.
def density_cal(h):
	return -0.00666837 + (1.224977 - -0.00666837)/(1 + (h / 8.150575) ** 2.807024)

#Creates a list with heights above sea level in kilometres between 0 & 86 (Increments every kilometre).
height = []
height_sea_level = 0
while height_sea_level < 87:
	height.append(height_sea_level)
	height_sea_level += 1
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
plt.ylabel("Height Above Sea Level (km)")
plt.show()