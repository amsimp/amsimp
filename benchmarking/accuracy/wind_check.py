# Import dependencies.
import amsimp
import numpy as np

# Select maximum level of detail.
detail = amsimp.Wind(5)

# Define the zonal and meridional component of geostrophic wind.
u = detail.zonal_wind()
v = detail.meridional_wind()

# Get the gradient of each component in the desired direction, i.e.
# the delta x direction for u, and the delta y direction for v.
u_gradient = np.gradient(u)
v_gradient = np.gradient(v)

u_gradientx = u_gradient[0]
v_gradienty = v_gradient[1]

# Distance between longitude lines at the equator.
eq_longd = 111.19 * detail.units.km
eq_longd = eq_longd.to(detail.units.m)
# Distance of one degree of longitude (e.g. 0W - 1W/1E), measured in metres.
# The distance between two lines of longitude is not constant.
long_d = np.cos(detail.latitude_lines()) * eq_longd
# Distance between latitude lines in the class method, amsimp.Backend.latitude_lines().
delta_x = (
    detail.longitude_lines()[-1].value - detail.longitude_lines()[-2].value
) * long_d

# Defining a 3D longitudinal distance NumPy array.
delta_x = delta_x.value
long_alt = []
len_altitude = len(detail.altitude_level())
for x in delta_x:
    x = x + np.zeros(len_altitude)
    x = list(x)
    long_alt.append(x)

list_deltax = []
len_longitudelines = len(detail.longitude_lines())
n = 0
while n < len_longitudelines:
    list_deltax.append(long_alt)
    n += 1
delta_x = np.asarray(list_deltax)
delta_x *= detail.units.m

# Distance of one degree of latitude (e.g. 0N - 1N/1S), measured in metres.
lat_d = (2 * np.pi * detail.a) / 360
# Distance between latitude lines in the class method, amsimp.Backend.latitude_lines().
delta_y = (
    detail.latitude_lines()[-1].value - detail.latitude_lines()[-2].value
) * lat_d

# Vertification of geostrophic wind calculations through the utilisation of
# the Continuity Equation. Answer should be close to zero.
ans = (u_gradientx / delta_x) + (v_gradienty / delta_y)
ans = np.abs(ans)
print("The Continuity Equation holds (" + str(np.mean(ans.value)) + ").")
