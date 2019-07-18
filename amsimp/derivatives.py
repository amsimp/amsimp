"""
AMSIMP Derivatives File - This particular file calculates all the derivatives used in the
AMSIMP Wind, and Water classes. Calculations of such derivatives are made using the
autograd module. DO NOT INTERACT WITH THE FUNCTIONS WITHIN THIS CLASS!!!!!!!!!
If you see any problems: contact the developer, create an issue, or a pull request
on GitHub.
"""

# -----------------------------------------------------------------------------------------#

# Importing Dependencies
from autograd import grad
import autograd.numpy as np
from scipy import integrate
from amsimp.backend import Backend

# -----------------------------------------------------------------------------------------#

def gravitational_acceleration(latitude, altitude):
    # Gravitational Acceleration
    latitude = np.radians(latitude)
    a = 6378137
    b = 6356752.3142
    g_e = 9.7803253359
    g_p = 9.8321849378

    g_0 = (
        (a * g_e * (np.cos(latitude) ** 2)) + (b * g_p * (np.sin(latitude) ** 2))
    ) / np.sqrt(
        ((a ** 2) * (np.cos(latitude) ** 2) + (b ** 2) * (np.sin(latitude) ** 2))
    )

    f = (a - b) / a
    m = ((Backend.Upomega ** 2) * (a ** 2) * b) / (Backend.G * Backend.M)

    g_z = g_0 * (
        1
        - (2 / a) * (1 + f + m - 2 * f * (np.sin(latitude) ** 2)) * altitude
        + (3 / (a ** 2)) * (altitude ** 2)
    )

    return g_z

# Calculations for Vertical Velocity (Differentiation of Pressure).
def pressure(latitude, altitude):
    p_0 = 101325

    if altitude < 11000:
        temperature = -0.0065*altitude + 288.185
    elif altitude < 20000:
        temperature = 216.65
    elif altitude < 32000:
        temperature = 0.001*altitude + 196.65
    elif altitude < 47000:
        temperature = 0.0028*altitude + 139.05
    else:
        temperature = 270.65

    g_z = gravitational_acceleration(latitude, altitude)

    # Fin
    y = p_0 * np.exp(-((Backend.m * g_z) / (Backend.R * temperature)) * altitude)

    return y

verticalvelocity_component = grad(pressure, 1)

# Differentiation of Geopotential Height
def geopotential_height(latitude, altitude):
    latitude = np.radians(latitude)
    a = 6378137
    b = 6356752.3142
    g_e = 9.7803253359
    g_p = 9.8321849378

    g_0 = (
        (a * g_e * (np.cos(latitude) ** 2)) + (b * g_p * (np.sin(latitude) ** 2))
    ) / np.sqrt(
        ((a ** 2) * (np.cos(latitude) ** 2) + (b ** 2) * (np.sin(latitude) ** 2))
    )

    f = (a - b) / a
    m = ((Backend.Upomega ** 2) * (a ** 2) * b) / (Backend.G * Backend.M)

    geopotential = altitude * (
        g_0
        * (
            1
            - (2 / a) * (1 + f + m - 2 * f * (np.sin(latitude) ** 2)) * altitude
            + (3 / (a ** 2)) * (altitude ** 2)
        )
    )

    geopotential_height = geopotential / Backend.g

    return geopotential_height


dev_geopotentialheight = grad(geopotential_height, 0)

# Differentiation of Precipitable Water
def precipitable_water(latitude, altitude):
    latitude = np.radians(latitude)
    p = pressure(latitude, altitude)
    a = 6378137
    b = 6356752.3142
    g_e = 9.7803253359
    g_p = 9.8321849378

    vapor_pressure = 140.6050125026231

    water_density = 0.0011290690010197218

    p_1 = pressure(latitude, 0)
    p_2 = pressure(latitude, 50000)
    t = np.linspace(p_1, p_2, 10000)
    mixingratio = (0.622 * vapor_pressure) / (t - vapor_pressure)
    integrated_mixingratio = np.trapz(mixingratio, t)

    g = -9.729648886943384

    precipitable_water = (1 / (water_density * g)) * integrated_mixingratio

    return precipitable_water