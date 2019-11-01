"""
AMSIMP - An Open Source Implementation to Simulating Atmospheric
Dynamics in the Troposphere and the Stratosphere, written in
Cython.

This is the test coverage suite for AMSIMP. Test coverage is a meausure
of how much of the source code is excuted when a particular test suite
is run. Software with a high percentage of coverage has a lower
probabilty of containing undetected software bugs.
"""
# Import dependencies.
import matplotlib.pyplot as plt
import amsimp

# Define two levels of detail. The variable detail_1 is created to
# ensure that the methods that require a level of detail of 3, or
# greater raise an exception when called by a lower level of
# detail.
detail = amsimp.Dynamics(5)
detail_1 = amsimp.Dynamics(1)
# Ensure detail_level and forecast_days raise an exception when
# it is appropriate to do so.
# Ensure detail_level cannot be above 5.
try:
    detail = amsimp.Dynamics(6)
except Exception:
    pass
else:
    raise Exception("Test Failed!")
# Ensure detail_level cannot be below 1.
try:
    detail = amsimp.Dynamics(0)
except Exception:
    pass
else:
    raise Exception("Test Failed!")
# Ensure forecast_days cannot be above 5.
try:
    detail = amsimp.Dynamics(3, 6)
except Exception:
    pass
else:
    raise Exception("Test Failed!")
# Ensure forecast_days cannot be below 1.
try:
    detail = amsimp.Dynamics(3, 0)
except Exception:
    pass
else:
    raise Exception("Test Failed!")

# Make Matplotlib plots interactive, so, they can be closed
# with the command, plt.close()
plt.ion()

# amsimp.Backend
# Ensure the gravitational_acceleration method functions correctly.
detail.gravitational_acceleration()
# Ensure that the methods, temperature and density, function at each
# level of detail.
for i in range(5):
    i += 1
    detail_i = amsimp.Backend(i)
    detail_i.latitude_lines()
    detail_i.temperature()
    detail_i.density()
# Ensure the methods, sigma_coordinates, potential_temperature
# and exner_function, function correctly.
detail.sigma_coordinates()
detail.potential_temperature()
detail.exner_function()
# Ensure that each plot option in the longitude_contourf method functions
# correctly. Also ensure that an exception is raised when the value of,
# which, is either greater than 2 or less than 0.
for i in range(5):
    i -= 1
    if i < 0 or i > 2:
        try:
            detail.longitude_contourf(i)
        except Exception:
            pass
        else:
            raise Exception("Test Failed!")
    else:
        detail.longitude_contourf(i)
# Ensure that an exception is raised when which is a non-integer
# number.
try:
    detail.longitude_contourf(1.5)
except Exception:
    pass
else:
    raise Exception("Test Failed!")
# Ensure that an exception is raised when, alt, has a value less
# than 0.
try:
    detail.longitude_contourf(0, -1)
except Exception:
    pass
else:
    raise Exception("Test Failed!")
detail.altitude_contourf()
# Ensure that an exception is raised when, which, has a value
# greater than 0.
try:
    detail.altitude_contourf(3)
except Exception:
    pass
else:
    raise Exception("Test Failed!")
# Ensure that an exception is raised when, central_long, has a value
# greater than 180.
try:
    detail.altitude_contourf(0, 200)
except Exception:
    pass
else:
    raise Exception("Test Failed!")
# Ensure that the pressurethickness_contourf method functions correctly.
detail.pressurethickness_contourf()

# amsimp.Wind
# Ensure that the methods, zonal_wind and meridional method, cannot be
# called when the level of detail is less than 3.
try:
    detail_1.zonal_wind()
except Exception:
    pass
else:
    raise Exception("Test Failed!")
try:
    detail_1.meridional_wind()
except Exception:
    pass
else:
    raise Exception("Test Failed!")
# Ensure that the zonal and meridional contour plots are generated
# correctly.
detail.wind_contourf()
detail.wind_contourf(1)
# Ensure that an exception is raised when, which, has a value greater
# than 1.
try:
    detail.wind_contourf(2)
except Exception:
    pass
else:
    raise Exception("Test Failed!")
# Ensure that an exception is raised when, central_long, has a value
# greater than 180.
try:
    detail.wind_contourf(0, 200)
except Exception:
    pass
else:
    raise Exception("Test Failed!")
# Ensure than the globe method functions correctly.
detail.globe()
# Ensure that an exception is raised in the globe method where it is
# appropriate to do so.
try:
    detail.globe(100)
except Exception:
    pass
else:
    raise Exception("Test Failed!")
try:
    detail.globe(45, 200)
except Exception:
    pass
else:
    raise Exception("Test Failed!")
try:
    detail.globe(45, 90, -1)
except Exception:
    pass
else:
    raise Exception("Test Failed!")

# amsimp.water.Water
# Ensure you cannot call the vapor_pressure method if the value of
# detail_level is below zero.
try:
    detail_1.vapor_pressure()
except Exception:
    pass
else:
    raise Exception("Test Failed!")
# Ensure the precipitable water method is functioning correctly when
# sum_altitude has a boolean value of True.
detail.precipitable_water()
# Ensure the precipitable water method only accepts boolean values
# for sum_altitude.
try:
    detail.precipitable_water(0)
except Exception:
    pass
else:
    raise Exception("Test Failed!")
# Ensure the water_contourf method functions correctly.
detail.water_contourf()

# amsimp.dynamics.Dynamics
# Ensure the forecast_temperature method functions correctly.
detail.forecast_temperature()
# Ensure the forecast_precipitablewater method functions correctly.
detail.forecast_precipitablewater()
