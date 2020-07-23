"""
AMSIMP | Quickstart Tutorial

The purpose of this quickstart tutorial is to both demonstrate a potential use case
for the software, but, also to showcase its flexibility and power for atmospheric
science research. This script couples the dynamical core of the software with a
simple radiative model, and propagates the radiative forcing into the temperature
calculations in the dynamical core. Radiative forcing is the difference between
insolation absorbed by the Earth and energy radiated back to space. The simple
radiative model in use is a multilayer atmospheric model. In this model, the
vertical structure of the atmosphere is approximated by dividing the atmosphere
into layers. This model assumes: the atmosphere is completely transparent to
solar radiation and is completely opaque to terrestrial radiation. However, 
these assumptions are not true. This model is useful for theoretical studies
of climate sensitivity. While this particular radiative model is used in this
tutorial, it is also possible to intergrate a more complex model, such as the
RRTM, with the software.
"""

# Importing Dependencies
import amsimp
import numpy as np
from astropy import units
from astropy import constants
from datetime import datetime

# User defined physical constants.
# Radiation received by the planet from the sun.
insolation = 1370 * (units.W / units.m ** 2)
# Albedo of the planet.
albedo = 0.5
# Heat capacity of the planet.
heat_capacity_earth = 1e7 * (units.J / units.K)
# Heat capacity of the atmosphere.
heat_capacity_atmosphere = 1e7 * (units.J / units.K)
# The absorptivity of the atmospheric layers.
epsilon = 0.75
# Stefan-Boltzmann constant.
sigma = constants.sigma_sb.value * constants.sigma_sb.unit
# Initial surface temperature.
surface_temperature = np.load("quickstart_data.npy") * units.K
surface_temperature = surface_temperature[-2:0:-1, :]
surface_temperature = surface_temperature[::5, ::5]
# Time.
date = datetime(2020, 7, 20, 0)
delta_t = 2 * units.min
t = 0

# Radiative forcing function to pass into dynamical core.
def radiative_forcing(current_state):
    # Global defined parameters.
    global surface_temperature
    # Time forward.
    global t

    # Retrieve current atmospheric temperature.
    atmospheric_temperature = current_state.temperature()
    # Make output NumPy array.
    radiative_forcing = np.zeros(atmospheric_temperature.value.shape) * (
        units.K / units.s
    )

    # Grid.
    # Latitude.
    lat = current_state.make_3dimensional_array(
        parameter=current_state.latitude_lines(), axis=1
    )
    # Longitude.
    lon = current_state.make_3dimensional_array(
        parameter=current_state.longitude_lines(), axis=2
    )

    # Determine the radiative forcing in the atmosphere.
    # Bottom layer of the atmosphere.
    radiative_forcing[0, :, :] = (
        units.m ** 2
        * (
            (epsilon * sigma * surface_temperature ** 4)
            + (epsilon * sigma * atmospheric_temperature[1, :, :] ** 4)
            - 2 * (epsilon * sigma * atmospheric_temperature[0, :, :] ** 4)
        )
        / (heat_capacity_atmosphere)
    )

    # N-layer of the atmosphere.
    radiative_forcing[1:-1, :, :] = (
        units.m ** 2
        * (
            (epsilon * sigma * atmospheric_temperature[:-2] ** 4)
            + (epsilon * sigma * atmospheric_temperature[2:] ** 4)
            - 2 * (epsilon * sigma * atmospheric_temperature[1:-1, :, :] ** 4)
        )
        / (heat_capacity_atmosphere)
    )

    # Top layer of the atmosphere.
    radiative_forcing[-1, :, :] = (
        units.m ** 2
        * (
            (epsilon * sigma * atmospheric_temperature[-2, :, :] ** 4)
            - 2 * (epsilon * sigma * atmospheric_temperature[-1, :, :] ** 4)
        )
        / (heat_capacity_atmosphere)
    )

    # Surface temperature.
    # Solar radiation.
    sun_longitude = (-t % (60 * 60 * 24)) * (360 / (60 * 60 * 24))
    solar = (
        insolation
        * np.cos(np.radians(lat[0].value))
        * np.cos(np.radians(lon[0].value - sun_longitude))
    )
    solar[solar < 0] = 0 * insolation.unit

    # Surface temperature.
    surface_temperature = surface_temperature + delta_t * units.m ** 2 * (
        ((1 - albedo) * solar)
        + (epsilon * sigma * atmospheric_temperature[0, :, :] ** 4)
        - (sigma * surface_temperature ** 4)
    ) / (heat_capacity_earth)

    # Update time.
    t += delta_t.to(units.s).value

    return radiative_forcing


# Define initial atmospheric state.
state = amsimp.Dynamics(
    ai=False,
    delta_latitude=5,
    delta_longitude=5,
    forecast_length=6,
    delta_t=delta_t,
    input_date=date,
)

# Simulation of atmospheric dynamics with user defined perturbation.
save = state.simulate(perturbations_temperature=radiative_forcing)

# Visualisation of the result.
state.visualise(data=save)

# Delete any downloaded files, and exit.
state.exit()
