# Import dependencies
from time import sleep
import amsimp
import pyowm
import numpy as np

# AMSIMP level of detail.
detail = amsimp.Dynamics(5)

# Geodetic coordinates.
latitude = detail.latitude_lines().value
longitude = detail.longitude_lines().value

# AMSIMP temperature.
amsimp_temperature = detail.temperature()[:, :, 0].value

# AMSIMP atmospheric density.
amsimp_density = detail.density()[:, :, 0].value

# Real time weather.
api_key = "d8e21a680191eb7613420d2cc20e6d0a"
owm = pyowm.OWM(api_key)

n = 0
actual_temperature = []
actual_pressure = []
while n < len(longitude):
    long = float(longitude[n])
    actualtemp_lat = []
    actualp_lat = []

    k = 0
    while k < len(latitude):
        lat = float(latitude[k])

        while True:
            try:
                owm.weather_at_coords(lat, long)
            except:
                sleep(1)
            else:
                break

        obs = owm.weather_at_coords(lat, long)
        w = obs.get_weather()

        actual_temp = w.get_temperature()
        actual_temp = actual_temp["temp"]

        actual_p = w.get_pressure()
        actual_p = actual_p["press"]

        sleep(1)

        actualtemp_lat.append(actual_temp)
        actualp_lat.append(actual_p)

        k += 1

    print("Round: " + str(n + 1) + "/24 complete.")

    actual_temperature.append(actualtemp_lat)
    actual_pressure.append(actualp_lat)

    n += 1

# Actual temperature
actual_temperature = np.asarray(actual_temperature)
actual_pressure = np.asarray(actual_pressure) * detail.units.hPa
actual_temp = actual_temperature * detail.units.K

# Calculate the actual atmospheric density using the Ideal Gas Law.
R = 287 * (detail.units.J / (detail.units.kg * detail.units.K))
actual_density = actual_pressure / (R * actual_temp)
actual_density = actual_density.to(detail.units.kg / (detail.units.m ** 3))
actual_density = actual_density.value

# The temperature mean, and median absolute percentage errors.
temp_error = (actual_temperature - amsimp_temperature) / actual_temperature
temp_error *= 100
temp_error = np.abs(temp_error)
meantemp_error = np.mean(temp_error)
mediantemp_error = np.median(temp_error)
meantemp_error = np.round(meantemp_error, 2)
mediantemp_error = np.round(mediantemp_error, 2)

# The density mean, and median absolute percentage errors.
density_error = (actual_density - amsimp_density) / actual_density
density_error *= 100
density_error = np.abs(density_error)
meandensity_error = np.mean(density_error)
mediandensity_error = np.median(density_error)
meandensity_error = np.round(meandensity_error, 2)
mediandensity_error = np.round(mediandensity_error, 2)

# Print mean absolute percentage error to console.
print(
    "The temperature mean absolute percentage error is: "
    + str(meantemp_error)
    + "%"
)
print(
    "The atmospheric density mean absolute percentage error is: "
    + str(meandensity_error)
    + "%"
)

# Print median absolute percentage error.
print(
    "The temperature median absolute percentage error is: "
    + str(mediantemp_error)
    + "%"
)
print(
    "The atmospheric density median absolute percentage error is: "
    + str(mediandensity_error)
    + "%"
)

mean_error = (meantemp_error + meandensity_error) / 2
median_error = (mediantemp_error + mediandensity_error) / 2

mean_error = np.round(mean_error, 2)
median_error = np.round(median_error, 2)

print("AMSIMP's MAPE: " + str(mean_error) + "%")
print("AMSIMP's MdAPE: " + str(median_error) + "%")
