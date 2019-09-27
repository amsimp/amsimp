from time import sleep
from datetime import datetime
import amsimp
import pyowm
import numpy as np

# Time information for AMSIMP.
date = datetime.now()
day = date.day - 1

# AMSIMP level of detail.
detail = amsimp.Dynamics(3)

# Geodetic coordinates.
latitude = detail.latitude_lines()
longitude = latitude * 2

# AMSIMP temperature prediction.
amsimp_temp = (detail.predict_temperature()[0] * day) + detail.predict_temperature()[1]
amsimp_temp = amsimp_temp[0]

# AMSIMP wind prediction.
amsimp_vg = (
    detail.predict_geostrophicwind()[0] * day
) + detail.predict_geostrophicwind()[1]
amsimp_vg = amsimp_vg[0]

# AMSIMP pressure prediction.
amsimp_p = detail.pressure()[0]

amsimp_temperature = []
amsimp_pressure = []
amsimp_wind = []
n = 0
while n < len(longitude):
    amsimp_temperature.append(list(amsimp_temp))
    amsimp_pressure.append(list(amsimp_p))
    amsimp_wind.append(list(amsimp_vg))
    n += 1

amsimp_temperature = np.asarray(amsimp_temperature)
amsimp_pressure = np.asarray(amsimp_pressure)
amsimp_wind = np.abs(np.asarray(amsimp_wind))

# Real time weather.
api_key = "d8e21a680191eb7613420d2cc20e6d0a"
owm = pyowm.OWM(api_key)

n = 0
actual_temperature = []
actual_pressure = []
actual_wind = []
while n < len(latitude):
    lat = float(latitude[n])
    actualtemp_long = []
    actualp_long = []
    actualvg_long = []

    k = 0
    while k < len(longitude):
        long = float(longitude[k])

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
        actual_p *= 100

        actual_vg = w.get_wind()
        actual_vg = actual_vg["speed"]

        sleep(1)

        actualtemp_long.append(actual_temp)
        actualp_long.append(actual_p)
        actualvg_long.append(actual_vg)

        k += 1

    print("Round: " + str(n + 1) + " complete.")

    actual_temperature.append(actualtemp_long)
    actual_pressure.append(actualp_long)
    actual_wind.append(actualvg_long)

    n += 1

actual_temperature = np.asarray(actual_temperature)
actual_pressure = np.asarray(actual_pressure)
actual_wind = np.asarray(actual_wind)

# The temperature mean, and median absolute percentage errors.
temp_error = (actual_temperature - amsimp_temperature) / actual_temperature
temp_error *= 100
temp_error = np.abs(temp_error)
meantemp_error = np.mean(temp_error)
mediantemp_error = np.median(temp_error)

# The pressure mean, and median absolute percentage errors.
pressure_error = (actual_pressure - amsimp_pressure) / actual_pressure
pressure_error *= 100
pressure_error = np.abs(pressure_error)
meanpressure_error = np.mean(pressure_error)
medianpressure_error = np.median(pressure_error)

# The wind mean, and median absolute percentage errors.
wind_error = (actual_wind - amsimp_wind) / actual_wind
wind_error *= 100
wind_error = np.abs(wind_error)
meanwind_error = np.mean(wind_error)
medianwind_error = np.median(wind_error)

# Print mean absolute percentage error to console.
print(
    "The temperature mean absolute percentage error is: +-"
    + str(meantemp_error / 2)
    + "%"
)
print(
    "The pressure mean absolute percentage error is: +-"
    + str(meanpressure_error / 2)
    + "%"
)
print(
    "The geostrophic wind mean absolute percentage error is: +-"
    + str(meanwind_error / 2)
    + "%"
)

# Print median absolute percentage error.
print(
    "The temperature median absolute percentage error is: +-"
    + str(mediantemp_error / 2)
    + "%"
)
print(
    "The pressure median absolute percentage error is: +-"
    + str(medianpressure_error / 2)
    + "%"
)
print(
    "The geostrophic wind median absolute percentage error is: +-"
    + str(medianwind_error / 2)
    + "%"
)

mean_error = (meantemp_error + meanpressure_error + meanwind_error) / 3
median_error = (mediantemp_error + medianpressure_error + medianwind_error) / 3

mean_error /= 2
median_error /= 2

print("AMSIMP's MAPE: +-" + str(mean_error) + "%")
print("AMSIMP's MdAPE: +-" + str(median_error) + "%")
