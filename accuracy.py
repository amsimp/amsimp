from time import sleep
import amsimp
import pyowm
import numpy as np
from pvlib.atmosphere import gueymard94_pw as pw

# AMSIMP level of detail.
detail = amsimp.Water(5)

# Geodetic coordinates.
latitude = detail.latitude_lines().value
longitude = detail.longitude_lines().value

# AMSIMP temperature prediction.
amsimp_temperature = detail.temperature()[:, :, 0].value

# AMSIMP wind prediction.
amsimp_wind = np.sqrt(
    (detail.zonal_wind()[:, :, 0] ** 2) + (detail.meridional_wind()[:, :, 0] ** 2)
).value
amsimp_wind = np.abs(amsimp_wind)

# Remove geostrophic wind values near the equator, as geostrophic balance
# does not hold in the tropics.
amsimp_wind = np.transpose(amsimp_wind)
amsimp_wind = list(amsimp_wind)
for i in range(8):
    del amsimp_wind[8]
amsimp_wind = np.asarray(amsimp_wind)
amsimp_wind = np.transpose(amsimp_wind)

# AMSIMP pressure prediction.
amsimp_pressure = detail.pressure()[:, :, 0].value

# AMSIMP precipitable water predicition.
amsimp_precipitablewater = detail.precipitable_water().value
list_precipitablewater = []
for amsimp_pwv in amsimp_precipitablewater:
    pwv_lat = []
    for amsimp_p in amsimp_pwv:
        ans = np.sum(amsimp_p)
        pwv_lat.append(ans)
    list_precipitablewater.append(pwv_lat)
amsimp_precipitablewater = np.asarray(list_precipitablewater)

# Real time weather.
api_key = "d8e21a680191eb7613420d2cc20e6d0a"
owm = pyowm.OWM(api_key)

n = 0
pwv = pw
actual_temperature = []
actual_pressure = []
actual_wind = []
actual_precipitablewater = []
while n < len(longitude):
    long = float(longitude[n])
    actualtemp_lat = []
    actualp_lat = []
    actualvg_lat = []
    actualpwv_lat = []

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

        actual_vg = w.get_wind()
        actual_vg = actual_vg["speed"]

        pwv_temp = actual_temp - 273.15
        actual_pwv = pwv(pwv_temp, 100)

        sleep(1)

        actualtemp_lat.append(actual_temp)
        actualp_lat.append(actual_p)
        actualvg_lat.append(actual_vg)
        actualpwv_lat.append(actual_pwv)

        k += 1

    print("Round: " + str(n + 1) + " complete.")

    actual_temperature.append(actualtemp_lat)
    actual_pressure.append(actualp_lat)
    actual_wind.append(actualvg_lat)
    actual_precipitablewater.append(actualpwv_lat)

    n += 1

actual_temperature = np.asarray(actual_temperature)
actual_pressure = np.asarray(actual_pressure)
actual_wind = np.asarray(actual_wind)
actual_precipitablewater = np.asarray(actual_precipitablewater) * 10

# Remove geostrophic wind values near the equator, as geostrophic balance
# does not hold in the tropics.
actual_wind = np.transpose(actual_wind)
actual_wind = list(actual_wind)
for i in range(8):
    del actual_wind[8]
actual_wind = np.asarray(actual_wind)
actual_wind = np.transpose(actual_wind)

# The temperature mean, and median absolute percentage errors.
temp_error = (actual_temperature - amsimp_temperature) / actual_temperature
temp_error *= 100
temp_error = np.abs(temp_error)
meantemp_error = np.mean(temp_error)
mediantemp_error = np.median(temp_error)
meantemp_error = np.round(meantemp_error, 1)
mediantemp_error = np.round(mediantemp_error)

# The pressure mean, and median absolute percentage errors.
pressure_error = (actual_pressure - amsimp_pressure) / actual_pressure
pressure_error *= 100
pressure_error = np.abs(pressure_error)
meanpressure_error = np.mean(pressure_error)
medianpressure_error = np.median(pressure_error)
meanpressure_error = np.round(meanpressure_error, 1)
medianpressure_error = np.round(medianpressure_error, 1)

# The wind mean, and median absolute percentage errors.
wind_error = (actual_wind - amsimp_wind) / actual_wind
wind_error *= 100
wind_error = np.abs(wind_error)
meanwind_error = np.mean(wind_error)
medianwind_error = np.median(wind_error)
meanwind_error = np.round(meanwind_error, 1)
medianwind_error = np.round(medianwind_error, 1)

# The precipitable water mean, and median absolute percentage errors.
pwv_error = (actual_precipitablewater - amsimp_precipitablewater) / actual_precipitablewater
pwv_error *= 100
pwv_error = np.abs(pwv_error)
meanpwv_error = np.mean(pwv_error)
medianpwv_error = np.median(pwv_error)
meanpwv_error = np.round(meanpwv_error, 1)
medianpwv_error = np.round(medianpwv_error, 1)

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
print(
    "The precipitable water mean absolute percentage error is: +-"
    + str(meanpwv_error / 2)
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
print(
    "The precipitable water median absolute percentage error is: +-"
    + str(medianpwv_error / 2)
    + "%"
)

mean_error = (meantemp_error + meanpressure_error + meanwind_error + meanpwv_error) / 4
median_error = (mediantemp_error + medianpressure_error + medianwind_error + medianpwv_error) / 4

mean_error /= 2
median_error /= 2

mean_error = np.round(mean_error, 2)
median_error = np.round(median_error, 2)

print("AMSIMP's MAPE: +-" + str(mean_error) + "%")
print("AMSIMP's MdAPE: +-" + str(median_error) + "%")
