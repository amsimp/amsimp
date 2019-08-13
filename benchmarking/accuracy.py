from time import sleep
from datetime import datetime
import amsimp
import pyowm
import numpy as np
import matplotlib.pyplot as plt

# Time information for AMSIMP.
date = datetime.now()
day = date.day - 1

# AMSIMP level of detail.
detail = amsimp.Weather(3)

# Geodetic coordinates.
latitude = detail.latitude_lines()
longitude = latitude * 2

# AMSIMP temperature prediction.
amsimp_temp = (detail.predict_temperature()[0] * day) + detail.predict_temperature()[1]
amsimp_temp = amsimp_temp[0]

# AMSIMP wind prediction.
amsimp_vg = (detail.predict_geostrophicwind()[0] * day) + detail.predict_geostrophicwind()[1]
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
api_key = 'd8e21a680191eb7613420d2cc20e6d0a'
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
        
        obs = owm.weather_at_coords(lat, long)
        w = obs.get_weather()
        
        actual_temp = w.get_temperature()
        actual_temp = actual_temp['temp']

        actual_p = w.get_pressure()
        actual_p = actual_p['press']
        actual_p *= 100

        actual_vg = w.get_wind()
        actual_vg = actual_vg['speed']

        actualtemp_long.append(actual_temp)
        actualp_long.append(actual_p)
        actualvg_long.append(actual_vg)

        k += 1
    
    print('Round: ' + str(n + 1) + ' complete.')

    actual_temperature.append(actualtemp_long)
    actual_pressure.append(actualp_long)
    actual_wind.append(actualvg_long)

    n += 1

actual_temperature = np.asarray(actual_temperature)
actual_pressure = np.asarray(actual_pressure)
actual_wind = np.asarray(actual_wind)

# The temperature mean absolute percentage error.
temp_error = (amsimp_temperature - actual_temperature) / actual_temperature
temp_error *= 100
temp_error = np.abs(temp_error)
temp_error = np.mean(temp_error)

# The pressure mean absolute percentage error
pressure_error = (amsimp_pressure - actual_pressure) / actual_pressure
pressure_error *= 100
pressure_error = np.abs(pressure_error)
pressure_error = np.mean(pressure_error)

# The wind mean absolute percentage error
wind_error = (amsimp_wind - actual_wind) / actual_wind
wind_error *= 100
wind_error = np.abs(wind_error)
wind_error = np.mean(wind_error)

print('The temperature mean absolute percentage error is: ' + str(temp_error) + '%')
print('The pressure mean absolute percentage error is: ' + str(pressure_error) + '%')
print('The wind mean absolute percentage error is: ' + str(wind_error) + '%')

print(actual_wind)
