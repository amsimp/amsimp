# Import dependices.
import pyowm
import numpy as np
from progress.bar import IncrementalBar
from metpy import calc as cal
from metpy.units import units
from time import sleep
from datetime import datetime
import os

# API Key.
api_key = '3691a6be3fbf3fe998b6f23d0aad1e8a'
owm = pyowm.OWM(api_key)

# Define the lines of latitude and longitude.
# Latitude lines.
sh = [
    i for i in np.arange(-89, 0, 10) if i != 0
]
start_nh = sh[-1] * -1
nh = [
    i for i in np.arange(start_nh, 90, 10) if i != 0 and i != 90
]
for deg in nh:
    sh.append(deg)
# Convert list to NumPy array and add the unit of measurement.
latitude_lines = np.asarray(sh)

# Longitude lines.
longitude_lines = [
    i for i in np.arange(0, 359, 10)
]
# Convert list to NumPy array and add the unit of measurement.
longitude_lines = np.asarray(longitude_lines)
longitude_lines[longitude_lines > 180] -= 360
longitude_lines = np.sort(longitude_lines)

# Atmospheric parameters to collect.
# Temperature.
# Air Temperature.
temperature = []
forecast_temperature = []
# Virtual Temperature.
virtual_temperature = []
forecast_virtualtemperature = []
# Pressure.
pressure = []
forecast_pressure = []
# Humidity.
humidity = []
forecast_humidity = []
# Wind.
# Zonal Wind.
zonal_wind = []
forecast_zonalwind = []
# Meridional Wind.
meridional_wind = []
forecast_meridionalwind = []

# Create a bar to determine progress.
max_bar = len(latitude_lines) * len(longitude_lines)
bar = IncrementalBar('Progress', max=max_bar)

for lat in latitude_lines:
    # Convert lat to float.
    lat = float(lat)

    # Longitudinal variation lists.
    # Temperature.
    # Air Temperature.
    temperature_lat = []
    forecast_temperature_lat = []
    # Virtual Temperature.
    virtual_temperature_lat = []
    forecast_virtualtemperature_lat = []
    # Pressure.
    pressure_lat = []
    forecast_pressure_lat = []
    # Humidity.
    humidity_lat = []
    forecast_humidity_lat = []
    # Wind.
    # Zonal Wind.
    zonal_wind_lat = []
    forecast_zonalwind_lat = []
    # Meridional Wind.
    meridional_wind_lat = []
    forecast_meridionalwind_lat = []

    for lon in longitude_lines:
        # Convert lon to float.
        lon = float(lon)
        
        # Current weather conditions.
        while True:
            try:
                obs = owm.weather_at_coords(lat, lon)
                weather = obs.get_weather()
            except:
                sleep(1)
            else:
                break
        
        # Temperature.
        temp = weather.get_temperature()
        temp = temp.get("temp")
        temperature_lat.append(temp)
        # Pressure.
        p = weather.get_pressure()
        p = p.get("press")
        pressure_lat.append(p)
        # Humidity.
        h = weather.get_humidity()
        humidity_lat.append(h)
        # Wind.
        v_vector = weather.get_wind()
        speed, direction = v_vector.get("speed"), v_vector.get("deg")
        try:
            direction = np.radians(direction)
            u = -speed * np.sin(direction)
            v = -speed * np.cos(direction)
        except:
            u, v = None, None
        # Zonal Wind.
        zonal_wind_lat.append(u)
        # Meridional Wind.
        meridional_wind_lat.append(v)
        # Virtual Temperature.
        if temp != None:
            r = cal.mixing_ratio_from_relative_humidity(
                pressure=p * units.millibars, temperature=temp * units.K, relative_humidity=(h / 100)
            )
            t_v = cal.virtual_temperature(temperature=temp * units.K, mixing=r)
            t_v = t_v.magnitude
        else:
            t_v = None
        virtual_temperature_lat.append(t_v)

        # Get the weather forecast for the next five days.
        while True:
            try:
                obs_forecast = owm.three_hours_forecast_at_coords(lat, lon)
            except:
                sleep(1)
            else:
                break
                
        forecast = obs_forecast.get_forecast()

        # Time variation lists.
        # Temperature.
        # Air Temperature.
        forecast_temperature_time = []
        # Virtual Temperature.
        forecast_virtualtemperature_time = []
        # Pressure.
        forecast_pressure_time = []
        # Humidity.
        forecast_humidity_time = []
        # Wind.
        # Zonal Wind.
        forecast_zonalwind_time = []
        # Meridional Wind.
        forecast_meridionalwind_time = []
        for weather in forecast:
            # Temperature.
            temp = weather.get_temperature()
            temp = temp.get("temp")
            forecast_temperature_time.append(temp)
            # Pressure.
            p = weather.get_pressure()
            p = p.get("press")
            forecast_pressure_time.append(p)
            # Humidity.
            h = weather.get_humidity()
            forecast_humidity_time.append(h)
            # Wind.
            v_vector = weather.get_wind()
            speed, direction = v_vector.get("speed"), v_vector.get("deg")
            try:
                direction = np.radians(direction)
                u = -speed * np.sin(direction)
                v = -speed * np.cos(direction)
            except:
                u, v = None, None
            # Zonal Wind.
            forecast_zonalwind_time.append(u)
            # Meridional Wind.
            forecast_meridionalwind_time.append(v)
            # Virtual Temperature.
            if temp != None:
                r = cal.mixing_ratio_from_relative_humidity(
                    pressure=p * units.millibars, temperature=temp * units.K, relative_humidity=(h / 100)
                )
                t_v = cal.virtual_temperature(temperature=temp * units.K, mixing=r)
                t_v = t_v.magnitude
            else:
                t_v = None
            forecast_virtualtemperature_time.append(t_v)

        # Append to latitudinal list.
        # Temperature.
        # Air Temperature.
        forecast_temperature_lat.append(forecast_temperature_time)
        # Virtual Temperature.
        forecast_virtualtemperature_lat.append(forecast_virtualtemperature_time)
        # Pressure.
        forecast_pressure_lat.append(forecast_pressure_time)
        # Humidity.
        forecast_humidity_lat.append(forecast_humidity_time)
        # Wind.
        # Zonal Wind.
        forecast_zonalwind_lat.append(forecast_zonalwind_time)
        # Meridional Wind.
        forecast_meridionalwind_lat.append(forecast_meridionalwind_time)

        # Wait a second to ensure API requests per minute is not exceeded.
        sleep(1)
        bar.next()

    # Append to main list.
    # Temperature.
    # Air Temperature.
    temperature.append(temperature_lat)
    forecast_temperature.append(forecast_temperature_lat)
    # Virtual Temperature.
    virtual_temperature.append(virtual_temperature_lat)
    forecast_virtualtemperature.append(forecast_virtualtemperature_lat)
    # Pressure.
    pressure.append(pressure_lat)
    forecast_pressure.append(forecast_pressure_lat)
    # Humidity.
    humidity.append(humidity_lat)
    forecast_humidity.append(forecast_humidity_lat)
    # Wind.
    # Zonal Wind.
    zonal_wind.append(zonal_wind_lat)
    forecast_zonalwind.append(forecast_zonalwind_lat)
    # Meridional Wind.
    meridional_wind.append(meridional_wind_lat)
    forecast_meridionalwind.append(forecast_meridionalwind_lat)

# Convert lists to NumPy arrays.
# Temperature.
# Air Temperature.
temperature = np.asarray(temperature)
forecast_temperature = np.asarray(forecast_temperature)
# Virtual Temperature.
virtual_temperature = np.asarray(virtual_temperature)
forecast_virtualtemperature = np.asarray(forecast_virtualtemperature)
# Pressure.
pressure = np.asarray(pressure)
forecast_pressure = np.asarray(forecast_pressure)
# Humidity.
humidity = np.asarray(humidity)
forecast_humidity = np.asarray(forecast_humidity)
# Wind.
# Zonal Wind.
zonal_wind = np.asarray(zonal_wind)
forecast_zonalwind = np.asarray(forecast_zonalwind)
# Meridional Wind.
meridional_wind = np.asarray(meridional_wind)
forecast_meridionalwind = np.asarray(forecast_meridionalwind)

# Current date.
date = datetime.now()
# Define the date, in terms of numbers.
day = date.day
month = date.month
year = date.year
hour = date.hour

# Adds zero before single digit numbers.
if day < 10:
    day = "0" + str(day)   
if month < 10:
    month =  "0" + str(month)
if hour < 10:
    hour = "0" + str(hour)

# Save files.
folder = "openweatherapi/" + str(year) + "/" + str(month) + '/' + str(day) + '/' + str(hour) + '/'
try:
    os.mkdir("openweatherapi/"+str(year))
except OSError:
    pass
try:
    os.mkdir("openweatherapi/"+str(year)+'/'+str(month))
except OSError:
    pass
try:
    os.mkdir("openweatherapi/"+str(year) + "/" + str(month) + '/' + str(day))
except OSError:
    pass
try:
    os.mkdir("openweatherapi/"+str(year) + "/" + str(month) + '/' + str(day) + '/' + str(hour))
except OSError:
    pass

# Temperature.
# Air Temperature.
np.save(folder+'temperature.npy', temperature)
np.save(folder+'forecast_temperature.npy', forecast_temperature)
# Virtual Temperature.
np.save(folder+'virtual_temperature.npy', virtual_temperature)
np.save(folder+'forecast_virtualtemperature.npy', forecast_virtualtemperature)
# Pressure.
np.save(folder+'pressure.npy', pressure)
np.save(folder+'forecast_pressure.npy', forecast_pressure)
# Humidity.
np.save(folder+'humidity.npy', humidity)
np.save(folder+'forecast_humidity.npy', forecast_humidity)
# Wind.
# Zonal Wind.
np.save(folder+'zonal_wind.npy', zonal_wind)
np.save(folder+'forecast_zonalwind.npy', forecast_zonalwind)
# Meridional Wind.
np.save(folder+'meridional_wind.npy', meridional_wind)
np.save(folder+'forecast_meridionalwind.npy', forecast_meridionalwind)

bar.finish()
