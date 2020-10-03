# Import dependencies.
import os
import iris
import numpy as np
from iris.cube import CubeList, Cube
import wget
from datetime import datetime, timedelta
from urllib.error import HTTPError
import numpy as np
from tqdm import tqdm
from astropy import units
import sys
import tarfile

# File locations (NOAA).
file_locations = [
    "HAS011571508", # Jan
    "HAS011571510", # Feb
    "HAS011571512", # Mar
    "HAS011571514", # Apr
    "HAS011571516", # May
    "HAS011571518", # Jun
    "HAS011571521", # Jul
    "HAS011571524", # Aug
    "HAS011571526", # Sep
    "HAS011571531", # Oct
    "HAS011571533", # Nov
    "HAS011571535"  # Dec
]

# Define the start date.
date = datetime(2019, 1, 1, 0)
end = datetime(2020, 1, 1, 0)
diff = end - date
days_diff = diff.days
it = int(days_diff / 2)
# Define the date components.
day = date.day
month = date.month
year = date.year

# Retrieve pressure and longitude grid from example.nc file.
lon = iris.load('example.nc')[0].coord('longitude')
p = iris.load('example.nc')[0].coord('air_pressure').points

# Function to define coordinates of forecast and preprocess data.
def preprocess(cube):
    # Ensure highest pressure surface is the first index of pressure,
    # and the south pole as the first index of latitude. Points of singularity
    # at poles not considered.
    cube = cube[::-1, ::-1, :]
    cube = cube[1:-1, 1:-1, :]

    # Decrease resolution to 3 degrees of latitude and longitude.
    cube = cube[:, ::3, ::3]

    # Convert pressure surfaces to hectopascals.
    cube.coord('pressure').convert_units('hectopascal')

    # Change longitude coordinate system.
    cube_data = cube.data
    cube_edata, cube_wdata = np.split(cube_data, 2, axis=2)
    cube_data = np.concatenate((cube_wdata, cube_edata), axis=2)

    # Capture time, forecast reference time, and forecast period.
    # Forecast reference time.
    ref_time = cube.coord('forecast_reference_time')
    # Forecast period.
    forecast_period = cube.coord('forecast_period')
    # Time.
    time = cube.coord('time')
    
    # Create new cube.
    cube = Cube(cube_data,
        standard_name=cube.standard_name,
        units=cube.units,
        dim_coords_and_dims=[
            (cube.coord('pressure'), 0), 
            (cube.coord('latitude'), 1), 
            (lon, 2)
        ],
        attributes={
            'GRIB_PARAM': cube.attributes['GRIB_PARAM'],
        }
    )

    # Interpolate.
    grid_points = [
        ('pressure', p),
        ('latitude',  cube.coord('latitude').points),
        ('longitude', cube.coord('longitude').points)
    ]
    cube = cube.interpolate(grid_points, iris.analysis.Nearest())

    # Add aux coords.
    # Forecast reference time.
    cube.add_aux_coord(ref_time)
    # Forecast period.
    cube.add_aux_coord(forecast_period)
    # Time.
    cube.add_aux_coord(time)

    return cube

for i in range(it):
    # Define the date.
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

    # Converts integers to strings.
    day = str(day)
    month = str(month)
    year = str(year)
    hour = str(hour)

    # Retrieve GFS from the NOAA database.
    # Define current forecast day.
    folder = "http://www1.ncdc.noaa.gov/pub/has/model/" + file_locations[date.month-1] + "/"
    file = "gfs_3_" + year + month + day
    download_file = folder + file + hour + ".g2.tar"

    # Advance date.
    date = date + timedelta(days=+2)

    # Download file.
    data_file = wget.download(download_file)

    # Unzip file.
    os.mkdir("temp")
    tar_file = tarfile.open(data_file)
    tar_file.extractall('temp')
    tar_file.close()
    
    # Remove zip file.
    os.remove(data_file)

    # Define cube lists.
    # Air temperature.
    temperature_cubelist = CubeList([])
    # Wind.
    # Zonal.
    zonalwind_cubelist = CubeList([])
    # Meridional.
    meridionalwind_cubelist = CubeList([])
    # Relative humidity.
    relativehumidity_cubelist = CubeList([])
    # Geopotential height.
    geopotentialheight_cubelist = CubeList([])

    # Download forecast of 120 hours in length.
    t = 3
    forecast_length = 120
    while t <= forecast_length:
        # Define file to download.
        fname = "temp/" + file + "_0000_" + ('%03d' % t) + ".grb2"

        # Load file.
        data = iris.load(fname)

        # Retrieve temperature, wind, relative humidity, and geopotential height data.
        temperature = data.extract('air_temperature')[0]
        rh = data.extract('relative_humidity')[0]
        geopotential_height = data.extract('geopotential_height')[0]
        zonal_wind = data.extract('x_wind')[-2]
        meridional_wind = data.extract('y_wind')[-2]

        # Append cubes to cube list.
        # Air temperature.
        temperature = preprocess(temperature)
        temperature_cubelist.append(temperature)
        # Relative humidity.
        rh = preprocess(rh)
        relativehumidity_cubelist.append(rh)
        # Wind.
        # Zonal.
        zonal_wind = preprocess(zonal_wind)
        zonalwind_cubelist.append(zonal_wind)
        # Meridional.
        meridional_wind = preprocess(meridional_wind)
        meridionalwind_cubelist.append(meridional_wind)
        # Geopotential height.
        geopotential_height = preprocess(geopotential_height)
        geopotentialheight_cubelist.append(geopotential_height)

        # Add time.
        t += 3

    # Remove temporary directory.
    os.rmdir("temp")

    # Merge cubes.
    # Air temperature.
    temperature_cube = temperature_cubelist.merge()[0]
    # Wind.
    # Zonal.
    zonalwind_cube = zonalwind_cubelist.merge()[0]
    # Meridional.
    meridionalwind_cube = meridionalwind_cubelist.merge()[0]
    # Relative humidity.
    relativehumidity_cube = relativehumidity_cubelist.merge()[0]
    # Geopotential height.
    geopotentialheight_cube = geopotentialheight_cubelist.merge()[0]

    # Define output cube list.
    DataList = CubeList(
        [
            temperature_cube,
            zonalwind_cube,
            meridionalwind_cube,
            relativehumidity_cube,
            geopotentialheight_cube
        ]
    )
    
    # Save to file.
    # Define folder.
    folder = 'gfs-forecasts/' 

    # Save forecast in .nc file
    iris.save(DataList, folder + date.strftime('%Y%m%d') + '.nc')

    # Print progress.
    print("")
    print("Date: " + date.strftime('%Y-%m-%d'))
