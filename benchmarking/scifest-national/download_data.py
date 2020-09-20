import cdsapi

c = cdsapi.Client()

var = [
    'geopotential',
    'relative_humidity', 
    'temperature', 
    'u_component_of_wind', 
    'v_component_of_wind',
]

for i in range(5):
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'variable': var[i],
            'pressure_level': [
                '1', '2', '5',
                '10', '20', '50',
                '100', '200', '300',
                '400', '500', '600',
                '700', '800', '900',
                '950', '1000',
            ],
            'year': '2019',
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '02:00', '04:00',
                '06:00', '08:00', '10:00',
                '12:00', '14:00', '16:00',
                '18:00', '20:00', '22:00',
            ],
            'format': 'netcdf',
            'grid': [3.0, 3.0],
            'area': [
                89, -180, -89,
                180,
            ],
        },
        'historical-data/'+var[i]+'.nc')
