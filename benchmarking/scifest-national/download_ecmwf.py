# Import dependices.
from ecmwfapi import ECMWFDataServer

# Define server.
server = ECMWFDataServer()

# Retrieve data.
server.retrieve({
    "class": "ti",
    "dataset": "tigge",
    "date": "2019-01-01/to/2020-01-01",
    "expver": "prod",
    "grid": "3.0/3.0",
    "levelist": "200/250/300/500/700/850/925/1000",
    "levtype": "pl",
    "origin": "ecmf",
    "area": "89/-180/-89/180",
    "param": "130/131/132/133/156",
    "step": "0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120",
    "time": "00:00:00",
    "type": "fc",
    "format" : "netcdf",
    "target": "ecmwf_forecasts.nc"
})