# Import dependencies.
import amsimp
import iris

# Load historical data.
data = iris.load('example.nc')

# Define atmospheric state.
state = amsimp.Weather(historical_data=data, forecast_length=(24*30))


