# Import dependencies.
import amsimp
import iris
from time import time
import numpy as np

# Load historical data.
data = iris.load('example.nc')

# Define atmospheric state.
state = amsimp.Weather(historical_data=data, forecast_length=(24*30))

# Define performance list.
performance = []

# Generate 10 forecasts, and measure time to generate each one.
total = 10
for i in range(total):
    # Time before forecast is generated.
    start = time()

    # Generate forecast.
    fct = state.generate_forecast()

    # Time taken to generate forecast.
    runtime = time() - start

    # Append to list
    performance.append(runtime)

    # Print progress.
    print("Progress: " + str(i+1) + "/" + str(total))

np.save('results/performance_month.npy', np.asarray(performance))
