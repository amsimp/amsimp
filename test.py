# Import dependicies.
import amsimp
import iris

# Define the operational model and the forecast length.
model = amsimp.OperationalModel(forecast_length=120)

# # Generate a near real-time weather forecast.
fct = model.generate_forecast()

# Save forecast.
iris.save(fct, 'gfm_example.nc')
