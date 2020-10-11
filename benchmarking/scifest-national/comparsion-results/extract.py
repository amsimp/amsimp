# Import dependices.
import pickle
import xarray as xr
import numpy as np

# Load metrics.
# ACC.
acc = pickle.load(open('acc.pkl', 'rb'))

# RMSE.
rmse = pickle.load(open('rmse.pkl', 'rb'))

# MAE.
mae = pickle.load(open('mae.pkl', 'rb'))

# Metrics.
metrics = [acc, rmse, mae]

# Forecast Model.
forecast_models = ["Operational", "IFS T63", "IFS T42", "Persistence", "Climatology"]

# Parameters.
# Geopotential.
geopotential = np.zeros((3, len(forecast_models), 10))
# Temperature.
temperature = np.zeros((3, len(forecast_models), 10))

# Loop through to get models and metrics of interest.
for i in range(3):
    for j in range(len(forecast_models)):
        # Model.
        model = metrics[i][forecast_models[j]]

        # Parameters.
        if forecast_models[j] == "Persistence":
            # Geopotential.
            z = model.z.values

            # Temperature.
            t = model.t.values
        elif forecast_models[j] == "Climatology":
            # Geopotential.
            z = model.z.values
            z = np.zeros(10) + z

            # Temperature.
            t = model.t.values
            t = np.zeros(10) + t
        else:
            # Geopotential.
            z = model.z.values[1:]

            # Temperature.
            t = model.t.values[1:]

        # Change shape.
        if len(z) == 20:
            z = z[1::2]
            t = t[1::2]
        elif len(z) == 14:
            z = z[:10]
            t = t[:10]
        elif len(z) == 28:
            z = z[:20]
            z = z[1::2]
            t = t[:20]
            t = t[1::2]

        # Add to NumPy array.
        # Geopotential.
        geopotential[i, j] = z

        # Temperature.
        temperature[i, j] = t

# Save.
np.save('geopotential.npy', geopotential)
np.save('temperature.npy', temperature)
