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
geopotential = np.zeros((3, len(forecast_models), 14))
# Temperature.
temperature = np.zeros((3, len(forecast_models), 14))
# Total precipitation.
total_precipitation = np.zeros((3, len(forecast_models), 14))
# 2 metre temperature.
temperature_2m = np.zeros((3, len(forecast_models), 14))

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

            # Total precipitation.
            tp = model.tp.values

            # 2 metre temperature.
            t2m = model.t2m.values 
        elif forecast_models[j] == "Climatology":
            # Geopotential.
            z = model.z.values
            z = np.zeros(14) + z

            # Temperature.
            t = model.t.values
            t = np.zeros(14) + t

            # Total precipitation.
            tp = model.tp.values
            tp = np.zeros(14) + tp

            # 2 metre temperature.
            t2m = model.t2m.values 
            t2m = np.zeros(14) + t2m
        else:
            # Geopotential.
            z = model.z.values[1:]

            # Temperature.
            t = model.t.values[1:]

            # Total precipitation.
            if not "IFS" in forecast_models[j]:
                tp = model.tp.values[1:]

            # 2 metre temperature.
            t2m = model.t2m.values[1:] 

        # Change shape.
        if len(z) == 20:
            # Geopotential.
            z = z[1::2]
            z = np.resize(z, (14))

            # Temperature.
            t = t[1::2]
            t = np.resize(t, (14))

            # Total precipitation.
            if not "IFS" in forecast_models[j]:
                tp = tp[1::2]
                tp = np.resize(tp, (14))

            # 2 metre temperature.
            t2m = t2m[1::2]
            t2m = np.resize(t2m, (14))
        elif len(z) == 28:
            # Geopotential.
            z = model.z.values[1::2]

            # Temperature.
            t = model.t.values[1::2]

            # Total precipitation.
            if not "IFS" in forecast_models[j]:
                tp = model.tp.values[1::2]

            # 2 metre temperature.
            t2m = model.t2m.values[1::2] 

        # Add to NumPy array.
        # Geopotential.
        geopotential[i, j] = z

        # Temperature.
        temperature[i, j] = t

        # Total precipitation.
        total_precipitation[i, j] = tp

        # 2 metre temperature.
        temperature_2m[i, j] = t2m

# Save.
np.save('geopotential.npy', geopotential)
np.save('temperature.npy', temperature)
np.save('total_precipitation.npy', total_precipitation)
np.save('2m_temperature.npy', temperature_2m)
