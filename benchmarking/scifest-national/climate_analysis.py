# Import dependices.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import pickle

# Read NumPy files.
# Folder.
folder = "results/"

# AMSIMP.
# Geopotential.
geopotential = np.load(folder + "climate-geopotential.npy")
amsimp_geopotential, naive_geopotential = np.split(geopotential, 2, axis=1)
# Air Temperature.
temperature = np.load(folder + "climate-temperature.npy")
amsimp_temperature, naive_temperature = np.split(temperature, 2, axis=1)

# Comparsion results.
# Folder.
folder = "comparsion-results/"

# Geopotential.
comparsion_geopotential = np.load(folder + "geopotential.npy")

# Temperature.
comparsion_temperature = np.load(folder + "temperature.npy")

# Plotting.
# Function.
types = "Operational IFS", "IFS T63", "IFS T42", "Persistence", "Climateology"
def plot(x, y, naive, comparsion, title, metric):
    # AMSIMP.
    plt.plot(x[::12], np.mean(y, axis=0)[::12], linestyle="-")
    plt.scatter(x[::12], np.mean(y, axis=0)[::12], label="AMSIMP")

    # Comparsion.
    # Persistence.
    plt.plot(x[::12], np.mean(naive, axis=0)[::12], linestyle=":")
    plt.scatter(x[::12], np.mean(naive, axis=0)[::12], label=types[3])

    # Climateology.
    plt.plot(x[::12], np.resize(comparsion[4], x.shape)[::12], linestyle="--")
    plt.scatter(x[::12], np.resize(comparsion[4], x.shape)[::12], label=types[4])

    # Add labels to the axes.
    plt.xlabel("Forecast Period (Hours)")
    plt.ylabel(metric)

    # Add title.
    plt.title(title+" "+metric)

    # Add legend.
    plt.legend(loc=0)

    # Save figure to file.
    # Define folder.
    folder = title[:-10].lower()
    folder = folder.replace(" ", "_")
    folder = "appendices/"+folder+"/"

    # Check if folder exits, if not create a folder.
    folder_exist = os.path.isdir(folder)
    if not folder_exist:
        os.mkdir(folder)

    # Define filename.
    filename = metric.lower()
    filename = filename.replace(" ", "_")
    filename = filename.replace("-", "_")

    # Save.
    plt.savefig(folder+filename, dpi=300)

    # Close figure.
    plt.close()

def label_decide(num):
    if i == 0:
        label = "Anomaly Correlation Coefficient"
    elif i == 1:
        label = 'Root Mean Squared Error'
    elif i == 2:
        label = 'Mean Absolute Error'

    return label

x = np.linspace(2, 720, 360)

# Temperature.
# Air Temperature.
for i in range(3):
    metric = label_decide(i)
    title = "Air Temperature (800 hPa)"
    plot(
        x, amsimp_temperature[:, i, :], naive_temperature[:, i, :], comparsion_temperature[i, :, :], title, metric
    )

# Geopotential.
for i in range(3):
    metric = label_decide(i)
    title = "Geopotential (500 hPa)"
    plot(
        x, amsimp_geopotential[:, i, :], naive_geopotential[:, i, :], comparsion_geopotential[i, :, :], title, metric
    )
