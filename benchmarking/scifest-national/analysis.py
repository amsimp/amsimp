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
geopotential = np.load(folder + "geopotential.npy")
amsimp_geopotential, naive_geopotential = np.split(geopotential, 2, axis=1)
# Air Temperature.
temperature = np.load(folder + "temperature.npy")
amsimp_temperature, naive_temperature = np.split(temperature, 2, axis=1)
# Performance
performace = np.load(folder + "performance.npy")

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
def plot(x1, x2, y, naive, comparsion, title, metric):
    # AMSIMP.
    plt.plot(x1[::3], np.mean(y, axis=0)[::3], linestyle="-")
    plt.scatter(x1[::3], np.mean(y, axis=0)[::3], label="AMSIMP")

    # Comparsion.
    # Operational IFS.
    plt.plot(x2[:10], comparsion[0][:10], linestyle="-")
    plt.scatter(x2[:10], comparsion[0][:10], label=types[0])

    # IFS T63.
    plt.plot(x2, comparsion[1], linestyle="-")
    plt.scatter(x2, comparsion[1], label=types[1])

    # IFS T42.
    plt.plot(x2, comparsion[2], linestyle="-")
    plt.scatter(x2, comparsion[2], label=types[2])

    # Persistence.
    plt.plot(x1[::3], np.mean(naive, axis=0)[::3], linestyle=":")
    plt.scatter(x1[::3], np.mean(naive, axis=0)[::3], label=types[3])

    # Climateology.
    plt.plot(x2, comparsion[4], linestyle="--")
    plt.scatter(x2, comparsion[4], label=types[4])

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
    folder = "plots/"+folder+"/"

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

x1 = np.linspace(2, 168, 84)
x2 = np.linspace(12, 168, 14)

# Temperature.
# Air Temperature.
for i in range(3):
    metric = label_decide(i)
    title = "Air Temperature (800 hPa)"
    plot(
        x1, x2, amsimp_temperature[:, i, :], naive_temperature[:, i, :], comparsion_temperature[i, :, :], title, metric
    )

# Geopotential.
for i in range(3):
    metric = label_decide(i)
    title = "Geopotential (500 hPa)"
    plot(
        x1, x2, amsimp_geopotential[:, i, :], naive_geopotential[:, i, :], comparsion_geopotential[i, :, :], title, metric
    )

# Print the mean and median forecast generation time.
print("Mean forecast generation time: " + str(np.mean(performace)))
print("Median forecast generation time: " + str(np.median(performace)))

# Compare against the amount of time taken to generate a IFS T42.
print("Performance Increase (IFS T42): " + str(240 / np.mean(performace)))
