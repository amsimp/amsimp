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
amsimp_geopotential = np.load(folder + "geopotential.npy")
# Air Temperature.
amsimp_temperature = np.load(folder + "temperature.npy")
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
def plot(x1, x2, y, comparsion, title, metric):
    # AMSIMP.
    # Annual mean (AMSIMP).
    plt.plot(x1, np.mean(y, axis=0), linestyle="-")
    plt.scatter(x1, np.mean(y, axis=0), label="AMSIMP (Annual Mean)")

    # Seasonal variation in performance.
    # Split dataset into the appropriate months.
    # AMSIMP.
    y_q1, y_q2, y_q3, y_q4 = np.split(y[1:, :], 4, axis=0)

    # Jan - Mar.
    # AMSIMP.
    plt.plot(x1, np.mean(y_q1, axis=0), linestyle="-")
    plt.scatter(x1, np.mean(y_q1, axis=0), label="AMSIMP (Jan-Mar Mean)")

    # Apr - Jun.
    # AMSIMP.
    plt.plot(x1, np.mean(y_q2, axis=0), linestyle="-")
    plt.scatter(x1, np.mean(y_q2, axis=0), label="AMSIMP (Apr-Jun Mean)")

    # Jul - Sept.
    # AMSIMP.
    plt.plot(x1, np.mean(y_q3, axis=0), linestyle="-")
    plt.scatter(x1, np.mean(y_q3, axis=0), label="AMSIMP (Jul-Sept Mean)")

    # Oct - Dec.
    # AMSIMP.
    plt.plot(x1, np.mean(y_q4, axis=0), linestyle="-")
    plt.scatter(x1, np.mean(y_q4, axis=0), label="AMSIMP (Oct-Dec Mean)")

    # Comparsion.
    # Operational IFS.
    plt.plot(x1, np.mean(comparsion[0], axis=0), linestyle="-")
    plt.scatter(x1, np.mean(comparsion[0], axis=0), label=types[0])

    # IFS T63.
    plt.plot(x1, np.mean(comparsion[1], axis=0), linestyle="-")
    plt.scatter(x1, np.mean(comparsion[1], axis=0), label=types[1])

    # IFS T42.
    plt.plot(x1, np.mean(comparsion[2], axis=0), linestyle="-")
    plt.scatter(x1, np.mean(comparsion[2], axis=0), label=types[2])

    # Persistence.
    plt.plot(x1, np.mean(comparsion[3], axis=0), linestyle=":")
    plt.scatter(x1, np.mean(comparsion[3], axis=0), label=types[3])

    # Climateology.
    plt.plot(x1, np.mean(comparsion[4], axis=0), linestyle="--")
    plt.scatter(x1, np.mean(comparsion[4], axis=0), label=types[4])

    # Add labels to the axes.
    plt.xlabel("Forecast Period (Hours)")
    plt.ylabel(metric)

    # Add title.
    plt.title(title+" "+metric)

    # Add legend.
    plt.legend(loc=0)

    # Save figure to file.
    # Define folder.
    folder = title.lower()
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

x1 = np.linspace(2, 120, 60)
x2 = np.linspace(12, 120, 10)

# Temperature.
# Air Temperature.
for i in range(3):
    metric = label_decide(i)
    title = "Air Temperature"
    plot(x1, x2, amsimp_temperature[:, i, :], comparsion_temperature[i, :, :], title, metric)

# Geopotential.
for i in range(3):
    metric = label_decide(i)
    title = "Geopotential"
    plot(x1, x2, amsimp_geopotential[:, i, :], comparsion_geopotential[i, :, :], title, metric)

# Performance.
print("Performance: ")

# Print the mean and median forecast generation time.
print("Mean forecast generation time: " + str(np.mean(performace)))
print("Median forecast generation time: " + str(np.median(performace)))
