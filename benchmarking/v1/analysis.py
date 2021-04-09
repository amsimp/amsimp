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
z = np.load(folder + "z.npy")
amsimp_z, naive_z = np.split(z, 2, axis=1)
# Air temperature.
t = np.load(folder + "t.npy")
amsimp_t, naive_t = np.split(t, 2, axis=1)
# Performance
performace = np.load(folder + "performance.npy")

# Comparsion results.
# Folder.
folder = "comparsion-results/"

# Geopotential at 500 hPa.
comparsion_z = np.load(folder + "geopotential.npy")

# Air temperature at 850 hPa.
comparsion_t = np.load(folder + "temperature.npy")

# Plotting.
# Function.
types = "Operational IFS", "IFS T63", "IFS T42", "Persistence", "Climateology"
def plot(x1, x2, y, naive, comparsion, title, metric):
    # AMSIMP.
    plt.plot(x1[2::6], np.mean(y, axis=0)[2::6], linestyle="-")
    plt.scatter(x1[2::6], np.mean(y, axis=0)[2::6], label="AMSIMP")

    # Comparsion.
    # Operational IFS.
    plt.plot(x2[:10], comparsion[0][:10], linestyle="-")
    plt.scatter(x2[:10], comparsion[0][:10], label=types[0])

    # IFS T63.
    plt.plot(x2[:10], comparsion[1][:10], linestyle="-")
    plt.scatter(x2[:10], comparsion[1][:10], label=types[1])

    # IFS T42.
    plt.plot(x2[:10], comparsion[2][:10], linestyle="-")
    plt.scatter(x2[:10], comparsion[2][:10], label=types[2])

    # Persistence.
    plt.plot(x1[2::6], np.mean(naive, axis=0)[2::6], linestyle=":")
    plt.scatter(x1[2::6], np.mean(naive, axis=0)[2::6], label=types[3])

    # Climateology.
    plt.plot(x2[:10], comparsion[4][:10], linestyle="--")
    plt.scatter(x2[:10], comparsion[4][:10], label=types[4])

    # Add labels to the axes.
    plt.xlabel("Forecast Period (Hours)")
    plt.ylabel(metric)

    # Add title.
    plt.title(title+" "+metric)

    # Add legend.
    plt.legend(loc=0)

    # Save figure to file.
    # Define folder.
    if 'hPa' in title:
        folder = title[:-10].lower()
    else:
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
x2 = np.linspace(6, 168, 14)

# Air Temperature at 850 hPa.
for i in range(3):
    metric = label_decide(i)
    title = "Air Temperature (850 hPa)"
    plot(
        x1, x2, amsimp_t[:, i, :], naive_t[:, i, :], comparsion_t[i, :, :], title, metric
    )

# Geopotential.
for i in range(3):
    metric = label_decide(i)
    title = "Geopotential (500 hPa)"
    plot(
        x1, x2, amsimp_z[:, i, :], naive_z[:, i, :], comparsion_z[i, :, :], title, metric
    )

# Print the mean and median forecast generation time.
print("Mean forecast generation time: " + str(np.mean(performace)))
print("Median forecast generation time: " + str(np.median(performace)))

# Compare against the amount of time taken to generate a IFS T42.
print("Performance Increase (IFS T42): " + str(503 / np.mean(performace)))
