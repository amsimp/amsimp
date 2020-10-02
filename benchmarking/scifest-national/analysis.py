# Import dependices.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Read NumPy files.
# Folder.
folder = "results/"

# AMSIMP.
# Geopotential Height.
geo = np.load(folder+"geopotential.npy")
# Temperature
# Air Temperature.
temp = np.load(folder+"temperature.npy")
# Relative Humidity.
rh = np.load(folder+"relative_humidity.npy")
# Wind.
# Zonal Wind.
u = np.load(folder+"zonal_wind.npy")
# Meridional Wind.
v = np.load(folder+"meridional_wind.npy")
# Performance
performace = np.load(folder+"performance.npy")

# Global Forecast System (to be added).

# Plotting.
# Function.
types = "AMSIMP", "Global Forecast System"
def plot(x, y1, y2, title, metric):
    # Annual mean (AMSIMP).
    plt.plot(x, np.mean(y1, axis=0), linestyle="-")
    plt.scatter(x, np.mean(y1, axis=0), label=types[0]+" (Annual Mean)")

    # Annual mean (GFS).
    # plt.plot(x, np.mean(y2, axis=0), linestyle="--")
    # plt.scatter(x, np.mean(y2, axis=0), label=types[1]+" (Annual Mean)")

    # Seasonal variation in performance.
    # Split dataset into the appropriate months.
    # AMSIMP.
    y1_q1, y1_q2, y1_q3, y1_q4 = np.split(y1[1:, :], 4, axis=0)

    # GFS.
    # y2_q1, y2_q2, y2_q3, y2_q4 = np.split(y2[1:, :], 4, axis=0)

    # Jan - Mar.
    # AMSIMP.
    plt.plot(x, np.mean(y1_q1, axis=0), linestyle="-")
    plt.scatter(x, np.mean(y1_q1, axis=0), label=types[0]+" (Jan-Mar Mean)")

    # GFS.
    # plt.plot(x, np.mean(y2_q1, axis=0), linestyle="--")
    # plt.scatter(x, np.mean(y2_q1, axis=0), label=types[1]+" (Jan-Mar Mean)")

    # Apr - Jun.
    # AMSIMP.
    plt.plot(x, np.mean(y1_q2, axis=0), linestyle="-")
    plt.scatter(x, np.mean(y1_q2, axis=0), label=types[0]+" (Apr-Jun Mean)")

    # GFS.
    # plt.plot(x, np.mean(y2_q2, axis=0), linestyle="--")
    # plt.scatter(x, np.mean(y2_q2, axis=0), label=types[1]+" (Apr-Jun Mean)")

    # Jul - Sept.
    # AMSIMP.
    plt.plot(x, np.mean(y1_q3, axis=0), linestyle="-")
    plt.scatter(x, np.mean(y1_q3, axis=0), label=types[0]+" (Jul-Sept Mean)")

    # GFS.
    # plt.plot(x, np.mean(y2_q3, axis=0), linestyle="--")
    # plt.scatter(x, np.mean(y2_q3, axis=0), label=types[1]+" (Jul-Sept Mean)")

    # Oct - Dec.
    # AMSIMP.
    plt.plot(x, np.mean(y1_q4, axis=0), linestyle="-")
    plt.scatter(x, np.mean(y1_q4, axis=0), label=types[0]+" (Oct-Dec Mean)")

    # GFS.
    # plt.plot(x, np.mean(y2_q4, axis=0), linestyle="--")
    # plt.scatter(x, np.mean(y2_q4, axis=0), label=types[1]+" (Oct-Dec Mean)")

    # Define the naïve forecast mean absolute scaled error.
    if metric == 'Mean Absolute Scaled Error':
        plt.plot([x.min(), x.max()], [1, 1], label="Naïve", linestyle='dashdot')
    
    # Define the climatological normalised root mean squared error.
    if metric == "Normalised Root Mean Squared Error":
        plt.plot([x.min(), x.max()], [1, 1], label="Climatology", linestyle='dashdot')

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
        label = "Pearson Product-Moment Correlation Coefficient"
    elif i == 1:
        label = 'Root Mean Squared Error'
    elif i == 2:
        label = 'Normalised Root Mean Squared Error'
    elif i == 3:
        label = 'Mean Squared Error'
    elif i == 4:
        label = 'Mean Absolute Error'
    elif i == 5:
        label = 'Mean Absolute Scaled Error'
    return label

indices = np.linspace(2, 120, 60)

# Temperature.
# Air Temperature.
for i in range(len(temp[0])):
    metric = label_decide(i)
    title = "Air Temperature"
    plot(indices, temp[:, i, :], None, title, metric)

# Relative Humidity.
for i in range(len(rh[0])):
    metric = label_decide(i)
    title = "Relative Humidity"
    plot(indices, rh[:, i, :], None, title, metric)

# Wind.
# Zonal Wind.
for i in range(len(u[0])):
    metric = label_decide(i)
    title = "Zonal Wind"
    plot(indices, u[:, i, :], None, title, metric)

# Meridional Wind.
for i in range(len(v[0])):
    metric = label_decide(i)
    title = "Meridional Wind"
    plot(indices, v[:, i, :], None, title, metric)

# Geopotential.
for i in range(len(geo[0])):
    metric = label_decide(i)
    title = "Geopotential"
    plot(indices, geo[:, i, :], None, title, metric)

# Performance.
print("Performance: ")

# Print the mean and median forecast generation time.
print("Mean forecast generation time: " + str(np.mean(performace)))
print("Median forecast generation time: " + str(np.median(performace)))

# Add space between accuracy and performance in terminal output.
# print("")

# Statistical analysis.
# print("Accuracy")

# Metric (MSE)
i = 2

# Determine if the GFS leads to a significantly significant increase in accuracy.
# print("Two-sample independent t-test")

# Air temperature.
# print("Air temperature (GFS): " + str(stats.ttest_ind(temp[:, i, :], temp_gfs[:, i, :])))

# Geopotential.
# print("Geopotential (GFS): " + str(stats.ttest_ind(geo[:, i, :], geo_gfs[:, i, :])))

# Relative Humidity.
# print("Relative humidity (GFS): " + str(stats.ttest_ind(rh[:, i, :], rh_gfs[:, i, :])))

# Zonal Wind.
# print("Zonal wind (GFS): " + str(stats.ttest_ind(u[:, i, :], u_gfs[:, i, :])))

# Meridional Wind.
# print("Meridional wind (GFS): " + str(stats.ttest_ind(v[:, i, :], v_gfs[:, i, :])))
