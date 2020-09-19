# Import dependices.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Read CSV files.
# Physical Model.
# Folder.
folder = "accuracy/amsimp/physical_model/"

# Geopotential Height.
height_physical = pd.read_csv(folder+"geopotential_height.csv")
height_physical = height_physical.values
height_physical = np.transpose(height_physical)
height_physical = height_physical[1:, :]
# Temperature
# Air Temperature.
temp_physical = pd.read_csv(folder+"temperature.csv")
temp_physical = temp_physical.values
temp_physical = np.transpose(temp_physical)
temp_physical = temp_physical[1:, :]
# Virtual Temperature.
virtualtemp_physical = pd.read_csv(folder+"virtual_temperature.csv")
virtualtemp_physical = virtualtemp_physical.values
virtualtemp_physical = np.transpose(virtualtemp_physical)
virtualtemp_physical = virtualtemp_physical[1:, :]
# Relative Humidity.
rh_physical = pd.read_csv(folder+"relative_humidity.csv")
rh_physical = rh_physical.values
rh_physical = np.transpose(rh_physical)
rh_physical = rh_physical[1:, :]
# Wind.
# Zonal Wind.
u_physical = pd.read_csv(folder+"zonal_wind.csv")
u_physical = u_physical.values
u_physical = np.transpose(u_physical)
u_physical = u_physical[1:, :]
# Meridional Wind.
v_physical = pd.read_csv(folder+"meridional_wind.csv")
v_physical = v_physical.values
v_physical = np.transpose(v_physical)
v_physical = v_physical[1:, :]

# Physical Model with RNN.
# Folder.
folder = "accuracy/amsimp/physical_model_with_rnn/"

# Geopotential Height.
height_rnn = pd.read_csv(folder+"geopotential_height.csv")
height_rnn = height_rnn.values
height_rnn = np.transpose(height_rnn)
height_rnn = height_rnn[1:, :]
# Temperature
# Air Temperature.
temp_rnn = pd.read_csv(folder+"temperature.csv")
temp_rnn = temp_rnn.values
temp_rnn = np.transpose(temp_rnn)
temp_rnn = temp_rnn[1:, :]
# Virtual Temperature.
virtualtemp_rnn = pd.read_csv(folder+"virtual_temperature.csv")
virtualtemp_rnn = virtualtemp_rnn.values
virtualtemp_rnn = np.transpose(virtualtemp_rnn)
virtualtemp_rnn = virtualtemp_rnn[1:, :]
# Relative Humidity.
rh_rnn = pd.read_csv(folder+"relative_humidity.csv")
rh_rnn = rh_rnn.values
rh_rnn = np.transpose(rh_rnn)
rh_rnn = rh_rnn[1:, :]
# Wind.
# Zonal Wind.
u_rnn = pd.read_csv(folder+"zonal_wind.csv")
u_rnn = u_rnn.values
u_rnn = np.transpose(u_rnn)
u_rnn = u_rnn[1:, :]
# Meridional Wind.
v_rnn = pd.read_csv(folder+"meridional_wind.csv")
v_rnn = v_rnn.values
v_rnn = np.transpose(v_rnn)
v_rnn = v_rnn[1:, :]

# Physical Model with EPS.
# Folder.
folder = "accuracy/amsimp/physical_model_with_efs/"

# Geopotential Height.
height_efs = pd.read_csv(folder+"geopotential_height.csv")
height_efs = height_efs.values
height_efs = np.transpose(height_efs)
height_efs = height_efs[1:, :]
# Temperature
# Air Temperature.
temp_efs = pd.read_csv(folder+"temperature.csv")
temp_efs = temp_efs.values
temp_efs = np.transpose(temp_efs)
temp_efs = temp_efs[1:, :]
# Virtual Temperature.
virtualtemp_efs = pd.read_csv(folder+"virtual_temperature.csv")
virtualtemp_efs = virtualtemp_efs.values
virtualtemp_efs = np.transpose(virtualtemp_efs)
virtualtemp_efs = virtualtemp_efs[1:, :]
# Relative Humidity.
rh_efs = pd.read_csv(folder+"relative_humidity.csv")
rh_efs = rh_efs.values
rh_efs = np.transpose(rh_efs)
rh_efs = rh_efs[1:, :]
# Wind.
# Zonal Wind.
u_efs = pd.read_csv(folder+"zonal_wind.csv")
u_efs = u_efs.values
u_efs = np.transpose(u_efs)
u_efs = u_efs[1:, :]
# Meridional Wind.
v_efs = pd.read_csv(folder+"meridional_wind.csv")
v_efs = v_efs.values
v_efs = np.transpose(v_efs)
v_efs = v_efs[1:, :]

# OpenWeatherAPI.
# Folder.
folder = "accuracy/openweatherapi/error/"

# Temperature
# Air Temperature.
temp_owa = pd.read_csv(folder+"temperature.csv")
temp_owa = temp_owa.values
temp_owa = np.transpose(temp_owa)
temp_owa = temp_owa[1:, :]
# Virtual Temperature.
virtualtemp_owa = pd.read_csv(folder+"virtual_temperature.csv")
virtualtemp_owa = virtualtemp_owa.values
virtualtemp_owa = np.transpose(virtualtemp_owa)
virtualtemp_owa = virtualtemp_owa[1:, :]
# Relative Humidity.
rh_owa = pd.read_csv(folder+"humidity.csv")
rh_owa = rh_owa.values
rh_owa = np.transpose(rh_owa)
rh_owa = rh_owa[1:, :]
# Wind.
# Zonal Wind.
u_owa = pd.read_csv(folder+"zonal_wind.csv")
u_owa = u_owa.values
u_owa = np.transpose(u_owa)
u_owa = u_owa[1:, :]
# Meridional Wind.
v_owa = pd.read_csv(folder+"meridional_wind.csv")
v_owa = v_owa.values
v_owa = np.transpose(v_owa)
v_owa = v_owa[1:, :]

# Plotting.
# Function.
types = "AMSIMP Physical Model", "AMSIMP Physical Model with RNN", "AMSIMP Physical Model with EPS", "OpenWeatherAPI"
def plot(x, y1, y2, y3, y4, title, metric):
    plt.plot(x, y1, linestyle="-")
    plt.scatter(x, y1, label=types[0])
    plt.plot(x, y2, linestyle="--")
    plt.scatter(x, y2, label=types[1])
    plt.plot(x, y3, linestyle=":")
    plt.scatter(x, y3, label=types[2])
    plt.plot(x, y4, linestyle="-.")
    plt.scatter(x, y4, label=types[3])
    if metric == 'Mean Absolute Scaled Error':
        plt.plot([x.min(), x.max()], [1, 1], linestyle='dashdot')
    plt.xlabel("Forecast Period (Hours)")
    plt.ylabel(metric)
    plt.title(title+" "+metric)
    plt.legend(loc=0)
    folder = title.lower()
    folder = folder.replace(" ", "_")
    folder = "accuracy/plots/"+folder+"/"
    filename = metric.lower()
    filename = filename.replace(" ", "_")
    plt.savefig(folder+filename, dpi=300)
    plt.close()

def label_decide(num):
    if i == 0:
        label = 'Forecast Bias'
    elif i == 1:
        label = 'Mean Absolute Error'
    elif i == 2:
        label = 'Mean Squared Error'
    elif i == 3:
        label = 'Root Mean Squared Error'
    elif i == 4:
        label = 'Mean Absolute Percentage Error'
    elif i == 5:
        label = 'Mean Absolute Scaled Error'
    return label

indices = np.linspace(0, 120, 21)

# Temperature.
# Air Temperature.
for i in range(len(temp_physical)):
    metric = label_decide(i)
    title = "Air Temperature"
    if i != 4: 
        plot(
            indices[1:-1], temp_physical[i, 1:-1], temp_rnn[i, 1:-1], temp_efs[i, 1:-1], temp_owa[i, 1:], title, metric
        )
    else:
        plot(
            indices[1:-1], (temp_physical[i, 1:-1] * 100), (temp_rnn[i, 1:-1] * 100), (temp_efs[i, 1:-1] * 100), (temp_owa[i, 1:] * 100), title, metric
        )
# Virtual Temperature.
for i in range(len(virtualtemp_physical)):
    metric = label_decide(i)
    title = "Virtual Temperature"
    if i != 4: 
        plot(
            indices[1:-1], virtualtemp_physical[i, 1:-1], virtualtemp_rnn[i, 1:-1], virtualtemp_efs[i, 1:-1], virtualtemp_owa[i, 1:], title, metric
        )
    else:
        plot(
            indices[1:-1],
            (virtualtemp_physical[i, 1:-1] * 100),
            (virtualtemp_rnn[i, 1:-1] * 100),
            (virtualtemp_efs[i, 1:-1] * 100),
            (virtualtemp_owa[i, 1:] * 100),
            title,
            metric
        )

# Relative Humidity.
for i in range(len(rh_physical)):
    metric = label_decide(i)
    title = "Relative Humidity"
    if i != 4: 
        plot(
            indices[1:-1], rh_physical[i, 1:-1], rh_rnn[i, 1:-1], rh_efs[i, 1:-1], rh_owa[i, 1:], title, metric
        )
    else:
        plot(
            indices[1:-1], (rh_physical[i, 1:-1] * 100), (rh_rnn[i, 1:-1] * 100), (rh_efs[i, 1:-1] * 100), (rh_owa[i, 1:] * 100), title, metric
        )

# Wind.
# Zonal Wind.
for i in range(len(u_physical)):
    metric = label_decide(i)
    title = "Zonal Wind"
    if i != 4: 
        plot(
            indices[1:-1], u_physical[i, 1:-1], u_rnn[i, 1:-1], u_efs[i, 1:-1], u_owa[i, 1:], title, metric
        )
    else:
        pass
# Meridional Wind.
for i in range(len(v_physical)):
    metric = label_decide(i)
    title = "Meridional Wind"
    if i != 4: 
        plot(
            indices[1:-1], v_physical[i, 1:-1], v_rnn[i, 1:-1], v_efs[i, 1:-1], v_owa[i, 1:], title, metric
        )
    else:
        pass

# Performance Benchmark.
schemes = ["Physical Model", "Physical Model with RNN", "Physical Model with EPS"]
schemes_pos = [i for i, _ in enumerate(schemes)]
times = [3302.745843887329, 34511.096163749695, 49559.57580732262]
times = np.asarray(times)
times_per_time_step = [
    3302.745843887329 / (60**2),
    34511.096163749695 / (60**2),
    49559.57580732262 / (15 * (60**2))
]
ratio = times / (60 * 60 * 24 * 5)

# Bar Charts.
# Times.
folder = "performance/plots/"
plt.style.use('ggplot')
plt.bar(schemes_pos, times)
plt.xlabel("Scheme")
plt.ylabel("Runtime (s)")
plt.title("Runtime of the Various AMSIMP Schemes")
plt.xticks(schemes_pos, schemes, fontsize=8)
plt.savefig(folder+"runtime", dpi=300)
plt.close()

# Runtimes per time step.
plt.style.use('ggplot')
plt.bar(schemes_pos, times_per_time_step)
plt.xlabel("Scheme")
plt.ylabel("Runtime (s / time step)")
plt.title("Runtime of the Various AMSIMP Schemes Per Time Step")
plt.xticks(schemes_pos, schemes, fontsize=8)
plt.savefig(folder+"runtime_per_timestep", dpi=300)
plt.close()

# Runtime to forecast length ratio.
plt.style.use('ggplot')
plt.bar(schemes_pos, ratio)
plt.xlabel("Scheme")
plt.ylabel("Runtime - Forecast Length")
plt.title("Runtime - Forecast Length Ratio of the Various AMSIMP Schemes", fontsize=12)
plt.xticks(schemes_pos, schemes, fontsize=8)
plt.savefig(folder+"ratio", dpi=300)
plt.close()

# Statistical analysis.
# Determine if the RNN leads to an increase in accuracy.
# Metric (MSE).
i = 2
# Temperature.
print("Air Temperature (RNN): " + str(stats.ttest_ind(temp_physical[i, 1:], temp_rnn[i, 1:])))

# Virtual Temperature.
print("Virtual Temperature (RNN): " + str(stats.ttest_ind(virtualtemp_physical[i, 1:], virtualtemp_rnn[i, 1:])))

# Relative Humidity.
print("Relative Humidity (RNN): " + str(stats.ttest_ind(rh_physical[i, 1:], rh_rnn[i, 1:])))

# Geopotential Height.
print("Geopotential Height (RNN): " + str(stats.ttest_ind(height_physical[i, 1:], height_efs[i, 1:])))

# Zonal Wind.
print("Zonal Wind (RNN): " + str(stats.ttest_ind(u_physical[i, 1:], u_rnn[i, 1:])))

# Meridional Wind.
print("Meridional Wind (RNN): " + str(stats.ttest_ind(v_physical[i, 1:], v_rnn[i, 1:])))

print(" ")

# Determine if the EPS leads to an increase in accuracy.
# Temperature.
print("Air Temperature (EPS): " + str(stats.ttest_ind(temp_physical[i, 1:], temp_efs[i, 1:])))

# Virtual Temperature.
print("Virtual Temperature (EPS): " + str(stats.ttest_ind(virtualtemp_physical[i, 1:], virtualtemp_efs[i, 1:])))

# Relative Humidity.
print("Relative Humidity (EPS): " + str(stats.ttest_ind(rh_physical[i, 1:], rh_efs[i, 1:])))

# Geopotential Height.
print("Geopotential Height (EPS): " + str(stats.ttest_ind(height_physical[i, 1:], height_efs[i, 1:])))

# Zonal Wind.
print("Zonal Wind (EPS): " + str(stats.ttest_ind(u_physical[i, 1:], u_efs[i, 1:])))

# Meridional Wind.
print("Meridional Wind (EPS): " + str(stats.ttest_ind(v_physical[i, 1:], v_efs[i, 1:])))

print("")
# Determine if the OWA leads to a significantly significant increase in accuracy.
# Temperature.
print("Air Temperature (OWA): " + str(stats.ttest_ind(temp_rnn[i, 1:], temp_owa[i, 1:])))

# Virtual Temperature.
print("Virtual Temperature (OWA): " + str(stats.ttest_ind(virtualtemp_rnn[i, 1:], virtualtemp_owa[i, 1:])))

# Relative Humidity.
print("Relative Humidity (OWA): " + str(stats.ttest_ind(rh_physical[i, 1:], rh_owa[i, 1:])))

# Zonal Wind.
print("Zonal Wind (OWA): " + str(stats.ttest_ind(u_physical[i, 1:], u_owa[i, 1:])))

# Meridional Wind.
print("Meridional Wind (OWA): " + str(stats.ttest_ind(v_physical[i, 1:], v_owa[i, 1:])))

print("")
# Print the median of the mean absolute scaled error.
# Geopotential Height.
mean_height = np.mean(height_rnn[-1, 1:-1]) + np.mean(height_efs[-1, 1:-1]) + np.mean(height_physical[-1, 1:-1])
mean_height /= 3
print("Geopotential Height (MEAN MASE): " + str(mean_height))
# Temperature.
print("Temperature (MEAN MASE): " + str(np.mean(temp_rnn[-1, 1:])))
# Virtual Temperature.
print("Virtual Temperature (MEAN MASE): " + str(np.mean(virtualtemp_rnn[-1, 1:-1])))
# Relative Humidity.
print("Relative Humidity (MEAN MASE): " + str(np.mean(rh_physical[-1, 1:-1])))
# Zonal Wind.
mean_u = np.mean(u_rnn[-1, 1:-1]) + np.mean(u_efs[-1, 1:-1]) + np.mean(u_physical[-1, 1:-1])
mean_u /= 3
print("Zonal Wind (MEAN MASE): " + str(mean_u))
# Meridional Wind.
mean_v = np.mean(v_rnn[-1, 1:-1]) + np.mean(v_efs[-1, 1:-1]) + np.mean(v_physical[-1, 1:-1])
mean_v /= 3
print("Meridional Wind (MEAN MASE): " + str(mean_v))

print("")
# Print runtime to forecast length ratio
print("Runtime to Forecast Length Ratio: " + str(ratio))
print("Runtime to Forecast Length Ratio (Usable): " + str(1-ratio))
print("Runtime to Forecast Length Ratio (Time): " + str((1-ratio) * (5 * 24)))
