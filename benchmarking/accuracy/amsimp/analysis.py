# Import dependices.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read CSV files.
# Physical Model.
# Folder.
folder = "physical_model/"

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
folder = "physical_model_with_rnn/"

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

# Physical Model with EFS.
# Folder.
folder = "physical_model_with_efs/"

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

# Plotting.
# Function.
types = "Physical Model", "Physical Model with RNN", "Physical Model with EFS"
def plot(x, y1, y2, y3, title, metric):
    plt.plot(x, y1)
    plt.scatter(x, y1, label=types[0])
    plt.plot(x, y2, linestyle="--")
    plt.scatter(x, y2, label=types[1])
    plt.plot(x, y3, linestyle="dotted")
    plt.scatter(x, y3, label=types[2])
    plt.xlabel("Forecast Period (Hours)")
    plt.ylabel(metric)
    plt.title("AMSIMP "+title+" "+metric)
    plt.legend(loc=0)
    folder = title.lower()
    folder = folder.replace(" ", "_")
    folder = "plots/"+folder+"/"
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
# Geopotential Height.
for i in range(len(height_physical)):
    metric = label_decide(i)
    title = "Geopotential Height"
    if i != 4: 
        plot(
            indices[1:-1], height_physical[i, 1:-1], height_rnn[i, 1:-1], height_efs[i, 1:-1], title, metric
        )
    else:
        pass

# Temperature.
# Air Temperature.
for i in range(len(height_physical)):
    metric = label_decide(i)
    title = "Air Temperature"
    if i != 4: 
        plot(
            indices[1:-1], temp_physical[i, 1:-1], temp_rnn[i, 1:-1], temp_efs[i, 1:-1], title, metric
        )
    else:
        plot(
            indices[1:-1], (temp_physical[i, 1:-1] * 100), (temp_rnn[i, 1:-1] * 100), (temp_efs[i, 1:-1] * 100), title, metric
        )
# Virtual Temperature.
for i in range(len(virtualtemp_physical)):
    metric = label_decide(i)
    title = "Virtual Temperature"
    if i != 4: 
        plot(
            indices[1:-1], virtualtemp_physical[i, 1:-1], virtualtemp_rnn[i, 1:-1], virtualtemp_efs[i, 1:-1], title, metric
        )
    else:
        plot(
            indices[1:-1],
            (virtualtemp_physical[i, 1:-1] * 100),
            (virtualtemp_rnn[i, 1:-1] * 100),
            (virtualtemp_efs[i, 1:-1] * 100),
            title,
            metric
        )

# Relative Humidity.
for i in range(len(rh_physical)):
    metric = label_decide(i)
    title = "Relative Humidity"
    if i != 4: 
        plot(
            indices[1:-1], rh_physical[i, 1:-1], rh_rnn[i, 1:-1], rh_efs[i, 1:-1], title, metric
        )
    else:
        plot(
            indices[1:-1], (rh_physical[i, 1:-1] * 100), (rh_rnn[i, 1:-1] * 100), (rh_efs[i, 1:-1] * 100), title, metric
        )

# Wind.
# Zonal Wind.
for i in range(len(u_physical)):
    metric = label_decide(i)
    title = "Zonal Wind"
    if i != 4: 
        plot(
            indices[1:-1], u_physical[i, 1:-1], u_rnn[i, 1:-1], u_efs[i, 1:-1], title, metric
        )
    else:
        pass
# Meridional Wind.
for i in range(len(v_physical)):
    metric = label_decide(i)
    title = "Meridional Wind"
    if i != 4: 
        plot(
            indices[1:-1], v_physical[i, 1:-1], v_rnn[i, 1:-1], v_efs[i, 1:-1], title, metric
        )
    else:
        pass
