"""
Determine the forecast prediction accuracy of AMSIMP.
"""
# Import dependencies.
import io
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the forecast error csv file into a variable as a Pandas DataFrame.
file = "https://raw.githubusercontent.com/amsimp/amsimp-data/master/forecast/error.csv"
s = requests.get(file).content
df = pd.read_csv(io.StringIO(s.decode("utf-8")))

# The x-component of the plots that will be generated.
days = np.array([1, 2, 3, 4])
# Utilised to calculate the mean MAPE and MdAPE of a given atmospheric parameter
# on a given day.
aggregation_functions = {"mape": "mean", "mdape": "mean", "name": "first"}

# Deals with anything related to the MAPE.
mape = df.sort_values(["name", "day"], ascending=[False, True])
mape = np.split(mape, 4)
indiviual_mape = df.sort_values(["name", "day"], ascending=[True, True])
indiviual_mape = np.split(indiviual_mape, 4)

# Deals with anything related to the MdAPE.
mdape = df.sort_values(["name", "day"], ascending=[False, True])
mdape = np.split(mdape, 4)
indiviual_mdape = df.sort_values(["name", "day"], ascending=[True, True])
indiviual_mdape = np.split(indiviual_mdape, 4)

mean = []
median = []
list_mape = []
list_mdape = []

x = 0
while x < 4:
    # Calculate the mean MAPE, and MdAPE.
    val1 = mape[x]["mape"].values
    val1 = np.mean(val1)
    val2 = mdape[x]["mdape"].values
    val2 = np.mean(val2)
    mean.append(val1)
    median.append(val2)

    # Determine the MAPE and the MdAPE of the individual atmospheric
    # parameters.
    i_mape = indiviual_mape[x]
    i_mdape = indiviual_mdape[x]

    i_mape = i_mape.groupby(df["day"]).aggregate(aggregation_functions)
    i_mape = i_mape["mape"].values
    i_mape = i_mape.tolist()
    list_mape.append(i_mape)

    i_mdape = i_mdape.groupby(df["day"]).aggregate(aggregation_functions)
    i_mdape = i_mdape["mdape"].values
    i_mdape = i_mdape.tolist()
    list_mdape.append(i_mdape)

    x += 1

# Colors of the dashed line plots, and labels for the legened of the plot
colors = ["blue", "green", "red", "orange"]
names = ["Precipitable Water", "Pressure", "Temperature", "Thickness"]

def graph(x):
    "Add SALT to the graphs."
    if x == "MAPE":
        y = "Mean"
    else:
        y = "Median"

    plt.xlabel("Forecast Day")
    plt.ylabel(y + " Absolute Percentage Error")
    plt.title("Prediction Accuracy of AMSIMP")
    plt.legend(loc=0)
    plt.savefig(x.lower() + "_graph", dpi=400)
    plt.show()


# Plot the mean MAPE and the MAPE of the individual atmospheric
# parameters.
plt.plot(days, mean, color="black", linestyle="dashed", label="AMSIMP MAPE")
x = 0
while x < 4:
    plt.plot(days, list_mape[x], color=colors[x], linestyle="dashed", label=names[x])

    x += 1
graph("MAPE")

# Plot the mean MdAPE and the MdAPE of the individual atmospheric
# parameters.
plt.plot(days, median, color="black", linestyle="dashed", label="AMSIMP MdAPE")
x = 0
while x < 4:
    plt.plot(days, list_mdape[x], color=colors[x], linestyle="dashed", label=names[x])

    x += 1
graph("MdAPE")

print("AMSIMP's MAPE: " + str(np.round(np.mean(df["mape"]), 2)) + "%")
print("AMSIMP's MdAPE: " + str(np.round(np.mean(df["mdape"]), 2)) + "%")
