"""
Statistical analysis of benchmark results.
"""

from statistics import stdev
import numpy as np
import pandas as pd

mac = pd.read_csv("macbook_benchmark.csv")
mac = mac.sort_values(["detail_level", "time"], ascending=[True, True])
mac = np.split(mac, 5)

#windows = pd.read_csv('windows_benchmark.csv')
#windows = windows.sort_values(['detail_level', 'time'], ascending = [True, True])
#windows = np.split(windows, 5)

vm = pd.read_csv('google_benchmark.csv')
vm = vm.sort_values(['detail_level', 'time'], ascending=[True, True])
vm = np.split(vm, 5)

linux = pd.read_csv('linux_benchmark.csv')
linux = linux.sort_values(['detail_level', 'time'], ascending = [True, True])
linux = np.split(linux, 5)

sd = []
avg = []

x = 0
while x < 5:
    data = (mac[x]["time"] + vm[x]['time'] + linux[x]['time']) / 3 # + windows[x]['time']) / 4
    std = stdev(data)
    mean_x = data.mean()
    sd.append(std)
    avg.append(mean_x)
    x += 1

sd = np.asarray(sd)
avg = np.asarray(avg)

np.set_printoptions(suppress=True)

print("Standard deviation: " + str(sd.round(5)))
print("Mean: " + str(avg.round(5)))

coefficient_of_variation = sd / avg

print("Coefficient of variation: " + str(coefficient_of_variation.round(5)))
