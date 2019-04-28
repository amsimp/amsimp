import pandas as pd 
from statistics import stdev
import numpy as np 

mac = pd.read_csv('macos_benchmark.csv')
mac = mac.sort_values(['detail_level', 'time'], ascending = [True, True])
mac = np.split(mac, 5)

windows = pd.read_csv('windows_benchmark.csv')
windows = windows.sort_values(['detail_level', 'time'], ascending = [True, True])
windows = np.split(windows, 5)

vm = pd.read_csv('vm_benchmark.csv')
vm = vm.sort_values(['detail_level', 'time'], ascending = [True, True])
vm = np.split(vm, 5)

linux = pd.read_csv('linux_benchmark.csv')
linux = linux.sort_values(['detail_level', 'time'], ascending = [True, True])
linux = np.split(linux, 5)

x = 0

sd = []
avg = []

while x < 5:
	data = mac[x]['time'] + windows[x]['time'] + vm[x]['time'] + linux[x]['time']
	std = stdev(data)
	mean_x = data.mean()
	sd.append(std)
	avg.append(mean_x)
	print("Progress: " + str(x) + " round")
	x += 1

print(sd)
print(avg)

sd = np.asarray(sd)
avg = np.asarray(avg)

coefficient_of_variation = sd / avg

print(coefficient_of_variation)