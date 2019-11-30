"""
Statistical analysis of benchmark results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import amsimp
from scipy.stats import ttest_ind

# Python Data.
mac_pythondata = pd.read_csv("python/macbook_benchmark.csv")
mac_python = mac_pythondata.sort_values(
    ["detail_level", "time"], ascending=[True, True]
)
mac_python = np.split(mac_python, 5)

windows_pythondata = pd.read_csv('python/windows_benchmark.csv')
windows_python = windows_pythondata.sort_values(
    ['detail_level', 'time'], ascending = [True, True]
)
windows_python = np.split(windows_python, 5)

vm_pythondata = pd.read_csv('python/google_benchmark.csv')
vm_python = vm_pythondata.sort_values(
    ['detail_level', 'time'], ascending=[True, True]
)
vm_python = np.split(vm_python, 5)

# Cython Data.
mac_cythondata = pd.read_csv("cython/macbook_benchmark.csv")
mac_cython = mac_cythondata.sort_values(
    ["detail_level", "time"], ascending=[True, True]
)
mac_cython = np.split(mac_cython, 5)

windows_cythondata = pd.read_csv('cython/windows_benchmark.csv')
windows_cython = windows_cythondata.sort_values(
    ['detail_level', 'time'], ascending = [True, True]
)
windows_cython = np.split(windows_cython, 5)

vm_cythondata = pd.read_csv('cython/google_benchmark.csv')
vm_cython = vm_cythondata.sort_values(
    ['detail_level', 'time'], ascending=[True, True]
)
vm_cython = np.split(vm_cython, 5)

avg_python = []
avg_cython = []

x = 0
while x < 5:
    data_python = (
        mac_python[x]["time"] + vm_python[x]['time'] + windows_python[x]['time']
    ) / 3
    data_cython = (
        mac_cython[x]["time"] + vm_cython[x]['time'] + windows_cython[x]['time']
    ) / 3
    mean_python = data_python.mean()
    mean_cython = data_cython.mean()
    avg_python.append(mean_python)
    avg_cython.append(mean_cython)
    x += 1

avg_python = np.asarray(avg_python)
avg_cython = np.asarray(avg_cython)

np.set_printoptions(suppress=True)

print("Mean (Python): " + str(avg_python.round(5)))
print("Mean (Cython): " + str(avg_cython.round(5)))

speed_increase = (avg_python - avg_cython) / avg_cython
speed_increase = np.abs(speed_increase)

print("Speed Increase: " + str(speed_increase.round(5)))

# Plot benchmark results.
detail_level = np.array([1, 2, 3, 4, 5])

detail = amsimp.Backend(5)
guess = [3, -0.0010, -3.1968]
c_python = curve_fit(detail.fit_method, detail_level, avg_python, guess)
abc_python = c_python[0]
c_cython = curve_fit(detail.fit_method, detail_level, avg_cython, guess)
abc_cython = c_cython[0]

x = np.linspace(1, 5, num=1000)
python_line = detail.fit_method(
    x, abc_python[0], abc_python[1], abc_python[2]
)
cython_line = detail.fit_method(
    x, abc_cython[0], abc_cython[1], abc_cython[2]
)

plt.scatter(detail_level, avg_python, color='blue')
plt.scatter(detail_level, avg_cython, color='orange')
plt.plot(x, python_line, label='Python')
plt.plot(x, cython_line, label='Cython')
plt.xlabel('Level of Detail')
plt.ylabel('Execution Time ($s$)')
plt.legend(loc=0)
plt.savefig('results', dpi=350)
plt.show()

# Calculate p-value based on Welsh's t-test.
mac_pythondata = np.sort(mac_pythondata["time"].values)
mac_cythondata = np.sort(mac_cythondata["time"].values)
windows_pythondata = np.sort(windows_pythondata["time"].values)
windows_cythondata = np.sort(windows_cythondata["time"].values)
vm_pythondata = np.sort(vm_pythondata["time"].values)
vm_cythondata = np.sort(vm_cythondata["time"].values)

def ttest(a, b):
    t_stat, p_value = ttest_ind(a, b, equal_var=False)
    return t_stat, p_value

p_values = []

p = ttest(mac_pythondata, mac_cythondata)
p = p[1]
p_values.append(p)

p = ttest(windows_pythondata, windows_cythondata)
p = p[1]
p_values.append(p)

p = ttest(vm_pythondata, vm_cythondata)
p = p[1]
p_values.append(p)

p_value = np.mean(p_values)
print(p_value)
