# AMSIMP - Model for Atmospheric Dynamics

![Travis (.org)](https://img.shields.io/travis/amsimp/amsimp.svg?style=for-the-badge)
![PyPI](https://img.shields.io/pypi/v/amsimp.svg?style=for-the-badge)
![GitHub](https://img.shields.io/github/license/amsimp/amsimp.svg?style=for-the-badge)
![GitHub stars](https://img.shields.io/github/stars/amsimp/amsimp.svg?style=for-the-badge)
![GitHub last commit](https://img.shields.io/github/last-commit/amsimp/amsimp.svg?style=for-the-badge)

An open source implementation to modelling atmospheric dynamics in the troposphere and the stratosphere. Read the [paper](https://github.com/amsimp/papers/raw/master/SciFest/Project%20Book/main.pdf)

## Features
	- Written in Python, and by extension, it can run on macOS, Linux, and Windows.
	- Simulation of various components of atmospheric dynamics.
	- Includes a visualisation of said components.
	- Provides a simulation of prevailing wind vectors onto a simulated globe.

## Installation
In order to install this package, you will need to install the latest version of [Anaconda 3](https://www.anaconda.com/distribution/). You will need this in order to install some of the modules that are not available on PyPi.

Following the installation of Anaconda, you will need to manually install a package called, Cartopy, as it is not currently available on PyPi. This can be done using the following line of code:

```bash
$ conda install -c conda-forge cartopy 
```

Now, you can simply install the package using pip:

```bash
$ pip install amsimp 
```

### macOS Users
Once Anaconda 3 is installed, you will need to install python.app from Anaconda, as Python needs to be installed as a framework in order for Matplotlib to properly function with this package. This can be done using the following line of code:

```bash
$ conda install -c anaconda python.app 
```

In order to import the package, you must use python.app. This can be run using the following line of code:

```bash
$ pythonw
```

Following which, you can use the package as normal:

```python
import amsimp
```

### An Example Use Case
```python
###############################################
#Simulation of Prevailing Winds (At Sea Level)#
###############################################

import amsimp

detail = amsimp.Dynamics(4) #Defines the level of detail that will be used in the simulation.

detail.simulate() #Simulates ...
```

![Simulation](https://github.com/amsimp/papers/raw/master/SciFest/Images/globe.png)