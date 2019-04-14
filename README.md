# AMSIMP - Model for Atmospheric Dynamics

![Travis (.org)](https://img.shields.io/travis/amsimp/amsimp.svg?style=for-the-badge)
![PyPI](https://img.shields.io/pypi/v/amsimp.svg?style=for-the-badge)
![GitHub](https://img.shields.io/github/license/amsimp/amsimp.svg?style=for-the-badge)
![GitHub stars](https://img.shields.io/github/stars/amsimp/amsimp.svg?style=for-the-badge)
![GitHub last commit](https://img.shields.io/github/last-commit/amsimp/amsimp.svg?style=for-the-badge)

An open source implementation to modelling atmospheric dynamics in the troposphere and the stratosphere.

## Installation
In order to install this package, you will need to install the latest version of     [Anaconda 3](https://www.anaconda.com/distribution/). You will need this in order to install some of the modules that are not available on PyPi.

Once Anaconda 3 is installed, you will need to install python.app from Anaconda, as Python needs to be installed as a framework in order for Matplotlib to properly function with this package. This can be done using the following line of code:

```bash
$ conda install -c anaconda python.app 
```

Following which, you will need to manually install a package called, proj4, as it is not currently available on PyPi. This package is needed to install Cartopy. This can be done using the following line of code:

```bash
$ conda install -c conda-forge proj4 
```

Now, you can simply install the package using pip:

```bash
$ pip install amsimp 
```

## Usage
In order to import the package, you must use python.app. This can be run using the following line of code:

```bash
$ pythonw
```

Following which, you can use the package as normal:

```python
import amsimp
```