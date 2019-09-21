# AMSIMP - Simulator of Atmospheric Dynamics

![Azure DevOps builds](https://dev.azure.com/16ccasey/AMSIMP/_apis/build/status/amsimp.amsimp?branchName=master)
![Anaconda-Server Badge](https://anaconda.org/amsimp/amsimp/badges/version.svg)
![GitHub](https://img.shields.io/github/license/amsimp/amsimp.svg?style=flat-square)
![Anaconda-Server Badge](https://anaconda.org/amsimp/amsimp/badges/downloads.svg)
![GitHub last commit](https://img.shields.io/github/last-commit/amsimp/amsimp.svg?style=flat-square)

An open-source implementation to simulating atmospheric dynamics in the troposphere and the stratosphere.

**Features:**

* Provides a visualisation of a rudimentary simulation of tropospheric and stratsopheric dynamics on a synoptic scale (Motus Aeris @ AMSIMP).
* Provides example visualisations of different atmospheric processes, an example being a contour plot of geostrophic wind, overlayed by wind vectors, with axes being transformed onto a [Nearside Projection](https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html) (a perspective view looking directly down at a point on the globe).
* Provides a mathematical representation of the [COSPAR International Reference Atmosphere (CIRA-86)](https://ccmc.gsfc.nasa.gov/modelweb/atmos/cospar1.html), with values provided being temperature, geopotential height (through the Hypsometric Equation), atmospheric pressure, atmospheric density (through the Ideal Gas Law), and geostrophic wind.

## Installation

This package is available on [Anaconda Cloud](https://anaconda.org/amsimp/amsimp), and can be installed using conda:

```bash
$ conda install -c amsimp amsimp  
```

For more information, please [read the documentation](https://amsimp.github.io) on the website.
