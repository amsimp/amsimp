# AMSIMP - Simulator of Atmospheric Dynamics

![Travis (.org)](https://img.shields.io/travis/amsimp/amsimp.svg?style=for-the-badge)
![GitHub](https://img.shields.io/github/license/amsimp/amsimp.svg?style=for-the-badge)
![GitHub stars](https://img.shields.io/github/stars/amsimp/amsimp.svg?style=for-the-badge)
![GitHub last commit](https://img.shields.io/github/last-commit/amsimp/amsimp.svg?style=for-the-badge)

An open-source implementation to simulating atmospheric dynamics in the troposphere and the stratosphere.

 **Features:**
    - Provides a visualisation of a rudimentary numerical weather prediction scheme for geostrophic wind, temperature, saturated precipitable water, and pressure thickness (Tempestas Praenuntientur @ AMSIMP).
    - Provides example visualisations of different atmospheric processes, an example being a contour plot of geostrophic wind, overlayed by wind vectors, with axes being transformed onto a [Nearside Projection](https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html) (a perspective view looking directly down at a point on the globe).
    - Provides a mathematical representation of the [COSPAR International Reference Atmosphere (CIRA-86)](https://ccmc.gsfc.nasa.gov/modelweb/atmos/cospar1.html), with values provided being temperature, geopotential height (through the Hypsometric Equation), atmospheric pressure, atmospheric density (through the Ideal Gas Law), and geostrophic wind.

## Installation
This package is available on Anaconda Cloud, and can be installed using conda:

```bash
$ pip install amsimp
```

Documentation for AMSIMP is currently unavailable, however, it is planned for public release in the near future. Please check back soon, and check an eye on [the website](amsimp.github.io)
