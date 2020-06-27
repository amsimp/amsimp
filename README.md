# AMSIMP - Simulator of Atmospheric Dynamics

![Azure DevOps builds](https://dev.azure.com/16ccasey/AMSIMP/_apis/build/status/amsimp.amsimp?branchName=master)
![Anaconda-Server Badge](https://anaconda.org/amsimp/amsimp/badges/version.svg)
![GitHub](https://img.shields.io/github/license/amsimp/amsimp.svg?style=flat-square)
![Anaconda-Server Badge](https://anaconda.org/amsimp/amsimp/badges/downloads.svg)
![GitHub last commit](https://img.shields.io/github/last-commit/amsimp/amsimp.svg?style=flat-square)

An open-source implementation to simulating atmospheric dynamics in the troposphere and the stratosphere. Read the [paper](https://github.com/amsimp/papers/raw/master/scifest-online/project-book/main.pdf).

**Features:**

* Provides a visualisation of tropospheric and stratsopheric dynamics on a synoptic scale (Motus Aeris @ AMSIMP).
* Provides the raw data of the simulation of tropospheric and stratsopheric dynamics. The forecasted values are achieved by numerically solving the Height Tendency and Advection Equations through the finite difference method.
* Provides an ensemble prediction system simulation, which can be used to determine uncertainty in a weather forecast, as well as, providing a better prediction for the future state of the atmosphere. Instead of running a atmospheric dynamical simulation once (this would be regarded as deterministic), the simulation is run many different time with slightly different initial conditions.
* Provides an option to incorpate a recurrent neural network with the traditional physical model to improve overall accuracy.
* Provides example visualisations of different atmospheric processes, an example being a contour plot of geostrophic wind, overlayed by wind vectors, with axes being transformed onto a [Nearside Projection](https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html)(a perspective view looking directly down at a point on the globe).
* Provides extensive unit support through the utilisation of the unit module within astropy.

## Installation

This package is available on [Anaconda Cloud](https://anaconda.org/amsimp/amsimp), and can be installed using conda:

```bash
$ conda install -c amsimp amsimp  
```

For more information, please [read the documentation](https://docs.amsimp.com) on the website.

<sub><sup>AMSIMP DOES NOT PROVIDE A WEATHER FORECAST! PLEASE VISIT THE WEBSITE OF YOUR LOCAL METEOROLOGICAL AGENCY FOR ACCURATE WEATHER INFORMATION. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.</sup></sub>
