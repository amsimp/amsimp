==============
Visualisations
==============

The software also provides additional visualisations of different
atmospheric parameters, such as geostrophic wind and precipitable
water. These visualisations can be found in the :class:`Backend`,
:class:`Wind`, :class:`Moist` classes. In this particular tutorial,
we will initialise with the :class:`Wind` class, as it also
inherits the other classes we are interested in:

.. code-block:: python

    # Initialise the Wind class.
    detail = amsimp.Wind()

Longitude - Latitude Contour Plots
----------------------------------

The default altitude at which these longitude - latitude contour plots
are generated is at a pressure surface of 1000 hectopascals, however,
a particular pressure surface, in hectopascals, can be specified the
majority of the time by doing the following:

.. code-block:: python

    # Specify a particular pressure surface, in hectopascals.
    detail.method(psurface=500)
    # Pressire Surface (hPa) ^ 

Atmospheric Density
^^^^^^^^^^^^^^^^^^^

The atmospheric density is the mass of the atmosphere per unit
volume.

.. code-block:: python

    # amsimp.Backend
    # Generate a longitude - latitude contour plot for atmospheric density
    # based on current atmospheric conditions.
    detail.longitude_contourf(2, psurface=1000)
    #                         ^--This number determines the atmospheric parameter.

.. image:: https://github.com/amsimp/papers/raw/master/scifest-online/project-book/Graphs/contour_plots/density.png
  :width: 90%
  :align: center

Geopotential Height
^^^^^^^^^^^^^^^^^^^

Geopotential Height is the height above sea level of a pressure level.
For example, if a station reports that the 500 hPa height at its
location is 5600 m, it means that the level of the atmosphere over
that station at which the atmospheric pressure is 500 hPa is 5600
meters above sea level.

.. code-block:: python

    # amsimp.Backend
    # Generate a longitude - latitude contour plot for geopotential height
    # based on current atmospheric conditions.
    detail.longitude_contourf(1, psurface=1000)
    #                         ^--This number determines the atmospheric parameter.

.. image:: https://github.com/amsimp/papers/raw/master/scifest-online/project-book/Graphs/contour_plots/height.png
  :width: 90%
  :align: center

Geostrophic Wind
^^^^^^^^^^^^^^^^

Geostrophic Wind is a theoretical wind that would result from an exact
balance between the pressure-gradient force, and the Coriolis force. This
contour plot overlayed by wind vectors, with axes being transformed onto
a `Nearside Projection <https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html>`_
(a perspective view looking directly down at a point on the globe).

.. code-block:: python

    # amsimp.Wind
    # Generate a longitude - latitude contour plot for geostrophic wind
    # based on current atmospheric conditions.
    detail.globe(psurface=1000)

.. image:: https://github.com/amsimp/papers/raw/master/scifest-online/project-book/Graphs/contour_plots/globe.png
  :width: 90%
  :align: center

The point at which you are directly look down upon can be specified by
givng the latitude and longitude lines of interest. The default point
has coordinates 53.1424, -7.6921 (Dublin City, Ireland):

.. code-block:: python

    # Specify a particular altitude, in metres.
    detail.globe(central_lat=13.17, central_long=-8.78, alt=0)
    #         Latitude (deg)---^ Longitude (deg) --^

Precipitable Water
^^^^^^^^^^^^^^^^^^

Precipitable water is the total atmospheric water vapor contained in a
vertical column of unit cross-sectional area extending between any two
specified levels.

.. code-block:: python

    # amsimp.Water
    # Generate a longitude - latitude contour plot for precipitable water
    # based on current atmospheric conditions.
    detail.water_contourf()

.. image:: https://github.com/amsimp/papers/raw/master/scifest-online/project-book/Graphs/contour_plots/precipitable_water.png
  :width: 90%
  :align: center

Relative Humidity
^^^^^^^^^^^^^^^^^

Relative Humidity is the amount of water vapour present in air expressed
as a percentage of the amount needed for saturation at the same temperature.

.. code-block:: python

    # amsimp.Backend
    # Generate a longitude - latitude contour plot for relative humidity
    # based on current atmospheric conditions.
    detail.longitude_contourf(3, psurface=1000)
    #                         ^--This number determines the atmospheric parameter.

.. image:: https://github.com/amsimp/papers/raw/master/scifest-online/project-book/Graphs/contour_plots/humidity.png
  :width: 90%
  :align: center

Latitude - Pressure Contour Plots
---------------------------------

Meridional Wind
^^^^^^^^^^^^^^^

Merdional flow is a meteorological term regarding atmospheric circulation
following a general flow pattern along longitudinal lines. A positive
merdional wind value indiciates that the wind is flowing from south to north.
Note: AMSIMP deals solely with geostrophic wind.

.. code-block:: python

    # amsimp.Wind
    # Generate a latitude - pressure contour plot for meridional wind
    # based on current atmospheric conditions.
    detail.wind_contourf(1)
    #                    ^--This number determines the atmospheric parameter.

.. image:: https://github.com/amsimp/papers/raw/master/scifest-online/project-book/Graphs/contour_plots/meridional_wind.png
  :width: 90%
  :align: center

Temperature
^^^^^^^^^^^

Temperature is defined as the mean kinetic energy density of molecular
motion.

.. code-block:: python

    # amsimp.Backend
    # Generate a latitude - pressure contour plot for temperature
    # based on current atmospheric conditions.
    detail.psurface_contourf(0)
    #                        ^--This number determines the atmospheric parameter.

.. image:: https://github.com/amsimp/papers/raw/master/scifest-online/project-book/Graphs/contour_plots/temp.png
  :width: 90%
  :align: center

Zonal Wind
^^^^^^^^^^

Zonal flow is a meteorological term regarding atmospheric circulation following
a general flow pattern along latitudinal lines. A positive zonal wind value
indiciates that the wind is flowing from east to west. Note: AMSIMP deals
solely with geostrophic wind.

.. code-block:: python

    # amsimp.Wind
    # Generate a latitude - pressure contour plot for zonal wind
    # based on current atmospheric conditions.
    detail.wind_contourf(0)
    #                    ^--This number determines the atmospheric parameter.

.. image:: https://github.com/amsimp/papers/raw/master/scifest-online/project-book/Graphs/contour_plots/zonal_wind.png
  :width: 90%
  :align: center
