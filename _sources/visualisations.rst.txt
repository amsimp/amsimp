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
    state = amsimp.Wind()

Longitude - Latitude Contour Plots
----------------------------------

The default altitude at which these longitude - latitude contour plots
are generated is at a pressure surface of 1000 hectopascals, however,
a particular pressure surface, in hectopascals, can be specified the
majority of the time by doing the following:

.. code-block:: python

    # Specify a particular pressure surface, in hectopascals.
    state.method(psurface=500)
    # Pressire Surface (hPa) ^ 

Atmospheric Density
^^^^^^^^^^^^^^^^^^^

The atmospheric density is the mass of the atmosphere per unit
volume.

.. code-block:: python

    # amsimp.Backend
    # Generate a longitude - latitude contour plot for atmospheric density
    # based on current atmospheric conditions.
    state.longitude_contourf('density', psurface=1000)

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
    state.longitude_contourf('geopotential_height', psurface=1000)

.. image:: https://github.com/amsimp/papers/raw/master/scifest-online/project-book/Graphs/contour_plots/height.png
  :width: 90%
  :align: center

Precipitable Water
^^^^^^^^^^^^^^^^^^

Precipitable water is the total atmospheric water vapor contained in a
vertical column of unit cross-sectional area extending between any two
specified levels.

.. code-block:: python

    # amsimp.Water
    # Generate a longitude - latitude contour plot for precipitable water
    # based on current atmospheric conditions.
    state.longitude_contourf('precipitable_water')

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
    state.longitude_contourf('relative_humidity', psurface=1000)

.. image:: https://github.com/amsimp/papers/raw/master/scifest-online/project-book/Graphs/contour_plots/humidity.png
  :width: 90%
  :align: center

Latitude - Pressure Contour Plots
---------------------------------

Temperature
^^^^^^^^^^^

Temperature is defined as the mean kinetic energy density of molecular
motion.

.. code-block:: python

    # amsimp.Backend
    # Generate a latitude - pressure contour plot for temperature
    # based on current atmospheric conditions.
    state.psurface_contourf('temperature')

.. image:: https://github.com/amsimp/papers/raw/master/scifest-online/project-book/Graphs/contour_plots/temp.png
  :width: 90%
  :align: center
