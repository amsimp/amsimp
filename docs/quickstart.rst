Getting Started
===============

Once AMSIMP is successfully installed, you can follow
this tutorial to get an example use case of the software.
AMSIMP is a Python package that is written in the
programming language of Cython, hence, you can utilise
the software by importing it into Python:

.. code-block:: python

    # Import AMSIMP.
    import amsimp

Once imported, you can utilise the four classes that
are available to you. The classes available are:
:class:`Backend`, :class:`Wind`, :class:`Water`,
and :class:`Dynamics`.

In most situations, you will interact with the
:class:`Dynamics` class. Once you have chosen
a particular class, you must then decide on a
level of simulation detail between 1 and 5. If
you are interacting with the :class:`Dynamics`
class, you must also specify the length of
forecast that will be generated. The forecast
length must at least one day in length, and
can have a maximum length of five. An example
of the initialisation needed for the :class:`Dynamics`
class is given below:

.. code-block:: python

    # Choose a level of simulation detail combined with a class you want to use.
    # The forecast length must also be specified.
    detail = amsimp.Dynamics(5, 3)
    #    Simulation detail---^  ^---Forecast length

In this tutorial, I will show you how to
get a visualisation of a rudimentary
simulation of tropospheric and stratsopheric
dynamics on a synoptic scale. To do this,
you can use the code below:

.. code-block:: python

    # Visualisation of a rudimentary simulation of tropospheric and stratsopheric
    # dynamics on a synoptic scale.
    detail.simulate()

.. image:: https://github.com/amsimp/papers/raw/master/project-book/Graphs/contour_plots/forecast.png
  :width: 90%
  :align: center
  :alt: Visualisation of a rudimentary simulation of tropospheric and stratsopheric dynamics on a synoptic scale.
