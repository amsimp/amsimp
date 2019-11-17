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
level of simulation detail. You can choose
a level of simulation detail between 1 and
5, as seen below:

.. code-block:: python

    # Choose a level of simulation detail
    # combined with a class you want to use.
    detail = amsimp.Dynamics(3)

In this tutorial, I will show you how to
get a visualisation of a rudimentary
simulation of tropospheric and stratsopheric
dynamics on a synoptic scale. To do this,
you can use the code below:

.. code-block:: python

    # Visualisation of a rudimentary
    # simulation of tropospheric and stratsopheric
    # dynamics on a synoptic scale.
    detail.simulate()

.. image:: https://github.com/amsimp/amsimp/raw/master/images/september/dynamics.png
  :width: 90%
  :align: center
  :alt: Visualisation of a rudimentary simulation of tropospheric and stratsopheric dynamics on a synoptic scale.
