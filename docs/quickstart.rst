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
:class:`Backend`, :class:`Wind`, :class:`Moist`,
and :class:`Dynamics`.

In most situations, you will interact with the
:class:`Dynamics` class. Once you have chosen
a particular class, you must then decide on the grid size. 
If you are interacting with the :class:`Dynamics`
class, you must also specify the length of
forecast that will be generated. The forecast
length, in hours, must at least one hour in length, and
can have a maximum length of 168 hours. An example
of the initialisation needed for the :class:`Dynamics`
class is given below:

.. code-block:: python

    # Choose a simulation grid size combined with a class you want to use.
    # The forecast length must also be specified.
    detail = amsimp.Dynamics(
        delta_latitude=10, delta_longitude=10, forecast_length=72
    )

In this tutorial, I will show you how to
generate a ensemble forecast prediction for
the next 3 days,  with the recurrent neural
network enabled. Following which, the output
will be saved as a file on the machine.

.. code-block:: python

    # Import package.
    import amsimp

    # Initialise class.
    detail = amsimp.Dynamics(
        delta_latitude=10,
        delta_longitude=10,
        forecast_length=72,
        efs=True,
        ai=True
    )

    # Generate forecast and save output.
    detail.atmospheric_prognostic_method(save_file=True)
