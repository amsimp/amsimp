============
Installation
============

Prerequisites
-------------
To install AMSIMP, you will need to utilise Python 3.7, or later.
Preferably, you will also want to have an installation of either
Anaconda, or Miniconda on your machine. For installation instructions,
please visit `Anaconda's documentation page`_.

.. _Anaconda's documentation page: https://docs.anaconda.com/anaconda/install/

Installing from Anaconda Cloud
------------------------------
Before installation via this particular method, TensorFlow needs to be installed through the Python Package Index (PyPI)
using the following command:

.. code:: sh

   $ pip install tensorflow

AMSIMP is distributed on `Anaconda Cloud <https://anaconda.org/amsimp/amsimp>`_ and can be installed using conda:

.. code:: sh

   $ conda install -c amsimp amsimp

If you run into any issues, you can either create an issue on
`GitHub <https://github.com/amsimp/amsimp/issues>`_ or
contact us by `email <support@amsimp.com>`_.

Installing from Source
----------------------

You can also install AMSIMP from the source code. First, clone
the repository off of GitHub:

.. code:: sh

   $ git clone https://github.com/amsimp/amsimp.git && cd amsimp

You will then need to install the software requirements, you can
do this via:

.. code:: sh

   $ conda env create -f environment.yml && conda activate amsimp

Further information about the required dependencies can be found here:

- **Python** 3.7.x (https://www.python.org/)
- **NumPy** (https://numpy.org/) |
  Python package for scientific computing including a powerful N-dimensional array object.
- **Astropy** (https://www.astropy.org) |
  A Community Python Library for Astronomy.
- **Matplotlib** (https://matplotlib.org/) | 
  Python package for 2D plotting. Python package required for any graphical capabilities.
- **Cartopy** (https://scitools.org.uk/cartopy/docs/latest/index.html) |
  Cartopy is a Python package designed for geospatial data processing in order to produce maps and other geospatial data analyses.
- **Iris** (https://scitools.org.uk/iris/docs/latest/) |
  A powerful, format-agnostic, community-driven Python library for analysing and visualising Earth science data.
- **SciPy** (https://www.scipy.org/) |
  A Python package for scientific computing.
- **scikit-learn** (https://scikit-learn.org/stable/) |
  Machine Learning in Python
- **Tensorflow** 2.0 or later (https://www.tensorflow.org) |
  TensorFlow is an end-to-end open source platform for machine learning.
- **Progress** (http://github.com/verigak/progress/) |
  Easy progress reporting for Python.

Finally, you can the software via install:

.. code:: sh

   $ python setup.py install
