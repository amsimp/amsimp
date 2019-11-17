============
Installation
============

Prerequisites
-------------
To install AMSIMP, you will need to utilise Python 3.7, or later.
Preferably, you will also want to have an installation of either
Anaconda, or Miniconda on your machine. For installation instructions,
please visit `Anaconda's documentation page`_. I strongly discourage
the use of either PyPI or building from source as some of the packages
needed for the installation of AMSIMP are extremely difficult to install
without the use of Anaconda Cloud.

.. _Anaconda's documentation page: https://docs.anaconda.com/anaconda/install/

Installing from Anaconda Cloud
------------------------------
AMSIMP is distributed on `Anaconda Cloud <https://anaconda.org/amsimp/amsimp>`_ and can be installed using conda:

.. code:: sh

   $ conda install -c amsimp amsimp

If you run into any issues, you can either create an issue on
`GitHub <https://github.com/amsimp/amsimp/issues>`_ or
contact me by `email <support@amsimp.com>`_.

Installing from PyPI
--------------------

AMSIMP is also available on `PyPI <https://pypi.org/project/amsimp/>`_ and can be installed using pip,
however, as mentioned previously this approach is strongly discouraged:

.. code:: sh

   $ pip install amsimp

Please note: you will also need to have a C compiler installed
on your machine. On macOS or Linux, you could use GCC and on
Windows, you could use MiniGW.

Installing from Source
----------------------

Finally, you can also install AMSIMP from the source code. Again, I
strongly discourage this approach. First, clone
the repository off of GitHub:

.. code:: sh

   $ git clone https://github.com/amsimp/amsimp.git && cd amsimp

You will then need to install the software requirements, you can
do this via:

.. code:: sh

   $ pip install -r requirements.txt

Please note, you may have some difficulty installing some of the
software requirements, specifically Cartopy and Proj.

Finally you can install:

.. code:: sh

   $ python setup.py install
