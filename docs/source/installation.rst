=========================
Installation instructions
=========================


.. note::

    We have not yet made Duplexity pip-installable, as the
    project is still in the early development stages. This means the code is 
    constantly changing, and we currently don't guarantee backwards compatibility 
    when changes are made. Once a stable release of Duplexity is available, we will
    upload Duplexity to PyPI to make it pip-installable.


Prerequisites
-------------
Ensure you have the following prerequisites installed on your system:
* `Python (version 3.7 or later) <http://www.python.org/>`_ (lower versions may work but are not tested).
* Git
* `numpy <http://www.numpy.org/>`_
* `pandas <http://pandas.pydata.org/>`_
* `Xarray <http://xarray.pydata.org/en/stable/>`_
* `matplotlib <http://matplotlib.org/>`_





Install Duplexity from `source <https://github.com/lexixu19/Duplexity>`_
----------------------------------------------------------------------------




Clone the directory
~~~~~~~~~~~~~~~~~~~

Navigate to the directory you would like to clone the Duplexity repository into, and clone from GitHub:
.. code-block:: console

    git clone https://github.com/lexixu19/duplexity.git



Activate your environment
~~~~~~~~~~~~~~~~~~~~~~~~~

Create and/or activate the `conda` or `pip` environment you would like to use Duplexity within. For example:

.. code-block:: console

    conda create --name duplexity
    conda activate duplexity

Install Duplexity
~~~~~~~~~~~~~~~~~
Ensure you are in the highest level of the Duplexity directory on your local system:
.. code-block:: console

    cd duplexity

You should be able to see `setup.py` in this directory. Run the following command to install Duplexity in your environment:


.. code-block:: console

    pip install .



Note: if you are a contributor or editor of the Duplexity environment, you should use `pip install -e` . to allow you to make edits which are immediately reflected when you import locally.


Importing Duplexity
You should now be able to import Duplexity when running your environment! Try it out:

.. code-block:: python

    import duplexity


Test your installation
~~~~~~~~~~~~~~~~~~~~~~
To test that Duplexity has been installed correctly, run the following command in your Python environment:

.. code-block:: python

    import duplexity.deterministic_score as ds
    import numpy as np
    x = np.array([5, 2, 7])
    y = np.array([8, 2, 6])

    ds.mean_squared_error(x, y)

