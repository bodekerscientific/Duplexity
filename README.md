![Duplexity Logo](./logo_v1.png)

Duplexity is a Python package that calculates validation metrics on meteorological data, including dynamical weather forecasts, AI model outputs and atmospheric reanalyses. 

## Installation

We have not yet made Duplexity pip-installable, as the project is still in the early development stages. This means the code is constantly changing, and we currently don't guarantee backwards compatibility when changes are made. Once a stable release of Duplexity is available, we will upload Duplexity to PyPI to make it pip-installable.

### Follow these steps to install Duplexity:

#### Prerequisites
Ensure you have the following prerequisites installed on your system:

 - Python (version 3.7 or later)
 - Git

#### Clone the directory

Navigate to the directory you would like to clone the Duplexity repository into, and clone from GitHub:

```
git clone https://github.com/lexixu19/duplexity.git
```
#### Activate your environment
Create and/or activate the [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [pip](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) environment you would like to use Duplexity within. For example:

```
conda create --name duplexity
conda activate duplexity
```
#### Install Duplexity
Ensure you are in the highest level of the Duplexity directory on your local system:

```
cd duplexity
```
You should be able to see `pyproject.toml` in this directory. Run the following command to install Duplexity in your environment:

```
pip install .
```

Note: if you are a contributor or editor of the Duplexity environment, you should use `pip install -e .` to allow you to make edits which are immediately reflected when you import locally.

## Contributing

We welcome contributions of new or desired metrics to Duplexity. If there are metrics you would like to see added to Duplexity, please feel free to look at our [contributing guide](https://github.com/bodekerscientific/Duplexity/blob/contributing_guide/CONTRIBUTING.md) or open an issue to request a new metric.

## License

[MIT](https://choosealicense.com/licenses/mit/)
