from setuptools import setup, find_packages

setup(
    name="duplexity",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "xarray",
        "scikit-learn",
        "tqdm",
        "seaborn",  
        "scikit-image"
    ],
    author="Lexi Xu",
    author_email="lexi.xu12@gmail.com",
    description="Duplexity is a Python package that calculates validation metrics on meteorological data, including dynamical weather forecasts, AI model outputs and atmospheric reanalyses.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/lexixu19/Duplexity",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
