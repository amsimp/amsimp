from setuptools import find_packages, setup
import numpy

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="amsimp",
    version="0.6.1",
    author="AMSIMP",
    author_email="support@amsimp.com",
    description="Numerical Weather Prediction using Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://amsimp.com",
    download_url="https://github.com/amsimp/amsimp.git",
    include_package_data=True,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    zip_safe=False,
)
