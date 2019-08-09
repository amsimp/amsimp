from setuptools import setup, Extension, Command, find_packages
from Cython.Distutils import build_ext
import numpy

with open("README.md", "r") as fh:
    long_description = fh.read()

ext_modules = [
    Extension("amsimp.backend", ["amsimp/backend.pyx"]),
    Extension("amsimp.wind", ["amsimp/wind.pyx"]),
    Extension("amsimp.water", ["amsimp/water.pyx"]),
    Extension("amsimp.weather", ["amsimp/weather.pyx"]),
]

setup(
    name="amsimp",
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
    version="0.2.0",
    author="Conor Casey",
    author_email="conorcaseyc@icloud.com",
    description="Simulator for Atmospheric Dynamics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://amsimp.github.io",
    download_url="https://github.com/amsimp/amsimp.git",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["scipy", "astropy", "matplotlib", "numpy", "cartopy", "pandas"],
)
