from setuptools import find_packages, Extension, setup
import numpy
import sys

with open("README.md", "r") as fh:
    long_description = fh.read()

if '--use-cython' in sys.argv:
    USE_CYTHON = True
    sys.argv.remove('--use-cython')
else:
    USE_CYTHON = False

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [
    Extension("amsimp.download", ["amsimp/download"+ext], language='c', include_dirs=['amsimp/']),
    Extension("amsimp.backend", ["amsimp/backend"+ext], language='c',
    include_dirs=['amsimp/']),
    Extension("amsimp.wind", ["amsimp/wind"+ext], language='c',
    include_dirs=['amsimp/']),
    Extension("amsimp.moist", ["amsimp/moist"+ext], language='c',
    include_dirs=['amsimp/']),
    Extension("amsimp.dynamics", ["amsimp/dynamics"+ext], language='c',
    include_dirs=['amsimp/']),
    ]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(
    name="amsimp",
    ext_modules=extensions,
    include_dirs=[numpy.get_include()],
    version="1.0.0",
    author="Conor Casey",
    author_email="support@amsimp.com",
    description="Simulator for Atmospheric Dynamics",
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
