from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    Extension("amsimp.backend", ["amsimp/backend.pyx"]),
    Extension("amsimp.wind", ["amsimp/wind.pyx"]),
    Extension("amsimp.water", ["amsimp/water.pyx"]),
    Extension("amsimp.weather", ["amsimp/weather.pyx"]),
]

setup(
    name = 'amsimp',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    include_dirs=[numpy.get_include()],
)
