from distutils.core import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension("amsimp.backend", ["amsimp/backend.pxd"]),
    Extension("amsimp.wind", ["amsimp/wind.pxd"]),
    Extension("amsimp.water", ["amsimp/water.pxd"]),
    Extension("amsimp.dynamics", ["amsimp/dynamics.pxd"]),
    Extension("amsimp.backend", ["amsimp/backend.pyx"]),
    Extension("amsimp.wind", ["amsimp/wind.pyx"]),
    Extension("amsimp.water", ["amsimp/water.pyx"]),
    Extension("amsimp.dynamics", ["amsimp/dynamics.pyx"]),
]

setup(
    name = 'amsimp',
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
)