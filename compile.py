from distutils.core import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from Cython.Compiler.Options import get_directive_defaults
import numpy

get_directive_defaults()['linetrace'] = True
get_directive_defaults()['binding'] = True

ext_modules = [
    Extension("amsimp.backend", ["amsimp/backend.pxd"]),
    Extension("amsimp.wind", ["amsimp/wind.pxd"]),
    Extension("amsimp.water", ["amsimp/water.pxd"]),
    Extension("amsimp.dynamics", ["amsimp/dynamics.pxd"]),
    Extension("amsimp.backend", ["amsimp/backend.pyx"], define_macros=[('CYTHON_TRACE', '1')]),
    Extension("amsimp.wind", ["amsimp/wind.pyx"], define_macros=[('CYTHON_TRACE', '1')]),
    Extension("amsimp.water", ["amsimp/water.pyx"], define_macros=[('CYTHON_TRACE', '1')]),
    Extension("amsimp.dynamics", ["amsimp/dynamics.pyx"], define_macros=[('CYTHON_TRACE', '1')]),
]

setup(
    name = 'amsimp',
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
)