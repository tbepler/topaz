from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension('topaz.utils.picks', ['topaz/utils/picks.pyx']),
    Extension('topaz.metrics', ['topaz/metrics.pyx']),
]

setup(
    ext_modules = cythonize(ext_modules)
    , include_dirs=[numpy.get_include()]
)
