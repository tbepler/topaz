#from distutils.core import setup
#from distutils.extension import Extension

from setuptools import setup, find_packages
from setuptools.extension import Extension

from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension('topaz.utils.picks', ['topaz/utils/picks.pyx']),
    Extension('topaz.metrics', ['topaz/metrics.pyx']),
]

setup(
    name = 'topaz',
    packages=find_packages(),
    #package_dir = {'': 'topaz'},
    entry_points = {'console_scripts': ['topaz = topaz.main:main']},
    ext_modules = cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
)
