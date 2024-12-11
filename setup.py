import os
import re
from setuptools import setup, find_packages

# Read version from _version.py
version_file = os.path.join(os.path.dirname(__file__), 'topaz', '_version.py')
with open(version_file) as f:
    version = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M).group(1)

# Read requirements from requirements.txt
requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
with open(requirements_file) as f:
    requirements = [line.strip() for line in f if line.strip()]


name = 'topaz-em'

description = 'Particle picking with positive-unlabeled CNNs'
long_description = 'Particle picking software for single particle cryo-electron microscopy using convolutional neural networks and positive-unlabeled learning. Includes methods for micrograph denoising.'

keywords = 'cryoEM particle-picking CNN positive-unlabeled denoise topaz'

url = 'https://github.com/tbepler/topaz'

author = 'Tristan Bepler'
author_email = 'tbepler@mit.edu'

license = 'GPLv3'

setup(
    name = name,
    version=version,
    description=description,
    long_description=long_description,
    keywords=keywords,
    url=url,
    author=author,
    author_email=author_email,
    license=license,

    packages=find_packages(),
    #package_dir = {'': 'topaz'},
    entry_points = {'console_scripts': ['topaz = topaz.main:main']},
    include_package_data = True,
    python_requires='>=3.8,<=3.12',
    install_requires=requirements,
)
