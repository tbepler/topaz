from setuptools import setup, find_packages

name = 'topaz-em'
version = '0.2.1'

description = 'Particle picking with positive-unlabeled CNNs'
long_description = 'Particle picking software for single particle cryo-electron microscopy using convoluational neural networks and positive-unlabeled learning.'

keywords = 'cryoEM particle-picking CNN positive-unlabeled topaz'

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

    install_requires=[
        'torch>=1.0.0',
        'torchvision',
        'numpy>=1.11',
        'pandas',
        'scikit-learn',
        'pillow>=4.2.1',
    ],
)
