from setuptools import setup, find_packages

setup(
    name = 'topaz',
    packages=find_packages(),
    #package_dir = {'': 'topaz'},
    entry_points = {'console_scripts': ['topaz = topaz.main:main']},
)
