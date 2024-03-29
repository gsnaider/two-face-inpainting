from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['matplotlib==2.2.2', 'scikit-image==0.14.1', 'h5py==2.8.0', 'fs==2.1.2', 'fs-gcsfs==0.3.0']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)
