from setuptools import setup
from setuptools import find_packages

# Setup parameters from Google Cloud ML Engine
setup(name='trainer',
    version='0.1',
    packages=find_packages(),
    description='Example to run Keras on ML Engine',
    include_package_data=True,
    author='Rodrigo Cunha',
    author_email='rdr.cunha@gmail.com',
    license='MIT',
    install_requires=[
      'keras==2.1.6',
      'tensorflow==2.5.0',
      'h5py'
    ],
    zip_safe=False)