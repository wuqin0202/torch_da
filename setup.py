from setuptools import setup, find_packages

setup(
    name='torch-da',
    description='Domain Adaptation in Pytorch',
    version='0.1',
    packages=find_packages(include=['torch_da']),
    install_requires=[
        'spectral',
        'scipy',
        'numpy',
        'scikit-learn',
        'tabulate',
        'hdf5storage',
    ],
)
