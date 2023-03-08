from distutils.core import setup
import numpy as np

setup(
    name='pyWRspice',
    version='0.1',
    author='Raytheon BBN Technologies - QEC Group',
    packages=["pyWRspice"],
    package_data={'pyWRspice':['data/*.csv','data/*.py']},
    scripts=[],
    url='https://github.com/BBN-Q/pyWRspice',
    license='MIT License',
    description='Python wrapper for WRspice circuit simulation',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy >= 1.12.1",
        "scipy >= 0.17.1",
        "pandas >= 1.3.5",
        "paramiko >= 2.9.5"
    ],
)
