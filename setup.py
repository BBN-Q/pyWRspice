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
    long_description=open('README.md').read()
)
