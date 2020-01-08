# pyWRspice

#### Copyright (c) 2019 Raytheon BBN Technologies - Quantum Group


PyWRspice is Python wrapper for [WRspice](http://www.wrcad.com/), a SPICE simulation engine modified by Whiteley Research (WR) featuring Josephson junctions.

#### Features:
- Run WRspice simulation and retrieve output data
- Multiprocessing for running multiple simulations in parallel
- Adaptive run for parameter optimization
- Run WRspice remotely on an SSH server
- Programmatically build circuit scripts

#### In the package:
- ```simulation.py```: Simulate a complete or parametric WRspice script via WRspice simulator.
- ```remote.py```: Execute WRspice simulation remotely on an SSH server.
- ```script.py```: Programmatically construct a WRspice script.

#### Install pyWRspice wrapper
Run ```python setup.py install``` from the main directory.


#### Install WRspice simulation engine

Get and install the software [here](http://www.wrcad.com/xictools/index.html).


*Important* : Make sure to take note where the executable ```wrspice``` is on your machine.

On Unix, it is likely "/usr/local/xictools/bin/wrspice".

On Windows, "C:/usr/local/xictools/bin/wrspice.bat".


Check out ```examples/Tutorial.ipynb``` for basic usage, and ```examples/Remote.ipynb``` for simulation on SSH server.
