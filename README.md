# MONDPMesh
Python N-body code made for MOND simulations.

This code uses an altered particle mesh method to solve the Poisson equation of the AQUAL version of MOND. It was made for my Bachelor's thesis. The thesis can be found at (http://resolver.tudelft.nl/uuid:ad94e143-0ce4-4b17-8a72-54ab9f656236).


The example file contains the complete code together with an example simulation of a two body system. This file can be immediately run when the required libraries are installed, and will plot the trajectories of the bodies with the total energy. Both of these are plotted together with exact formula for these quantities. 

The Clean file contains the complete code without anything else. You can create a particlelist object and run simulations yourself. This file also includes comments on what the code does.

Please message me if you find any errors or bugs.

Note: One error that has been found is that the information of the last time step is not saved.



Required libraries:
Numpy, math, itertools, these are all included in base Python.
Pyfftw, which needs the C FFTW library. Its documentation can be found in https://pyfftw.readthedocs.io/en/latest/. 
Cupy, see: https://docs.cupy.dev/en/stable/install.html
Numba, see: https://numba.pydata.org/

A C++ version of the code has also been included, which works in the same way as the Python faster, but it is quite a bit faster and more memory efficient. The Python version is, however, easier to install and modify, and has some more example systems implemented. This version contains an example two-body simulation. The C++ version of the code needs the Eigens and either the FFTW or MKL libraries to work. The MKL and FFTW libraries use the same code. 


