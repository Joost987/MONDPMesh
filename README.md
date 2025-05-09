# MONDPMesh
Python N-body code made for MOND simulations.

This code uses an altered particle mesh method to solve the Poisson equation of the AQUAL version of MOND and includes the external field effect. It was made for my Bachelor's thesis, which can be found at (https://resolver.tudelft.nl/uuid:c0132153-01e6-460c-83e0-fd7b625719d7). It is based on code made by Joost de Nijs, whom's thesis can be found at (http://resolver.tudelft.nl/uuid:ad94e143-0ce4-4b17-8a72-54ab9f656236). The method was developed in a paper by P.M. Visser, S.W.H. Eijt and J.V. de Nijs: 	https://doi.org/10.1051/0004-6361/202347830. Note that the code used in the paper is the code in Joost's branch. The code in this branch is an improved version.


Please message me if you find any errors or bugs.



Required libraries:
Numpy, math, itertools, these are all included in base Python.
Cupy, see: https://docs.cupy.dev/en/stable/install.html
Numba, see: https://numba.pydata.org/


![Energy figure](https://github.com/Joost987/MONDPMesh/blob/CodeJKoster/Energy.png)
![Orbit figure](https://github.com/Joost987/MONDPMesh/blob/CodeJKoster/Orbit.png)
