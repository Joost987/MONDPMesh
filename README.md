# MONDPMesh
Python N-body code made for MOND simulations.


This code uses an altered particle mesh method to solve the Poisson equation of the AQUAL version of MOND and includes the external field effect. It was made for my Bachelor's thesis, which can be found at (https://resolver.tudelft.nl/uuid:c0132153-01e6-460c-83e0-fd7b625719d7). It is based on code made by Joost de Nijs, whom's thesis can be found at (http://resolver.tudelft.nl/uuid:ad94e143-0ce4-4b17-8a72-54ab9f656236). The method was developed in a paper by P.M. Visser, S.W.H. Eijt and J.V. de Nijs: 	https://doi.org/10.1051/0004-6361/202347830. Note that the code used in the paper is the code in Joost's branch. The code in this branch is an improved version.


Please message me if you find any errors or bugs.

# Method

The code is a particle mesh code, except when we want to solve for the AQUAL-MOND potential, we iteratively solve the following system of equations:

$$\boldsymbol f = \boldsymbol g \mu(g/a_0)$$  [interpolation formula]

$$\boldsymbol f = -\nabla\phi_N + \nabla \times \boldsymbol A $$ [newtonian gravity field]

$$\boldsymbol g = -\nabla\phi_M $$[MOND gravity field]


It uses FFTs to solve the two (linear) differential equations in Fourier space.


# Required libraries:

The following are included in base Python:
* Numpy,
* math,
* itertools, 

Now there are two versions of the code, one that runs mainly on the GPU and one that runs only on the CPU. The start of the title of the code reflects this. 

For the GPU version, the following libraries are additionally needed.

* Cupy, for the installation see: https://docs.cupy.dev/en/stable/install.html
* Numba, for the installation see: https://numba.pydata.org/

  
We use Cupy to do calculations on the GPU; it is essentially numpy except it works on the GPU, using CUDA. 
We use Numba for JIT compilation.

# Advantages:
* FFT's are implemented using Cupy, allowing us to use the GPU to compute them. This speeds up the code by a lot.
* Conversion between particles and the mass density or acceleration on the grid has been parallelised and Just-In-Time (JIT) compiled, meaning that high particle numbers can be simulated faster.
* Algorithm has complexity N log(N)+n for N cells and n particles
  
# Disadvantages:
* Periodic boundary conditions often require putting the system in a large empty volume
* Mesh refinement is not possible; hence a large number of cells are needed

#  Work in progress:
Numerical:
* Replace numba library with the jax library and implement JIT compilation for larger parts of the code
* Using real-to-complex FFT's and half precision, though the latter would also require testing for accuracy.
  
Inclusion of physical effects:
* Include tidal field of the Milky Way, or arbitrary external field
* Include Coriolis and centrifugal force for rotating systems
* Collision detection

# Figures

The following figure shows the energies for a two particle simulation. The energies are: total energy (blue), kinetic energy (orange), gravitational energy (green), potential energy (red). 

![Energy figure](https://github.com/Joost987/MONDPMesh/blob/CodeJKoster/Energy2.png)

The next figure shows the orbits of the two particles.

![Orbit figure](https://github.com/Joost987/MONDPMesh/blob/CodeJKoster/Orbit2.png)
