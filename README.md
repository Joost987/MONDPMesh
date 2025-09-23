# MONDPMesh
Python N-body code made for MOND simulations.


This code uses an altered particle mesh method to solve the Poisson equation of the AQUAL version of MOND and includes the external field effect. It was developed by Joost de Nijs, Jart Koster and Martin Vingerhoets.

Their theses can be found here:

* http://resolver.tudelft.nl/uuid:ad94e143-0ce4-4b17-8a72-54ab9f656236
* https://resolver.tudelft.nl/uuid:c0132153-01e6-460c-83e0-fd7b625719d7
* https://repository.tudelft.nl/record/uuid:c52b411f-2779-4ea8-b85c-dfe2cce5e17f

The method was developed in a paper by P.M. Visser, S.W.H. Eijt and J.V. de Nijs: 	https://doi.org/10.1051/0004-6361/202347830. Note that the code used in the paper is the code in Joost's branch. The code in this branch is an improved version.

**Both .py files are needed, JAXMONDPMESH.py and ExampleSystems.py. Furthermore, if you change their filenames, you should also change the name where they are imported, which is in the first few lines of both files.**


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


This version of the code also uses JAX. You can either use JAX with your CPU or with your GPU, depending on how you install it. See https://docs.jax.dev/en/latest/installation.html#supported-platforms
* JAX 

JAX supports Just-In-Time compilation, can run on both CPU and GPU depending on what the user wants and can run on a wider range of GPU's than for example Cupy, which only supports Nvidia GPU's. The GPU's that are supported by JAX can be found on the installation page linked above. 

# Advantages:
* FFT's are implemented using JAX, allowing us to use the GPU to compute them. This speeds up the code by a lot.
* Conversion between particles and the mass density or acceleration on the grid has been parallelised and Just-In-Time (JIT) compiled, meaning that high particle numbers can be simulated faster.
* Algorithm has complexity N log(N)+n for N cells and n particles
  
# Disadvantages:
* Periodic boundary conditions often require putting the system in a large empty volume
* Mesh refinement is not possible; hence a large number of cells are needed

#  Work in progress:
Numerical:
* Using real-to-complex FFT's
  
Inclusion of physical effects:
* Include tidal field of the Milky Way, or arbitrary external field
* Include Coriolis and centrifugal force for rotating systems
* Collision detection

# Figures

The following figure shows the energies for a two particle simulation. The energies are: total energy (blue), kinetic energy (orange), gravitational energy (green), potential energy (red). 

![Energy figure](https://github.com/Joost987/MONDPMesh/blob/main/Energy2.png)

The next figure shows the orbits of the two particles.

![Orbit figure](https://github.com/Joost987/MONDPMesh/blob/main/Orbit2.png)
