# MONDPMesh Cluster
Python N-body code based on the code of J. de Nijs to simulate the evolution of three globular clusters over time (the code can easily be adapted to simulate other clusters using data from Vizier.com and the Holger Baumgart globular cluster database), with the possibility of including an external field.


# Workflow
To add observational data into the simulation results, 

1. Edit the name string to the desired globular cluster.

2. Set simulate_cluster=False (the code returns simulation parameters fitted to the data)

3. Edit the parameters and set simulate_cluster=True
4. If 2d density plots and potential plots are required, run plot_potential_t0 after completing a simulation in NBodyMONDMeshClean

# Required libraries:
The following are not included in base Python:
* numpy,
* scipy,
* jax,
* matplotlib

