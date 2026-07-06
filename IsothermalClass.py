from NBodyMONDPMeshClean import Particlelist, halfpixels, Body2MOND,celllen,cellleninv,G,a0, size_of_box
import numpy as np
import jax.numpy as jnp
class IsoThermalParticlelist(
    Particlelist):  # Isothermal sphere of N particles in hydrostatic equillibrium. N should be sufficiently high to approximate the thermodynamic limit.
    def __init__(self, m, b, N):
        M = N * m
        self.m = M  # Total mass
        self.b = b  # Effective radius
        self.N = N  # Number of particles
        self.v2 =  2/3 * np.sqrt(G*M*a0) * cellleninv**2
        np.random.seed(0)
        [eta1, eta2, eta3] = [np.random.uniform(low=0, high=1, size=N) for i in [0, 1, 2]]
        [zeta1, zeta2, zeta3] = [np.random.uniform(low=0, high=2 * np.pi, size=N) for i in [0, 1, 2]]
        xi = np.random.uniform(low=-1, high=1, size=N)

        # rvec: position of particle
        # vvec: velocity of particle
        b_grid = b * cellleninv
        rvec = np.transpose(np.array([[halfpixels] * 3] * N)) + b_grid * (1 / np.sqrt(eta1) - 1) ** (-2 / 3) * np.array(
            [(np.sqrt(1 - xi ** 2)) * np.cos(zeta1), (np.sqrt(1 - xi ** 2)) * np.sin(zeta1), xi]
        )
        vvec = (np.array([np.sqrt(-1 / 1.5 * self.v2 * np.log(eta2)) * np.cos(zeta2),
                          np.sqrt(-1 / 1.5 * self.v2 * np.log(eta2)) * np.sin(zeta2),
                          np.sqrt(-1 / 1.5 * self.v2 * np.log(eta3)) * np.cos(zeta3)]))
        particlelist = [[m, rvec[0, i], rvec[1, i], rvec[2, i], vvec[0, i], vvec[1, i], vvec[2, i]] for i in
                        range(np.shape(rvec)[1])]

        margin = 4
        L = 2 * halfpixels

        particlelist2 = []
        for part in particlelist:
            x, y, z = part[1], part[2], part[3]
            if (margin <= x < L - margin) and (margin <= y < L - margin) and (margin <= z < L - margin):
                particlelist2.append(part)

        particlelist = np.array(particlelist2)
        self.list = jnp.array(particlelist)
