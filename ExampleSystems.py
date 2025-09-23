
from JAXMONDPMESH import Particlelist, halfpixels, Body2MOND,celllen,cellleninv,G,a0
import jax.numpy as jnp
import numpy as np

#JAXNBodyMONDPMeshExample21_09_2025
# Now different classes of physical systems are made. These are specific systems of which analytical solutions in deep MOND are known.
# These also include functions to calculate the analytical accelerations, and if known the analytical potential energy.
# The simulation function is also altered to include a simulation with the exact accelerations.

class TwoBodyParticlelist(Particlelist):  # Arbitary two body system
    def __init__(self, m1, m2, rvec1, rvec2, vvec1, vvec2):
        rvec1 += jnp.array([halfpixels] * 3)  # Position vector of particle 1
        rvec2 += jnp.array([halfpixels] * 3)  # Position vector of particle 2
        self.list = jnp.array([[m1, *rvec1, *vvec1], [m2, *rvec2, *vvec2]])
        self.m1 = m1
        self.m2 = m2

    def Analyticalacc(self):
        particle1 = self.list[0]
        particle2 = self.list[1]
        Force = Body2MOND(particle1[1:4], particle2[1:4], particle1[0], particle2[0]) * (
                    particle2[1:4] - particle1[1:4]) / jnp.linalg.norm((particle1[1:4] - particle2[1:4]))
        return [1 / particle1[0] * Force, -1 / particle2[0] * Force]

    def EPotAna(self):
        return 2 / 3 * jnp.sqrt(G * a0) * (
                    (self.m1 + self.m2) ** (3 / 2) - self.m1 ** (3 / 2) - self.m2 ** (3 / 2)) * jnp.log(
            jnp.linalg.norm(self.list[0, 1:4] - self.list[1, 1:4]))

    # def TimeSim(self, timesteps, dt, itersteps, regime=0):
    #     posmat = jnp.zeros([len(self.list), timesteps, 3])
    #     vecmat = jnp.zeros([len(self.list), timesteps, 3])

    #     posmat2 = jnp.zeros([len(self.list), timesteps, 3])
    #     vecmat2 = jnp.zeros([len(self.list), timesteps, 3])

    #     MomMat = jnp.zeros([timesteps, 3])
    #     AngMat = jnp.zeros([timesteps, 3])
    #     EMat = jnp.zeros([timesteps])

    #     MomMat2 = jnp.zeros([timesteps, 3])
    #     AngMat2 = jnp.zeros([timesteps, 3])
    #     EMat2 = jnp.zeros([timesteps])

    #     accnew = self.UpdateAccsMOND(iterlen=4, regime=regime)
    #     for t in range(timesteps):
    #         posmat[:, t, :] = self.list[:, 1:4]
    #         vecmat[:, t, :] = self.list[:, 4:7]

    #         AngMat[t, :] = self.AngMom()
    #         MomMat[t, :] = jnp.transpose(self.list[:, 4:7]) @ self.list[:, 0]
    #         EMat[t] = self.ETot()

    #         accold = accnew
    #         self.list[:, 1:4] += self.list[:,
    #                              4:7] * dt + 0.5 * accold * cellleninv * dt ** 2  # Leapfrog without half integer time steps
    #         try:
    #             accnew = self.UpdateAccsMOND(iterlen=itersteps, regime=regime)
    #         except:

    #             break
    #         self.list[:, 4:7] += (accold + accnew) * 0.5 * dt * cellleninv

    #     self.list[:, 1:4] = posmat[:, 0, :]
    #     self.list[:, 4:7] = vecmat[:, 0, :]

    #     accnew = jnp.array(self.Analyticalacc())
    #     for t in range(timesteps):
    #         posmat2[:, t, :] = self.list[:, 1:4]
    #         vecmat2[:, t, :] = self.list[:, 4:7]

    #         AngMat2[t, :] = self.AngMom()
    #         MomMat2[t, :] = jnp.transpose(self.list[:, 4:7]) @ self.list[:, 0]
    #         EMat2[t] = self.Ekin() + self.EPotAna()

    #         accold = accnew
    #         self.list[:, 1:4] += self.list[:,
    #                              4:7] * dt + 0.5 * accold * cellleninv * dt ** 2  # Leapfrog without half integer time steps
    #         accnew = jnp.array(self.Analyticalacc())
    #         self.list[:, 4:7] += (accold + accnew) * 0.5 * dt * cellleninv
    #         if t % 25 == 0: print(t)
    #     return posmat, vecmat, AngMat, MomMat, EMat, posmat2, vecmat2, AngMat2, MomMat2, EMat2


class TwoBodyCircparticlelist(
    TwoBodyParticlelist):  # Use this to create a two body system in which stable orbits are produced
    def __init__(self, m1, m2, r, phase):
        M = m1 + m2
        v = 1 / celllen * np.sqrt(2 / 3 * np.sqrt(G * a0 * (m1 + m2)) * (
                    1 / (1 + np.sqrt(m1 / (m1 + m2))) + 1 / (1 + np.sqrt(m2 / (m1 + m2)))))
        rvec1 = np.array([m2 / M * r * np.cos(phase), m2 / M * r * np.sin(phase), 0])
        rvec2 = np.array([-m1 / M * r * np.cos(phase), -m1 / M * r * np.sin(phase), 0])
        rvec1 += [halfpixels] * 3
        rvec2 += [halfpixels] * 3
        vvec1 = np.array([-m2 * v / M * np.sin(phase), m2 / M * v * np.cos(phase), 0])
        vvec2 = np.array([m1 * v / M * np.sin(phase), -m1 / M * v * np.cos(phase), 0])
        self.list = jnp.array(np.array([[m1, *rvec1, *vvec1], [m2, *rvec2, *vvec2]]))
        self.m1 = m1
        self.m2 = m2


class RingParticlelist(
    Particlelist):  # Ring consisting of N particles, with one central particle. Analytical potential is unknown.
    def __init__(self, m0, r2, N, m):
        M = m0 + N * m  # Total mass
        v = jnp.sqrt(2 * jnp.sqrt(G * a0) / (3 * N * m) * (M ** (3 / 2) - m0 ** (3 / 2) - N * m ** (
                    3 / 2))) * cellleninv  # Circular velocity in deep MOND regime
        particlecentre = jnp.array([m0, halfpixels, halfpixels, halfpixels, 0, 0, 0])  # Create the central particle
        particles = [[m, halfpixels + r2 * jnp.cos(zeta), halfpixels + r2 * jnp.sin(zeta), halfpixels, -v * jnp.sin(zeta),
                      v * jnp.cos(zeta), 0] for zeta in
                     np.random.uniform(0, 2 * jnp.pi, N)]  # Create particles in the ring
        particles.append(particlecentre)
        particlelist = jnp.array(particles)

        self.list = particlelist
        self.m0 = m0  # Mass of central particle
        self.r = r2  # Radius of ring
        self.N = N  # Number of particles in the ring
        self.m = m  # Mass of particles in ring

    def RingMONDacc(self):
        M = self.m0 + self.N * self.m  # Total mass
        rhat = -1 * jnp.transpose(jnp.transpose(self.list[:-1, 1:4] - self.list[-1, 1:4]) / jnp.linalg.norm(
            self.list[:-1, 1:4] - self.list[-1, 1:4],
            axis=1))  # Unit direction vector of acceleration (pointing towards origin)
        return 2 / 3 * jnp.sqrt(G * a0) / (self.r * celllen) * (
                    M ** (3 / 2) - self.m0 ** (3 / 2) - self.N * self.m ** (3 / 2)) / (self.m * self.N) * rhat


class IsoThermalParticlelist(
    Particlelist):  # Isothermal sphere of N particles in hydrostatic equillibrium. N should be sufficiently high to approximate the thermodynamic limit.
    def __init__(self, m, b, N):
        M = N * m
        self.m = M  # Total mass
        self.b = b  # Effective radius
        self.N = N  # Number of particles
        self.v2 = np.sqrt(G * a0 * self.m) / 3 * cellleninv ** 2 * 2  # Variance in velocity
        [eta1, eta2, eta3] = [np.random.uniform(low=0, high=1, size=N) for i in [0, 1, 2]]
        [zeta1, zeta2, zeta3] = [np.random.uniform(low=0, high=2 * np.pi, size=N) for i in [0, 1, 2]]
        xi = np.random.uniform(low=-1, high=1, size=N)

        # rvec: position of particle
        # vvec: velocity of particle
        rvec = np.transpose(np.array([[halfpixels] * 3] * N)) + b * (1 / np.sqrt(eta1) - 1) ** (-2 / 3) * np.array(
            [(np.sqrt(1 - xi ** 2)) * np.cos(zeta1), (np.sqrt(1 - xi ** 2)) * np.sin(zeta1), xi])
        vvec = (np.array([np.sqrt(-1 / 1.5 * self.v2 * np.log(eta2)) * np.cos(zeta2),
                          np.sqrt(-1 / 1.5 * self.v2 * np.log(eta2)) * np.sin(zeta2),
                          np.sqrt(-1 / 1.5 * self.v2 * np.log(eta3)) * np.cos(zeta3)]))
        particlelist = [[m, rvec[0, i], rvec[1, i], rvec[2, i], vvec[0, i], vvec[1, i], vvec[2, i]] for i in
                        range(np.shape(rvec)[1])]

        particlelist2 = []
        for part in particlelist:  # All particles inside the grid are selected
            if np.abs(part[1]) < 2 * halfpixels - 4 and np.abs(part[2]) < 2 * halfpixels - 4 and np.abs(
                    part[3]) < 2 * halfpixels - 4:  # we need to subtract 4 to account for the smoothing
                particlelist2.append(part)

        particlelist = np.array(particlelist2)
        self.list = jnp.array(particlelist)

    def Analyticalacc(self):  # Returns the analytical acceleration
        rvec = self.list[:, 1:4]
        rvec = rvec - jnp.array([halfpixels, halfpixels, halfpixels])  # rvec: position of particle

        r = jnp.linalg.norm(rvec, axis=jnp.where(jnp.array(jnp.shape(rvec)) == 3)[0][
            0])  # The axis expression makes sure it takes the norm at the axis where rvec has 3 components
        return -rvec * jnp.transpose(
            jnp.array([jnp.sqrt(G * self.m * a0 / (self.b ** 3 * r)) / (1 + (r / self.b) ** (3 / 2))] * 3)) * cellleninv

    def EPotAna(self):  # Returns the analytical potential energy
        return 2 / 3 * jnp.sqrt(G * self.m * a0) * self.m / self.N * jnp.sum(
            jnp.log(1 + (jnp.linalg.norm(self.list[:, 1:4] - jnp.array([halfpixels] * 3), axis=1) / self.b) ** (3 / 2)))

    def EGravAna(self):  # Returns the analytical gravitational energy
        pass

   # def TimeSim(self, timesteps, dt, itersteps, EFE):
    #    posmat = jnp.zeros([len(self.list), timesteps, 3])  # posmat = position vector
      #  vecmat = jnp.zeros([len(self.list), timesteps, 3])  # vecmat = velocity vector

       # posmat_a = jnp.zeros([len(self.list), timesteps, 3])  # posmat_a = analytical position vector
      #  vecmat_a = jnp.zeros([len(self.list), timesteps, 3])  # vecmat_a = analytical velocity vector

      #  MomMat = jnp.zeros([timesteps, 3])  # MomMat = momentum vector
      #  AngMat = jnp.zeros([timesteps, 3])  # AngMat = angular momentum vector
       # EMat = jnp.zeros([timesteps])  # EMat = energy

       # MomMat_a = jnp.zeros([timesteps, 3])  # MomMat_a = analytical momentum vector
      #  AngMat_a = jnp.zeros([timesteps, 3])  # AngMat_a = analytical angular momentum vector
    #    EMat_a = jnp.zeros([timesteps])  # EMat_a = analytical energy
#
      #  accnew = self.UpdateAccsMOND(self.list,EFE,iterlen=4)
      #  for t in range(timesteps):

       #     posmat[:, t, :] = self.list[:, 1:4]
       #     vecmat[:, t, :] = self.list[:, 4:7]

       #     AngMat[t, :] = self.AngMom()
      #      MomMat[t, :] = jnp.transpose(self.list[:, 4:7]) @ self.list[:, 0]
      #      EMat[t] = self.ETot()

         #   accold = accnew
        #    self.list[:, 1:4] += self.list[:,
          #                       4:7] * dt + 0.5 * accold * cellleninv * dt ** 2  # Leapfrog without half integer time steps
          #  try:
        #        accnew = self.UpdateAccsMOND(self.list,EFE,iterlen=itersteps)
        #    except:
        #        self.list[:, 1:4] = self.list[:, 1:4] % (2 * halfpixels - 4)
        #        accnew = self.UpdateAccsMOND(self.list,EFE,iterlen=itersteps)
        #    self.list[:, 4:7] += (accold + accnew) * 0.5 * dt * cellleninv

      #  self.list[:, 1:4] = posmat[:, 0, :]
      #  self.list[:, 4:7] = vecmat[:, 0, :]
      #  if not EFE[0]:
        #    accnew = jnp.array(self.Analyticalacc())
        #    for t in range(timesteps):
          #      posmat_a[:, t, :] = self.list[:, 1:4]
           #     vecmat_a[:, t, :] = self.list[:, 4:7]

           #     AngMat_a[t, :] = self.AngMom()
            #    MomMat_a[t, :] = jnp.transpose(self.list[:, 4:7]) @ self.list[:, 0]
                #EMat_a[t] = self.Ekin() + self.EPotAna() + self.EGravAna()

            #    accold = accnew
            #    self.list[:, 1:4] += self.list[:,
            #                        4:7] * dt + 0.5 * accold * cellleninv * dt ** 2  # Leapfrog without half integer time steps
            #    accnew = jnp.array(self.Analyticalacc())
              #  self.list[:, 4:7] += (accold + accnew) * 0.5 * dt * cellleninv
             #   if t % 25 == 0: print(t)
        #else:
           # print("Analytical simulation does not work with external field effect!")

        #return posmat, vecmat, AngMat, MomMat, EMat, posmat_a, vecmat_a, AngMat_a, MomMat_a, EMat_a

