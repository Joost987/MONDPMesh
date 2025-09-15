# %%
# %%
import time
import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
import scipy.optimize


# %% Simulation parameters and physical constants

# Some of the simulation parameters. You can change halfpixels, which is half the amount of pixels in one dimension of the grid
# You can also change celllen, which is the distance between neighbouring pixels. Some other constants are defined, as this ensures that these calculations are only done once.
halfpixels = 32 #For optimal FFT's, this has to be a power of 2.

size_of_box = 4*10**15 # m
size_of_box = 26738 # au
size_of_box = 1 # ly



timesteps = 1000 # total number of timesteps
# T is total simulated time
T = 400 # kyr (Alpha Centauri AB has oribtal period of about 80 kyr)
T = 0.4 # Myr
dt = T/timesteps

#Some physical constants
G = 6.674*10**(-11) # m^3/(s^2*kg)
G = 3.942*10**7 # au^3/(kyr^2*M☉)
G = 0.156 # ly^3/(Myr^2*M☉)
a0 = 1.2*10**(-10) # m/s^2
a0 = 0.7978 # au/kyr^2
a0 = 12.614 #ly/Myr^2

EFE_on = False #External field effect on or off
EFE_M_strength = 1 * a0  # au/kyr^2

EFE_M = [EFE_on, EFE_M_strength];
itersteps = 4;
regime = 3 #Which interpolation function should be used?
#Different values for regime parameter
#0: Deep MOND
#1: Standard
#2: McGaugh
#3: Bose-Einstein
#4: Verlinde
#5: Newton
#TODO: regime 2 cpu/gpu 



c = 4*np.pi*G
shape = (2*halfpixels,2*halfpixels,2*halfpixels)

celllen = size_of_box/(2*halfpixels)
cellleninv = 1/celllen
size = halfpixels*celllen
kstep = np.pi/(halfpixels*celllen)
kstep2inv = 1/kstep**2

cellvolume = celllen**3
cellvolumeinv = 1/cellvolume
oversqrt2pi = 1/np.sqrt(2*np.pi)
oversqrt2pi3 = oversqrt2pi**3

# a0 = 1.2*10^-10 m/s^2 = 0.7978 au/kyr^2 = 12.614 Mly/Gyr^2
# Acceleration from Milky Way on Sun = 3.47*10^-9 m/s^2 = 23.07 au/kyr^2 = 364.8 Mly/Gyr^2
# Newtons gravitational constant = 6.674*10^-11 m^3/(s^2*kg) = 3.942*10^7 au^3/(kyr^2*M☉) = 0.235 Mly^3/(Gyr^2*M_MW)

simulate_two_bodies = True #Do a simulation
allowNewtonCorrections = False #Do Newton corrections



def FindBall(N):
    ball = np.array([])
    for i in range(-N + 1, N):
        for j in range(-N + 1, N):
            for k in range(-N + 1, N):
                if i ** 2 + j ** 2 + k ** 2 < N ** 2:
                    ball = np.append(ball, [i, j, k])
    ball = np.reshape(ball, (251, 3)).astype("int32")
    return ball


ball4 = FindBall(4)


def AnalyticGravitationalPotential(r, r0, m, EFE):
    return -scipy.special.erf(np.linalg.norm(r - r0) / np.sqrt(2)) / np.linalg.norm(r - r0) * m * G


# Configure CUDA kernels



def AssignMassGaussShape(density,particlelist,cellvolumeinv,a=1,shape=ball4):  #N is order of the method divided by 2, amount of points used is (2N)^3
    #For a=1, N>3, a=1.5 N>4, a=2 N>5, a=3 N>8. 
  
    a3inv=(a)**(-3)
    for i in particlelist: 
        x=i[1]
        y=i[2]
        z=i[3]
      
        for j in shape: 
            cellcoords=(int(i[1])+j[0],int(i[2])+j[1],int(i[3])+j[2])
            weight=oversqrt2pi3*np.exp(-((cellcoords[0]-x)**2+(cellcoords[1]-y)**2+(cellcoords[2]-z)**2)/(2*a**2))
            density[cellcoords]+=i[0]*weight*cellvolumeinv*a3inv 




shape_ball = ball4

def AssignAccsGaussShape(accparts,accmat,particlelist2,a=1,shape=ball4):
    a3inv=a**(-3)

    for k,i in enumerate(particlelist2): 
        x=i[1]
        y=i[2]
        z=i[3]
        
        for j in shape:
            cellcoords=(int(x)+j[0],int(y)+j[1],int(z)+j[2]) 
            weight=weight=oversqrt2pi3*np.exp(-((cellcoords[0]-x)**2+(cellcoords[1]-y)**2+(cellcoords[2]-z)**2)/(2*a**2))
            accparts[k,:]+=(accmat[:,cellcoords[0],cellcoords[1],cellcoords[2]])*weight*a3inv



# %%
# =============================================================================
#
# Making classes
#
# =============================================================================

# First the Particlelist class is made. This is essentially a list of the masses, positions and velocities of the particles.
# together with different functions acting on this physical system. These are functions to find the kinetic energy, angular momentum
# and accelerations on the particles in the system. A function to simulate this system is also included.
class Particlelist:
    def __init__(self, particlelist):
        # particlelist object should for each particle contain a list with its mass, then the 3 components of its position
        # then the 3 components of its velocity, so [m,rx,ry,rz,vx,vy,vz].
        self.list = np.array(particlelist, dtype=np.float32)
        self.EPot = 0
        self.EGrav = 0

    def Ekin(self):
        return np.dot(1 / 2 * self.list[:, 0], np.diagonal(self.list[:, 4:7] @ np.transpose(
            self.list[:, 4:7]))) * celllen ** 2  # take the dot product between 1/2*masses and the velocities squared.
        # the self.list[:,4:7]@np.transpose(self.list[:,4:7]) part creates a matrix with all of the velocities of each particle multiplied by each other
        # we only want the velocities of each particle squared, so we take the diagonal of this.

    def ETot(self):
        return self.Ekin() + self.EPot + self.EGrav
        # Note that EPot can only be calculated by UpdateAccsMOND, as the potential is needed.

    def AngMom(self):
        return np.sum(np.diag(self.list[:, 0]) @ np.cross(self.list[:, 1:4] - np.array([halfpixels] * 3), self.list[:,
                                                                                                          4:7]))  # angular momentum with [halfpixels,halfpixels,halfpixels] as origin.

    def CenterOfMass(self):
        return 1 / np.sum(self.list[:, 0]) * (self.list[:, 0] @ self.list[:, 1:4]), 1 / np.sum(self.list[:, 0]) * (
                    self.list[:, 0] @ self.list[:, 4:7])

    def UpdateAccsMOND(self, EFE, shape_ball=ball4, sigma=1, iterlen=4, regime=0):
        density = np.zeros((2 * halfpixels, 2 * halfpixels, 2 * halfpixels), dtype=np.float32)
        AssignMassGaussShape(density, self.list, cellvolumeinv, sigma, shape_ball)
        densityfft = np.fft.fftn(density)
        potNDmat = CalcPot(densityfft)
        del densityfft
        accNDmat = CalcAccMat(potNDmat)
        if EFE[0] and (EFE[1] == 1):
            accNDmat[2, :, :, :] += -Calculate_gN_gal(EFE_M, EFE_M[1] / a0, regime)
        del potNDmat
        H = np.zeros([3, 2 * halfpixels, 2 * halfpixels, 2 * halfpixels], dtype=np.float32)
        for i in range(iterlen):
            accMONDmat, H = MainLoop(H, accNDmat, regime, EFE)
        accMONDmatfft = np.fft.fftn(accMONDmat, axes=(1, 2, 3))
        del accMONDmat  # the potential in Fourier space is -i*kvec*gvec, where gvec is the acceleration field
        potMONDmatfft = -KdotProd(accMONDmatfft) * K2inv / kstep
        del accMONDmatfft
        potMONDmat = np.imag(np.fft.ifftn(potMONDmatfft))
        del potMONDmatfft
        accMONDmat = CalcAccMat(potMONDmat)
        self.EGrav, EGravPot = EGrav(accMONDmat,H+accNDmat,regime)
        self.EPot = np.sum(potMONDmat * density) * cellvolume
        # if regime == 2 or regime == 4:
        #     self.EGrav = EGravPot - self.EPot
        del H
        del accNDmat
        
        del density
        del potMONDmat
        accparts = np.zeros((len(self.list), 3), dtype=np.float32)
        AssignAccsGaussShape(accparts, accMONDmat, self.list, sigma, shape_ball)
        return accparts

    def NewtonCorrection(self, particlelist, accelerations, func, correction_criteria, std):

        # Assume accelerations is an array of the form [[ax, ay, az], ... ]
        acceleration_norms = np.sqrt(np.sum(accelerations ** 2, axis=1))
        newtoncheck = inpol(x=acceleration_norms, func=func)

        correction_check = np.array([newtoncheck >= correction_criteria])[0]

        for index1, particle1 in enumerate(particlelist):
            if correction_check[index1] == False: continue
            correction = np.zeros(3)
            for index2, particle2 in enumerate(particlelist):
                if index1 == index2: continue
                dist_vec = particle1[1:4] - particle2[1:4]
                dist = np.linalg.norm(dist_vec)

                intermediatefactor = scipy.special.erf(dist / (2 * std)) / dist ** 2 \
                                     - np.exp(-dist ** 2 / (4 * std ** 2)) / (np.sqrt(np.pi) * dist * std)
                correction += intermediatefactor * dist_vec / dist * G * particle2[0] * cellleninv
                # TODO: check for division by 0 issues
                correction += (-G * particle2[0]) / dist ** 3 * dist_vec * cellleninv ** 2

            accelerations[index1] += correction

        return accelerations

    def TimeSim(self, timesteps, dt, itersteps, EFE, free_fall, regime=0):
        posmat = np.zeros([len(self.list), timesteps, 3], dtype=np.float32)
        vecmat = np.zeros([len(self.list), timesteps, 3], dtype=np.float32)

        MomMat = np.zeros([timesteps, 3], dtype=np.float32)
        AngMat = np.zeros([timesteps, 3], dtype=np.float32)
        EkinMat = np.zeros([timesteps], dtype=np.float32)
        EgravMat = np.zeros([timesteps], dtype=np.float32)
        EMat = np.zeros([timesteps], dtype=np.float32)

        accnew = self.UpdateAccsMOND(EFE, iterlen=4, regime=regime)
        if allowNewtonCorrections:
            #TODO: explain the 0.99.
            accnew = self.NewtonCorrection(self.list, accnew, regime, 0.99, 1)

        COM = np.zeros([7, timesteps], dtype=np.float32)

        for t in range(timesteps):

            posmat[:, t, :] = self.list[:, 1:4]
            vecmat[:, t, :] = self.list[:, 4:7]

            AngMat[t, :] = self.AngMom()
            MomMat[t, :] = np.transpose(self.list[:, 4:7]) @ self.list[:, 0]
            EkinMat[t] = self.Ekin()
            EgravMat[t] = self.EGrav
            EMat[t] = self.ETot()

            accold = accnew
            self.list[:, 1:4] += self.list[:,
                                 4:7] * dt + 0.5 * accold * cellleninv * dt ** 2  # Leapfrog without half integer time steps
            if (free_fall == 1) and (EFE[0] == True):
                if t > 0:
                    self.list[:, 3] += 0.5 * EFE[2] * cellleninv * dt ** 2  # + 1/2 g t^2
                    self.list[:, 6] += EFE[2] * cellleninv * dt  # + g t
            try:  # If the particles are outside of the grid this will raise an error. This catches this error
                # and breaks the loop, ensuring that the data from before the error can be returned.
                accnew = self.UpdateAccsMOND(EFE, iterlen=itersteps, regime=regime)
                if allowNewtonCorrections:
                    # TODO: explain the 0.99.
                    accnew = self.NewtonCorrection(self.list, accnew, regime, 0.99, 1)
            except:  # different ways of handling this exception can be made. For the isothermal sphere for example
                # the particles will enter
                print("particle outside box")
                break
            self.list[:, 4:7] += (accold + accnew) * 0.5 * dt * cellleninv
            if (free_fall == 2 or free_fall == 3) and (EFE[0] == True):
                if free_fall == 2:
                    COM[1:4, t] = self.CenterOfMass()[0]
                    COM[4:7, t] = self.CenterOfMass()[1]
                    COM[0, t] = self.EPot  # Epot shifted from center

                    self.list[:, 1:4] += -(self.CenterOfMass()[0] - np.array([halfpixels, halfpixels, halfpixels]))
                    self.list[:, 4:7] += -self.CenterOfMass()[1]

                if free_fall == 3:
                    COM[1:4, t] = self.CenterOfMass()[0]
                    COM[4:7, t] = self.CenterOfMass()[1]
                    COM[0, t] = self.EPot

                    self.list[:, 1:4] += -(self.CenterOfMass()[0] - [halfpixels, halfpixels, halfpixels])
                    self.list[:, 4:7] += -self.CenterOfMass()[1]
                    if free_fall == 3:
                        self.UpdateAccsMOND(EFE, iterlen=itersteps, regime=regime)
                        E_temp = self.EPot

                        for i in range(1, t):
                            self.list[:, 1:4] += -(
                                        dt * COM[4:7, i] + COM[1:4, i - 1] - [halfpixels, halfpixels, halfpixels])
                            self.UpdateAccsMOND(EFE, iterlen=itersteps, regime=regime)
                            self.list[:, 1:4] += dt * COM[4:7, i] + COM[1:4, i - 1] - [halfpixels, halfpixels,
                                                                                       halfpixels]
                            E_dt_keer_v_i = self.EPot
                            COM[0, t] += E_dt_keer_v_i - E_temp

            if t % 25 == 0: print(t)
        return posmat, vecmat, AngMat, MomMat, EkinMat, EgravMat, EMat, COM


# Now different classes of physical systems are made. These are specific systems of which analytical solutions in deep MOND are known.
# These also include functions to calculate the analytical accelerations, and if known the analytical potential energy.
# The simulation function is also altered to include a simulation with the exact accelerations.

class TwoBodyParticlelist(Particlelist):  # Arbitary two body system
    def __init__(self, m1, m2, rvec1, rvec2, vvec1, vvec2):
        rvec1 += np.array([halfpixels] * 3)  # Position vector of particle 1
        rvec2 += np.array([halfpixels] * 3)  # Position vector of particle 2
        self.list = np.array([[m1, *rvec1, *vvec1], [m2, *rvec2, *vvec2]])
        self.m1 = m1
        self.m2 = m2

    def Analyticalacc(self):
        particle1 = self.list[0]
        particle2 = self.list[1]
        Force = Body2MOND(particle1[1:4], particle2[1:4], particle1[0], particle2[0]) * (
                    particle2[1:4] - particle1[1:4]) / np.linalg.norm((particle1[1:4] - particle2[1:4]))
        return [1 / particle1[0] * Force, -1 / particle2[0] * Force]

    def EPotAna(self):
        return 2 / 3 * np.sqrt(G * a0) * (
                    (self.m1 + self.m2) ** (3 / 2) - self.m1 ** (3 / 2) - self.m2 ** (3 / 2)) * np.log(
            np.linalg.norm(self.list[0, 1:4] - self.list[1, 1:4]))

    # def TimeSim(self, timesteps, dt, itersteps, regime=0):
    #     posmat = np.zeros([len(self.list), timesteps, 3])
    #     vecmat = np.zeros([len(self.list), timesteps, 3])

    #     posmat2 = np.zeros([len(self.list), timesteps, 3])
    #     vecmat2 = np.zeros([len(self.list), timesteps, 3])

    #     MomMat = np.zeros([timesteps, 3])
    #     AngMat = np.zeros([timesteps, 3])
    #     EMat = np.zeros([timesteps])

    #     MomMat2 = np.zeros([timesteps, 3])
    #     AngMat2 = np.zeros([timesteps, 3])
    #     EMat2 = np.zeros([timesteps])

    #     accnew = self.UpdateAccsMOND(iterlen=4, regime=regime)
    #     for t in range(timesteps):
    #         posmat[:, t, :] = self.list[:, 1:4]
    #         vecmat[:, t, :] = self.list[:, 4:7]

    #         AngMat[t, :] = self.AngMom()
    #         MomMat[t, :] = np.transpose(self.list[:, 4:7]) @ self.list[:, 0]
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

    #     accnew = np.array(self.Analyticalacc())
    #     for t in range(timesteps):
    #         posmat2[:, t, :] = self.list[:, 1:4]
    #         vecmat2[:, t, :] = self.list[:, 4:7]

    #         AngMat2[t, :] = self.AngMom()
    #         MomMat2[t, :] = np.transpose(self.list[:, 4:7]) @ self.list[:, 0]
    #         EMat2[t] = self.Ekin() + self.EPotAna()

    #         accold = accnew
    #         self.list[:, 1:4] += self.list[:,
    #                              4:7] * dt + 0.5 * accold * cellleninv * dt ** 2  # Leapfrog without half integer time steps
    #         accnew = np.array(self.Analyticalacc())
    #         self.list[:, 4:7] += (accold + accnew) * 0.5 * dt * cellleninv
    #         if t % 25 == 0: print(t)
    #     return posmat, vecmat, AngMat, MomMat, EMat, posmat2, vecmat2, AngMat2, MomMat2, EMat2


class TwoBodyCircParticlelist(
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
        self.list = np.array(np.array([[m1, *rvec1, *vvec1], [m2, *rvec2, *vvec2]]))
        self.m1 = m1
        self.m2 = m2


class RingParticlelist(
    Particlelist):  # Ring consisting of N particles, with one central particle. Analytical potential is unknown.
    def __init__(self, m0, r2, N, m):
        M = m0 + N * m  # Total mass
        v = np.sqrt(2 * np.sqrt(G * a0) / (3 * N * m) * (M ** (3 / 2) - m0 ** (3 / 2) - N * m ** (
                    3 / 2))) * cellleninv  # Circular velocity in deep MOND regime
        particlecentre = np.array([m0, halfpixels, halfpixels, halfpixels, 0, 0, 0])  # Create the central particle
        particles = [[m, halfpixels + r2 * np.cos(zeta), halfpixels + r2 * np.sin(zeta), halfpixels, -v * np.sin(zeta),
                      v * np.cos(zeta), 0] for zeta in
                     np.random.uniform(0, 2 * np.pi, N)]  # Create particles in the ring
        particles.append(particlecentre)
        particlelist = np.array(particles)

        self.list = particlelist
        self.m0 = m0  # Mass of central particle
        self.r = r2  # Radius of ring
        self.N = N  # Number of particles in the ring
        self.m = m  # Mass of particles in ring

    def RingMONDacc(self):
        M = self.m0 + self.N * self.m  # Total mass
        rhat = -1 * np.transpose(np.transpose(self.list[:-1, 1:4] - self.list[-1, 1:4]) / np.linalg.norm(
            self.list[:-1, 1:4] - self.list[-1, 1:4],
            axis=1))  # Unit direction vector of acceleration (pointing towards origin)
        return 2 / 3 * np.sqrt(G * a0) / (self.r * celllen) * (
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
        self.list = np.array(particlelist)

    def Analyticalacc(self):  # Returns the analytical acceleration
        rvec = self.list[:, 1:4]
        rvec = rvec - np.array([halfpixels, halfpixels, halfpixels])  # rvec: position of particle

        r = np.linalg.norm(rvec, axis=np.where(np.array(np.shape(rvec)) == 3)[0][
            0])  # The axis expression makes sure it takes the norm at the axis where rvec has 3 components
        return -rvec * np.transpose(
            np.array([np.sqrt(G * self.m * a0 / (self.b ** 3 * r)) / (1 + (r / self.b) ** (3 / 2))] * 3)) * cellleninv

    def EPotAna(self):  # Returns the analytical potential energy
        return 2 / 3 * np.sqrt(G * self.m * a0) * self.m / self.N * np.sum(
            np.log(1 + (np.linalg.norm(self.list[:, 1:4] - np.array([halfpixels] * 3), axis=1) / self.b) ** (3 / 2)))

    def EGravAna(self):  # Returns the analytical gravitational energy
        pass

    def TimeSim(self, timesteps, dt, itersteps, EFE):
        posmat = np.zeros([len(self.list), timesteps, 3])  # posmat = position vector
        vecmat = np.zeros([len(self.list), timesteps, 3])  # vecmat = velocity vector

        posmat_a = np.zeros([len(self.list), timesteps, 3])  # posmat_a = analytical position vector
        vecmat_a = np.zeros([len(self.list), timesteps, 3])  # vecmat_a = analytical velocity vector

        MomMat = np.zeros([timesteps, 3])  # MomMat = momentum vector
        AngMat = np.zeros([timesteps, 3])  # AngMat = angular momentum vector
        EMat = np.zeros([timesteps])  # EMat = energy

        MomMat_a = np.zeros([timesteps, 3])  # MomMat_a = analytical momentum vector
        AngMat_a = np.zeros([timesteps, 3])  # AngMat_a = analytical angular momentum vector
        EMat_a = np.zeros([timesteps])  # EMat_a = analytical energy

        accnew = self.UpdateAccsMOND(EFE,iterlen=4)
        for t in range(timesteps):

            posmat[:, t, :] = self.list[:, 1:4]
            vecmat[:, t, :] = self.list[:, 4:7]

            AngMat[t, :] = self.AngMom()
            MomMat[t, :] = np.transpose(self.list[:, 4:7]) @ self.list[:, 0]
            EMat[t] = self.ETot()

            accold = accnew
            self.list[:, 1:4] += self.list[:,
                                 4:7] * dt + 0.5 * accold * cellleninv * dt ** 2  # Leapfrog without half integer time steps
            try:
                accnew = self.UpdateAccsMOND(EFE,iterlen=itersteps)
            except:
                self.list[:, 1:4] = self.list[:, 1:4] % (2 * halfpixels - 4)
                accnew = self.UpdateAccsMOND(EFE,iterlen=itersteps)
            self.list[:, 4:7] += (accold + accnew) * 0.5 * dt * cellleninv

        self.list[:, 1:4] = posmat[:, 0, :]
        self.list[:, 4:7] = vecmat[:, 0, :]
        if not EFE[0]:
            accnew = np.array(self.Analyticalacc())
            for t in range(timesteps):
                posmat_a[:, t, :] = self.list[:, 1:4]
                vecmat_a[:, t, :] = self.list[:, 4:7]

                AngMat_a[t, :] = self.AngMom()
                MomMat_a[t, :] = np.transpose(self.list[:, 4:7]) @ self.list[:, 0]
                #EMat_a[t] = self.Ekin() + self.EPotAna() + self.EGravAna()

                accold = accnew
                self.list[:, 1:4] += self.list[:,
                                    4:7] * dt + 0.5 * accold * cellleninv * dt ** 2  # Leapfrog without half integer time steps
                accnew = np.array(self.Analyticalacc())
                self.list[:, 4:7] += (accold + accnew) * 0.5 * dt * cellleninv
                if t % 25 == 0: print(t)
        else:
            print("Analytical simulation does not work with external field effect!")

        return posmat, vecmat, AngMat, MomMat, EMat, posmat_a, vecmat_a, AngMat_a, MomMat_a, EMat_a


# %% Functions

# Mondian acceleration between two bodies
def Body2MOND(x, y, m1, m2):
    M = m1 + m2
    return 2 / 3 * np.sqrt(G * a0) / (celllen * np.linalg.norm(x[0:2] - y[0:2], axis=0)) * (
                M ** (3 / 2) - m1 ** (3 / 2) - m2 ** (3 / 2))


# Calculate potential from Fourier transformed density.
def CalcPot(densityfft):
    potmatfft = -c * densityfft * K2inv / kstep ** 2  # *kstep2inv #kstep2inv is 1/kstep**2
    del densityfft
    potmat = np.fft.ifftn(potmatfft, s=shape)  # inverse Fourier Transform
    del potmatfft
    potmat = np.real(potmat)
    return potmat


# Use finite differences to calculate the acceleration field on the grid.
def CalcAccMat(potmat):
    accmat = np.array([(np.roll(potmat, 1, axis=0) - np.roll(potmat, -1, axis=0)) / (2 * celllen),
                       (np.roll(potmat, 1, axis=1) - np.roll(potmat, -1, axis=1)) / (2 * celllen),
                       (np.roll(potmat, 1, axis=2) - np.roll(potmat, -1, axis=2)) / (2 * celllen)])
    return accmat


def COMConverter(particles, pos, vec,
                 COM):  # COMConverter = Center Of Mass Converter. This function converts the coordinates where the COM was kept constant, to the coordinates where the starting position is constant
    posmat = np.zeros([len(particles.list), len(COM[0]), 3])
    vecmat = np.zeros([len(particles.list), len(COM[0]), 3])
    AngMat = np.zeros(len(COM[0]))
    MomMat = np.zeros(len(COM[0]))
    EkinMat = np.zeros(len(COM[0]))
    EMat = np.zeros(len(COM[0]))
    for t in range(len(COM[0])):
        vecmat[:, t, :] = vec[:, t, :] + np.sum(COM[4:7, 0:t], axis=1)
        posmat[:, t, :] = pos[:, t, :] + np.sum(COM[1:4, 0:t], axis=1) - [(t) * halfpixels, (t) * halfpixels,
                                                                          (t) * halfpixels]
        for i in range(t):
            posmat[:, t, :] += dt * np.sum(COM[4:7, 0:(t - i)], axis=1)
        EkinMat[t] = 0.5 * np.dot(particles.list[:, 0],
                                  np.diagonal(vecmat[:, t, :] @ np.transpose(vecmat[:, t, :]))) * celllen ** 2
        for i in range(1, t):
            EMat[t] += (COM[0, i] - COM[0, i - 1])
            print((COM[0, i] - COM[0, i - 1]))
        EMat[t] += COM[0, t]
    return posmat, vecmat, AngMat, MomMat, EkinMat, EMat


def KdotProd(
        A):  # Dot product of a vector field with k vector. K vector is an element of the Fourier transformed domain.
    return (inprodx * A[0] + inprody * A[1] + inprodz * A[2])


def inpol(x, func):  # Interpolation function \mu
    if func == 0: return x  # Deepmond
    if func == 1: return x / np.sqrt(1 + x ** 2)  # Standard
    if func == 2: return FindMu(lambda y: inpolinv(y, func), x)  # McGaugh
    if func == 3: return 1 - np.exp(-x)  # Bose-Einstein
    if func == 4: return 4*x/(1+np.sqrt(1+4*x))**2 # Verlinde
    if func == 5: return 1  # Newton


def inpolinv(y, func):  # Inverse interpolation function \nu
    if func == 0: return 1 / np.sqrt(y)  # Deepmond
    if func == 1: return np.sqrt(1 / 2 + 1 / 2 * np.sqrt(1 + 4 / y ** 2))  # Standard
    if func == 2: return 1 / (1 - np.exp(-np.sqrt(y)))  # McGaugh
    if func == 3: return FindNu(lambda x: inpol(x, func), y)  # Bose-Einstein
    if func == 4: return 1 + 1/np.sqrt(y) # Verlinde
    if func == 5: return 1  # Newton



def FindMu(nu, x, tol=1e-3):
    return scipy.optimize.newton(lambda mu: mu * nu(x * mu) - 1, x, tol=tol)


def FindNu(mu, y, tol=1e-3):
    return scipy.optimize.newton(lambda nu: nu * mu(y * nu) - 1, np.sqrt(1 / y), tol=tol)

def EGrav(accMONDmat,F,func):
    EGrav = 0
    EGravPot = 0 # Sum of potential energy and gravitational energy
    x = np.linalg.norm(accMONDmat,axis=0)/a0
    y = np.linalg.norm(F,axis=0)/a0
    if func == 0:
        EGrav = np.sum(x**3/3) 
    if func == 1:
        EGrav = np.sum((x*np.sqrt(1+x**2)-np.arcsinh(x))/2)
    if func == 2:
        eps=1e-8
        V_prime = lambda y0,V: y0/(1-np.exp(-np.sqrt(y0))) 
        EGravPot = np.sum(scipy.integrate.odeint(V_prime,eps,np.append(np.array(eps),np.sort(y.flatten())),tfirst=True)) 
        #We numerically integrate V_prime over y to find V. Afterwards we integrate V over space to find E_grav+E_pot. 
        #The numerical integration over y is done by first sorting the y array. As we will integrate over real spaces, the order of the y array does not matter
        #Then we solve the ode dV/dy=V_prime. Scipy will solve this ode and give us the value of the integral for all points
        #specified.
        EGravPot = EGravPot                                                                         
    if func == 3:
        EGrav = np.sum(x**2/2+(x+1)*np.exp(-x))
    if func == 4:
        EGrav = np.sum((x+x**2)/2-(1+4*x)**(3/2)/12)
        EGravPot = np.sum(y**2/2 + 2*y**(3/2)/3)
    if func == 5:
        EGrav = np.sum(x**2/2)
    return a0**2/(4*np.pi*G) * EGrav * cellvolume , -a0**2/(4*np.pi*G) * EGravPot * cellvolume
        


def CurlFreeProj(Ax, Ay, Az):  # Calculates the curl free projection of the vector field A = [Ax,Ay,Az] using FFT's
    A = np.array([Ax, Ay, Az])
    del Ax, Ay, Az
    Ahat = np.fft.fftn(A, s=shape)
    del A
    intermediatestep = K2inv * KdotProd(Ahat)
    del Ahat
    xyz = np.fft.ifftn(np.array([intermediatestep * inprodx, intermediatestep * inprody, intermediatestep * inprodz]),
                       s=shape)
    del intermediatestep
    return xyz


def DivFreeProj(Ax, Ay, Az):  # Calculates the divergence free projection of the vector field A = [Ax,Ay,Az] using FFT's
    A = np.array([Ax, Ay, Az])
    del Ax, Ay, Az
    Ahat = np.fft.fftn(A, s=shape, axes=(1, 2, 3))
    del A
    intermediatestep = K2inv * KdotProd(Ahat)
    xyz = np.fft.ifftn(np.array([Ahat[0] - intermediatestep * inprodx, Ahat[1] - intermediatestep * inprody,
                                 Ahat[2] - intermediatestep * inprodz]), s=shape, axes=(1, 2, 3))
    del intermediatestep
    del Ahat
    return xyz


def Calculate_gN_gal(EFE_M, x,
                     func):  # This calculates the field strength which needs to be added to the Newton acceleration field if an external field is simulated using method 3
    # x = gM/a0
    # y = gN/a0
    # mu(x) = inpol(x,func) = y/x
    # nu(y) = inpolinv(y,func) = x/y

    error = abs(EFE_M[1] / a0 - inpol(x, func) * x)
    if error < 0.001:
        return inpol(x, func) * x * a0
    else:
        i = 1
        while (EFE_M[1] / a0 - inpol(x * (1 + 1 / i), func) * x * (1 + 1 / i)) < 0:
            i += 1
        return Calculate_gN_gal(EFE_M, x * (1 + 1 / i), func)


def MainLoop(H, NDacc, func, EFE):  # This is the iteration loop. This calculates the MOND acceleration field from the Newtonian acceleration field. See thesis for information on why it works.
    # func refers to which interpolation function should be used.
    F = NDacc + H
    del H
    gM = inpolinv(np.linalg.norm(F, axis=0) / a0, func) * F  # might divide by zero
    del F
    gM2 = CurlFreeProj(gM[0], gM[1], gM[2])
    del gM
    if EFE[0] and EFE[1] == 1:
        gM2[2, :, :, :] += EFE[2]
    F = inpol(np.linalg.norm(gM2, axis=0) / a0, func) * gM2
    H = F - NDacc
    del F
    del NDacc
    H = DivFreeProj(H[0], H[1], H[2])
    return gM2, H


# %% Creating matrices related to the k vector

Kx = np.arange(-halfpixels, halfpixels, dtype=np.float32)[:, None, None] ** 2
Ky = np.arange(-halfpixels, halfpixels, dtype=np.float32)[:, None] ** 2
Kz = np.arange(-halfpixels, halfpixels, dtype=np.float32) ** 2

# KLM is a matrix where each entry is sum of the index's squared, or the sum of the function values of Kvect of the indices.
K2 = Kx + Ky + Kz
del Kx, Ky, Kz
K2 = np.roll(K2, halfpixels, axis=0)
K2 = np.roll(K2, halfpixels, axis=1)
K2 = np.roll(K2, halfpixels, axis=2)
K2[0, 0, 0] = 1
K2inv = 1 / K2
del K2

# The inproduct matrices are matrices where each entry is the x,y,z index, depending on if it is the x,y,z inproduct matrix.
inprodx = np.zeros([2 * halfpixels, 2 * halfpixels, 2 * halfpixels], dtype=np.float32)
inprody = np.zeros([2 * halfpixels, 2 * halfpixels, 2 * halfpixels], dtype=np.float32)
inprodz = np.zeros([2 * halfpixels, 2 * halfpixels, 2 * halfpixels], dtype=np.float32)
for i in np.roll(np.arange(-halfpixels, halfpixels), halfpixels):
    inprodx[i, :, :] = int(i)
    inprody[:, i, :] = int(i)
    inprodz[:, :, i] = int(i)

# %% Simulating and plotting: Two bodies
t_simulation_start = time.time()

if simulate_two_bodies:
    free_fall = 0  # 0 is static system, 1 is by shifting the positions each time step by 1/2*g*t^2 (not sure if correct, but may give insights into dynamics as compared to Newton dynamics)
# 2 is by keeping the center of mass in the middle (was used in simulations), 3 is static system but each timestep the particles are placed back to the origin (not sure if correct).


    m1, rx1, ry1, rz1, vx1, vy1, vz1 = 10, halfpixels * 6 / 8, halfpixels, halfpixels, halfpixels, halfpixels, 0
    m2, rx2, ry2, rz2, vx2, vy2, vz2 = 20, halfpixels * 9 / 8, halfpixels, halfpixels, -halfpixels, -halfpixels, 0

    particlelist = TwoBodyCircParticlelist(m1,m2,0.5*halfpixels,0)

    posmat, vecmat, AngMat, MomMat, EkinMat, EgravMat, EMat, COM = particlelist.TimeSim(timesteps, dt, itersteps, EFE_M, free_fall, regime)

    if free_fall == 3:
        posmat, vecmat, AngMat, MomMat, EkinMat, EgravMat, EMat= COMConverter(
            particlelist, posmat, vecmat, COM)

    # Orbit plot
    t_arr = np.linspace(0, T, timesteps)

    plt.figure(figsize=(7, 7))
    for i in range(len(particlelist.list[:,0])):
        plt.plot(posmat[i, :, 0] * size_of_box / (halfpixels * 2), posmat[i, :, 1] * size_of_box / (halfpixels * 2),
                 label="Orbit particle " + str(i))
    plt.xlabel("$x$ (ly)");
    plt.ylabel("$y$ (ly)")
    if free_fall == 0 or free_fall == 1 or free_fall == 2:
        plt.xlim(0, size_of_box);
        plt.ylim(0, size_of_box)
    else:
        plt.xlim(0, 2 * halfpixels);
    plt.grid()
    plt.legend()
    plt.savefig("Orbit.pdf")
    plt.show()

    # Velocity plot
    plt.figure(figsize=(7, 7))
    plt.plot(t_arr, vecmat[0, :, 1], 'k.')
    plt.plot(t_arr, vecmat[1, :, 1], 'r.')
    plt.xlabel("$T$ (Myr)");
    plt.ylabel("$v_y$")
    plt.grid()
    plt.show()

    # Energy plot
    plt.figure(figsize=(7, 7))
    plt.plot(t_arr, EMat, label="E total")
    plt.plot(t_arr, EkinMat, label="E kin")
    plt.plot(t_arr, EgravMat, label="E grav")
    plt.plot(t_arr, EMat - EkinMat - EgravMat, label="E Pot", zorder=1)
    plt.xlabel("Time (Myr)");
    plt.ylabel("Energy")
    plt.legend()

    plt.savefig("Energy.pdf")
    plt.show()
