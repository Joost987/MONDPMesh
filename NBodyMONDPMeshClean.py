# %%
import os

import numpy as np
import scipy
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp
import time
from scipy.special import hyp2f1
import datetime

import sys
sys.modules.setdefault("NBodyMONDPMeshClean", sys.modules[__name__])
# %% Simulation parameters and physical constants

# Some of the simulation parameters. You can change halfpixels, which is half the number of pixels in one dimension of the grid
# You can also change celllen, which is the distance between neighbouring pixels. Some other constants are defined, as this ensures that these calculations are only done once.
halfpixels = 32  # For optimal FFT's, this has to be a power of 2.

datatype = np.float32  # Choose precision

#Simulation parameters
N_particles = 10**4  # Amount of particles inside isothermal sphere
M = 2361.590348335877 # Total mass of isothermal sphere
b = 27.086544078042834  # Effective radius (ly)
size_of_box = 20 * b  # ly
# T is total simulated time
T = 30  # Myr
UseOrbit = True #whether to use T or the orbit time
orbits = 1 #integer number of circular orbits at radius b at the MOND average velocity

simulate_sphere = True #Simulates an isothermal sphere and plots the projected density when set to false, a fit is made to the data of the cluster selected below

# Which cluster:
#options : Data taken from Holger Baumgart globular cluster database https://people.smp.uq.edu.au/HolgerBaumgardt/globular/
#AM1 : M=2.0 · 10^4 M☉, HMR=20.32 pc
#ngc2419 : M=7.8 · 105 M☉, HMR=26.37 pc
#ngc5139 : 3.94 · 106 M☉, HMR=10.42 pc

name='am1'

show_data = True #Plots density measurements of the selected cluster
fit_to_data = True #finds and prints optimal parameters for simulation and plots a fitted curve to the data (use this to determine proper parameters before actually simulating)
if simulate_sphere:
    fit_to_data=False
mfit = True #for fitting the initial condition parameters to the data,
            #we can use the mass from the database (Set in simulation parameters) or allow python to determine an optimal mass to fit to Plummer curve

# Some physical constants
G = 6.674 * 10 ** (-11)  # m^3/(s^2*kg)
G = 3.942 * 10 ** 7  # au^3/(kyr^2*M☉)
G = 0.156  # ly^3/(Myr^2*M☉)
a0 = 1.2 * 10 ** (-10)  # m/s^2
a0 = 0.7978  # au/kyr^2
a0 = 12.614  # ly/Myr^2

if simulate_sphere:
    if UseOrbit:
        T = int(np.ceil(orbits * b * 2 * np.pi / (np.sqrt(2 / 3 * np.sqrt(G * M * a0)))))

EFE_on = True # External field effect on or off
if name == 'am1':
    R_mw = 120300*3.27 # ly
if name == 'ngc2419':
    R_mw = 95900*3.27
if name == 'ngc5139':
    R_mw = 6500*3.27
EFE_M_strength = G*(6*10**10) / (R_mw)**2 #ly/Myr^2
EFE_M = (EFE_on, (0.0, 0.0, float(EFE_M_strength)))

itersteps = 4  # Number of iterations used for the potential.
regime = 6  # Which interpolation function should be used? Regime 2,3 don't work currently.
# Different values for regime parameter
# 0: Deep MOND
# 1: Standard
# 2: McGaugh (Slower, EGrav does not work)
# 3: Bose-Einstein (Slower)
# 4: Verlinde
# 5: Newton
# 6: de Sitter
# 2,3 are slower as either the interpolation function, or its inverse, does not have an expression in terms of elementary functions
# These are therefore calculated using a Newton Raphson method, which is of course slower than using an elementary function
# For regime 2, the gravitational energy calculation has not been implemented, and EGrav=0 is used everywhere.
MAXNR_ITERATIONS = 10  # ONLY USED FOR REGIME=2 OR REGIME=3. Maximum number of iterations the Newton-Rhapson solver does.

c = 4 * jnp.pi * G
shape = (2 * halfpixels, 2 * halfpixels, 2 * halfpixels)

celllen = size_of_box / (2 * halfpixels)
cellleninv = 1 / celllen
size = halfpixels * celllen
kstep = jnp.pi / (halfpixels * celllen)
kstep2inv = 1 / kstep ** 2

cellvolume = celllen ** 3
cellvolumeinv = 1 / cellvolume
oversqrt2pi = 1 / jnp.sqrt(2 * jnp.pi)
oversqrt2pi3 = oversqrt2pi ** 3

# a0 = 1.2*10^-10 m/s^2 = 0.7978 au/kyr^2 = 12.614 Mly/Gyr^2
# Acceleration from Milky Way on Sun = 3.47*10^-9 m/s^2 = 23.07 au/kyr^2 = 364.8 Mly/Gyr^2
# Newtons gravitational constant = 6.674*10^-11 m^3/(s^2*kg) = 3.942*10^7 au^3/(kyr^2*M☉) = 0.235 Mly^3/(Gyr^2*M_MW)

b_grid = b / celllen  # to correctly keep the effective radius when changing pixel size and grid size.
allowNewtonCorrections = False  # Do Newton corrections
dt = min(np.sqrt(np.sqrt(celllen**4/(G* M/N_particles * a0))), np.sqrt(2*cellvolume / (G * M/N_particles)))/32 # Determining an efficient timestep in Newton and Deep mond and choosing the minimum
timesteps = int(np.ceil(T / dt))
dt = T / timesteps



def FindBall(N):
    ball = np.array([])
    for i in range(-N + 1, N):
        for j in range(-N + 1, N):
            for k in range(-N + 1, N):
                if i ** 2 + j ** 2 + k ** 2 < N ** 2:
                    ball = np.append(ball, [i, j, k])
    ball = np.reshape(ball, (251, 3)).astype("int32")
    return jnp.array(ball)


ball4 = FindBall(4)


def AnalyticGravitationalPotential(r, r0, m, EFE):
    return -jax.scipy.special.erf(jnp.linalg.norm(r - r0) / jnp.sqrt(2)) / jnp.linalg.norm(r - r0) * m * G


@jax.jit
def AssignMassGaussShapeJAX(density, particlelist, cellvolumeinv, a=1,
                            shape=ball4):  # N is order of the method divided by 2, amount of points used is (2N)^3
    # For a=1, N>3, a=1.5 N>4, a=2 N>5, a=3 N>8.

    a3inv = (1 / a) ** 3

    x = particlelist[:, 1]
    y = particlelist[:, 2]
    z = particlelist[:, 3]

    cellcoords = (particlelist[:, jnp.newaxis, 1].astype(int) + shape[jnp.newaxis, :, 0],
                  particlelist[:, jnp.newaxis, 2].astype(int) + shape[jnp.newaxis, :, 1],
                  particlelist[:, jnp.newaxis, 3].astype(int) + shape[jnp.newaxis, :, 2])
    weight = oversqrt2pi3 * jnp.exp(-(
                (cellcoords[0] - x[:, jnp.newaxis]) ** 2 + (cellcoords[1] - y[:, jnp.newaxis]) ** 2 + (
                    cellcoords[2] - z[:, jnp.newaxis]) ** 2) / (2 * a ** 2))

    idx_x = cellcoords[0].ravel()
    idx_y = cellcoords[1].ravel()
    idx_z = cellcoords[2].ravel()
    vals = (particlelist[:, jnp.newaxis, 0] * weight * cellvolumeinv * a3inv).ravel()
    density = density.at[idx_x, idx_y, idx_z].add(vals)
    return density


shape_ball = ball4


@jax.jit
def AssignAccsGaussShapeJAX(accparts, accmat, particlelist2, a=1, shape=ball4):  # Partially written by chatgpt
    a3inv = (1 / a) ** 3

    x = particlelist2[:, 1]
    y = particlelist2[:, 2]
    z = particlelist2[:, 3]

    # Compute all cell coordinates
    cellcoords_x = particlelist2[:, jnp.newaxis, 1].astype(int) + shape[jnp.newaxis, :, 0]
    cellcoords_y = particlelist2[:, jnp.newaxis, 2].astype(int) + shape[jnp.newaxis, :, 1]
    cellcoords_z = particlelist2[:, jnp.newaxis, 3].astype(int) + shape[jnp.newaxis, :, 2]

    # Compute weights
    dx = cellcoords_x - x[:, None]
    dy = cellcoords_y - y[:, None]
    dz = cellcoords_z - z[:, None]
    weight = oversqrt2pi3 * jnp.exp(-(dx ** 2 + dy ** 2 + dz ** 2) / (2 * a ** 2)) * a3inv

    # Gather accmat values for all particles
    acc_x = accmat[0, cellcoords_x, cellcoords_y, cellcoords_z]  # (num_particles, num_shape_points)
    acc_y = accmat[1, cellcoords_x, cellcoords_y, cellcoords_z]
    acc_z = accmat[2, cellcoords_x, cellcoords_y, cellcoords_z]

    # Multiply by weights and sum over shape points
    accparts = accparts.at[:, 0].add((acc_x * weight).sum(axis=1))
    accparts = accparts.at[:, 1].add((acc_y * weight).sum(axis=1))
    accparts = accparts.at[:, 2].add((acc_z * weight).sum(axis=1))

    return accparts


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
        self.list = jnp.array(particlelist, dtype=datatype)

    def Ekin(self):
        return jnp.dot(1 / 2 * self.list[:, 0], jnp.diagonal(self.list[:, 4:7] @ jnp.transpose(
            self.list[:, 4:7]))) * celllen ** 2  # take the dot product between 1/2*masses and the velocities squared.
        # the self.list[:,4:7]@np.transpose(self.list[:,4:7]) part creates a matrix with all of the velocities of each particle multiplied by each other
        # we only want the velocities of each particle squared, so we take the diagonal of this.

    def AngMom(self):
        return jnp.sum(
            jnp.diag(self.list[:, 0]) @ jnp.cross(self.list[:, 1:4] - jnp.array([halfpixels] * 3), self.list[:,
            4:7]))  # angular momentum with [halfpixels,halfpixels,halfpixels] as origin.

    def CenterOfMass(self):
        return 1 / jnp.sum(self.list[:, 0]) * (self.list[:, 0] @ self.list[:, 1:4]), 1 / jnp.sum(self.list[:, 0]) * (
                self.list[:, 0] @ self.list[:, 4:7])

    def UpdateAccsMOND(self, particlelist, EFE, shape_ball=ball4, sigma=1, iterlen=4, regime=0):
        density = jnp.zeros((2 * halfpixels, 2 * halfpixels, 2 * halfpixels), dtype=datatype)
        density = AssignMassGaussShapeJAX(density, particlelist, cellvolumeinv, sigma, shape_ball)
        densityfft = jnp.fft.fftn(density)
        potNDmat = Calcpot(densityfft)
        del densityfft

        # CalcAccMat gives only the internal Newtonian field g_N_int,
        # derived from the cluster's own mass density via Poisson's equation.
        accNDmat = CalcAccMat(potNDmat)
        del potNDmat

        if EFE[0]:
            # Add the galaxy's Newtonian field g_N_e to get the total
            # Newtonian field: accNDmat = g_N_int + g_N_e
            # Shape (3,) -> (3,1,1,1) to broadcast over the grid.
            g_N_e = jnp.array(EFE[1], dtype=datatype)[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
            accNDmat = accNDmat + g_N_e

        H = jnp.zeros([3, 2 * halfpixels, 2 * halfpixels, 2 * halfpixels], dtype=datatype)
        for i in range(iterlen):
            accMONDmat, H = MainLoop(H, accNDmat, regime, EFE)

        accMONDmatfft = jnp.fft.fftn(accMONDmat, axes=(1, 2, 3))
        del accMONDmat
        potMONDmatfft = -KdotProd(accMONDmatfft) * K2inv / kstep
        del accMONDmatfft
        potMONDmat = jnp.imag(jnp.fft.ifftn(potMONDmatfft))

        del potMONDmatfft
        accMONDmat = CalcAccMat(potMONDmat)

        Egrav, EGravPot = EGrav(accMONDmat, H + accNDmat, regime)

        EPot = jnp.sum(potMONDmat * density) * cellvolume
        del H
        del accNDmat
        del density
        del potMONDmat
        accparts = jnp.zeros((len(particlelist), 3), dtype=datatype)
        accparts = AssignAccsGaussShapeJAX(accparts, accMONDmat, particlelist, sigma, shape_ball)
        return accparts, Egrav, EPot

    UpdateAccsMOND = jax.jit(UpdateAccsMOND, static_argnames=["self", "EFE", "iterlen", "regime"])

    def NewtonCorrection(self, particlelist, accelerations, func, correction_criteria, std):

        # Assume accelerations is an array of the form [[ax, ay, az], ... ]
        acceleration_norms = jnp.sqrt(jnp.sum(accelerations ** 2, axis=1))
        newtoncheck = inpol(x=acceleration_norms, func=func)

        correction_check = jnp.array([newtoncheck >= correction_criteria])[0]

        for index1, particle1 in enumerate(particlelist):
            if correction_check[index1] == False: continue
            correction = jnp.zeros(3)
            for index2, particle2 in enumerate(particlelist):
                if index1 == index2: continue
                dist_vec = particle1[1:4] - particle2[1:4]
                dist = jnp.linalg.norm(dist_vec)

                intermediatefactor = scipy.special.erf(dist / (2 * std)) / dist ** 2 \
                                     - jnp.exp(-dist ** 2 / (4 * std ** 2)) / (jnp.sqrt(jnp.pi) * dist * std)
                correction += intermediatefactor * dist_vec / dist * G * particle2[0] * cellleninv
                # TODO: check for division by 0 issues
                correction += (-G * particle2[0]) / dist ** 3 * dist_vec * cellleninv ** 2

            accelerations[index1] += correction

        return accelerations

    def TimeSim(self, timesteps, dt, itersteps, EFE, free_fall, regime=0):
        posmat = jnp.zeros([len(self.list), timesteps, 3], dtype=datatype)
        vecmat = jnp.zeros([len(self.list), timesteps, 3], dtype=datatype)
        accmat = jnp.zeros([len(self.list), timesteps, 3], dtype=datatype)

        MomMat = jnp.zeros([timesteps, 3], dtype=datatype)
        AngMat = jnp.zeros([timesteps, 3], dtype=datatype)
        EkinMat = jnp.zeros([timesteps], dtype=datatype)
        EgravMat = jnp.zeros([timesteps], dtype=datatype)
        EMat = jnp.zeros([timesteps], dtype=datatype)
        EpotMat = jnp.zeros([timesteps], dtype=datatype)

        accnew, Egrav, Epot = self.UpdateAccsMOND(self.list, EFE, iterlen=4, regime=regime)




        # Jax arrays are immutable, and therefore are copied to a new array when altered.
        # By using JIT, these copy operations should be removed
        @jax.jit
        def UpdateMats(t, posmat, vecmat, accmat, AngMat, MomMat, EkinMat,EpotMat, EgravMat, EMat, particlelist, AngMom, Ekin,
                       Egrav, Epot, acc):

            posmat = posmat.at[:, t, :].set(particlelist[:, 1:4])
            vecmat = vecmat.at[:, t, :].set(particlelist[:, 4:7])
            accmat = accmat.at[:, t, :].set(acc)

            AngMat = AngMat.at[t, :].set(AngMom)
            MomMat = MomMat.at[t, :].set(jnp.transpose(particlelist[:, 4:7]) @ particlelist[:, 0])
            EkinMat = EkinMat.at[t].set(Ekin)
            EgravMat = EgravMat.at[t].set(Egrav)
            EMat = EMat.at[t].set(Egrav + Epot + Ekin)
            EpotMat = EpotMat.at[t].set(Epot)
            return posmat, vecmat, accmat, AngMat, MomMat, EkinMat, EpotMat, EgravMat, EMat

        if allowNewtonCorrections:
            # TODO: explain the 0.99.
            accnew = self.NewtonCorrection(self.list, accnew, regime, 0.99, 1)

        COM = jnp.zeros([7, timesteps], dtype=datatype)

        prev = time.time()
        for t in range(timesteps):

            posmat, vecmat, accmat, AngMat, MomMat, EkinMat, EpotMat, EgravMat, EMat = UpdateMats(t, posmat, vecmat, accmat,
                                                                                         AngMat, MomMat, EkinMat, EpotMat,
                                                                                         EgravMat, EMat, self.list,
                                                                                         self.AngMom(), self.Ekin(),
                                                                                         Egrav, Epot, accnew)

            accold = accnew
            self.list = self.list.at[:, 1:4].add(self.list[:,
            4:7] * dt + 0.5 * accold * cellleninv * dt ** 2)  # Leapfrog without half integer time steps

            try:  # If the particles are outside of the grid this will raise an error. This catches this error
                # and breaks the loop, ensuring that the data from before the error can be returned.
                accnew, Egrav, Epot = self.UpdateAccsMOND(self.list, EFE, iterlen=itersteps, regime=regime)
                if allowNewtonCorrections:
                    # TODO: explain the 0.99.
                    accnew = self.NewtonCorrection(self.list, accnew, regime, 0.99, 1)
            except:  # different ways of handling this exception can be made. For the isothermal sphere for example
                # the particles will enter
                print("particle outside box")
                break
            self.list = self.list.at[:, 4:7].add((accold + accnew) * 0.5 * dt * cellleninv)
            every_n_timesteps = max(int(timesteps/10), 1)
            if t % every_n_timesteps == 0:
                print("Timesteps done:", t, '/', timesteps)
                if t != 0:
                    now = time.time()
                    est_time_sec = int((timesteps - t) / every_n_timesteps * (now - prev))
                    print('Estimated time remaining : ', datetime.timedelta(seconds=est_time_sec))
                    prev = now
        return posmat, vecmat, accmat, AngMat, MomMat, EkinMat, EpotMat, EgravMat, EMat, COM


# %% Functions

# Mondian acceleration between two bodies
def Body2MOND(x, y, m1, m2):
    M = m1 + m2
    return 2 / 3 * jnp.sqrt(G * a0) / (celllen * jnp.linalg.norm(x[0:2] - y[0:2], axis=0)) * (
            M ** (3 / 2) - m1 ** (3 / 2) - m2 ** (3 / 2))


# Calculate potential from Fourier transformed density.
@jax.jit
def Calcpot(densityfft):
    potmatfft = -c * densityfft * K2inv / kstep ** 2  # *kstep2inv #kstep2inv is 1/kstep**2
    del densityfft
    potmat = jnp.fft.ifftn(potmatfft, s=shape)  # inverse Fourier Transform
    del potmatfft
    potmat = jnp.real(potmat)
    return potmat


# Use finite differences to calculate the acceleration field on the grid.
@jax.jit
def CalcAccMat(potmat):
    accmat = jnp.array([(jnp.roll(potmat, 1, axis=0) - jnp.roll(potmat, -1, axis=0)) / (2 * celllen),
                        (jnp.roll(potmat, 1, axis=1) - jnp.roll(potmat, -1, axis=1)) / (2 * celllen),
                        (jnp.roll(potmat, 1, axis=2) - jnp.roll(potmat, -1, axis=2)) / (2 * celllen)])
    return accmat


def COMConverter(particles, pos, vec,
                 COM):  # COMConverter = Center Of Mass Converter. This function converts the coordinates where the COM was kept constant, to the coordinates where the starting position is constant
    posmat = jnp.zeros([len(particles.list), len(COM[0]), 3])
    vecmat = jnp.zeros([len(particles.list), len(COM[0]), 3])
    AngMat = jnp.zeros(len(COM[0]))
    MomMat = jnp.zeros(len(COM[0]))
    EkinMat = jnp.zeros(len(COM[0]))
    EMat = jnp.zeros(len(COM[0]))
    for t in range(len(COM[0])):
        vecmat[:, t, :] = vec[:, t, :] + jnp.sum(COM[4:7, 0:t], axis=1)
        posmat[:, t, :] = pos[:, t, :] + jnp.sum(COM[1:4, 0:t], axis=1) - [(t) * halfpixels, (t) * halfpixels,
                                                                           (t) * halfpixels]
        for i in range(t):
            posmat[:, t, :] += dt * jnp.sum(COM[4:7, 0:(t - i)], axis=1)
        EkinMat[t] = 0.5 * jnp.dot(particles.list[:, 0],
                                   jnp.diagonal(vecmat[:, t, :] @ jnp.transpose(vecmat[:, t, :]))) * celllen ** 2
        for i in range(1, t):
            EMat[t] += (COM[0, i] - COM[0, i - 1])
            print((COM[0, i] - COM[0, i - 1]))
        EMat[t] += COM[0, t]
    return posmat, vecmat, AngMat, MomMat, EkinMat, EMat


@jax.jit
def KdotProd(
        A):  # Dot product of a vector field with k vector. K vector is an element of the Fourier transformed domain.
    return (inprodx * A[0] + inprody * A[1] + inprodz * A[2])


def inpol(x, func):  # Interpolation function \mu
    if func == 0: return x  # Deepmond
    if func == 1: return x / jnp.sqrt(1 + x ** 2)  # Standard
    if func == 2: return FindMu(lambda y: inpolinv(y, func), lambda y: der_inpolinv(y, func), x,
                                max_iterations=MAXNR_ITERATIONS)  # McGaugh
    if func == 3: return 1 - jnp.exp(-x)  # Bose-Einstein
    if func == 4: return 4 * x / (1 + jnp.sqrt(1 + 4 * x)) ** 2  # Verlinde
    if func == 5: return 1  # Newton
    if func == 6: return (jnp.sqrt(1 + 4 * x ** 2) - 1) / (2 * x)  # DeSitter


def inpolinv(y, func):  # Inverse interpolation function \nu
    if func == 0: return 1 / jnp.sqrt(y)  # Deepmond
    if func == 1: return jnp.sqrt(1 / 2 + 1 / 2 * jnp.sqrt(1 + 4 / y ** 2))  # Standard
    if func == 2: return 1 / (1 - jnp.exp(-jnp.sqrt(y)))  # McGaugh
    if func == 3: return FindNu(lambda x: inpol(x, func), lambda x: der_inpol(x, func), y,
                                max_iterations=MAXNR_ITERATIONS)  # Bose-Einstein
    if func == 4: return 1 + 1 / jnp.sqrt(y)  # Verlinde
    if func == 5: return 1  # Newton
    if func == 6: return jnp.sqrt(1 + 1 / y)  # DeSitter


# IMPORTANT: If you add interpolation functions or inverse interpolation function where the other is not known
# and you want to use the Newton Raphson method, you should first check that their derivative is never 0.
# Else, if the initial value for the Newton Raphson method gives derivative 0, the method will not converge.

def der_inpol(x,
              func):  # Derivative of interpolation function \mu. Only needed when \nu does not have an explicit expression
    if func == 3: return jnp.exp(-x)


def der_inpolinv(y,
                 func):  # Derivative of inverse interpolation function \nu. Only needed when \mu does not have an explicit expression
    if func == 2: return - jnp.exp(-jnp.sqrt(y)) / (2 * jnp.sqrt(y) * (1 - jnp.exp(-jnp.sqrt(y))))


def NewtonRaphson(func, derfunc, init, rtol=1e-3, max_iterations=20):
    def cond(state):
        currval, initfunc, init, counter = state
        return jnp.logical_and(jnp.all(jnp.abs(currval / initfunc) > rtol), counter < max_iterations)

    def body(state):
        currval, initfunc, init, counter = state
        derivative = derfunc(init)
        init = init - currval / derivative
        currval = func(init)
        counter += 1
        return currval, initfunc, init, counter

    initfunc = func(init)
    currval = initfunc
    counter = jnp.array(0)

    currval, initfunc, init, counter = jax.lax.while_loop(cond, body, (currval, initfunc, init, counter))

    return init


def FindMu(nu, der_nu, x, tol=1e-3, max_iterations=20):
    return NewtonRaphson(lambda mu: mu * nu(x * mu) - 1, lambda mu: nu(x * mu) + mu * der_nu(x * mu) * x, rtol=tol,
                         max_iterations=max_iterations)


def FindNu(mu, der_mu, y, tol=1e-3, max_iterations=20):
    return NewtonRaphson(lambda nu: nu * mu(y * nu) - 1, lambda nu: mu(y * nu) + nu * der_mu(y * nu) * y,
                         jnp.sqrt(1 / y), rtol=tol, max_iterations=max_iterations)


def EGrav(accMONDmat, F, func):
    EGrav = 0
    EGravPot = 0  # Sum of potential energy and gravitational energy
    x = jnp.linalg.norm(accMONDmat, axis=0) / a0
    y = jnp.linalg.norm(F, axis=0) / a0
    if func == 0:
        EGrav = jnp.sum(x ** 3 / 3)
    if func == 1:
        EGrav = jnp.sum((x * jnp.sqrt(1 + x ** 2) - jnp.arcsinh(x)) / 2)
    if func == 2:
        eps = 1e-8
        V_prime = lambda y0, V: y0 / (1 - np.exp(-np.sqrt(y0)))
        EGravPot = 0  # np.sum(scipy.integrate.odeint(V_prime,eps,np.append(np.array(eps),np.sort(y.flatten())),tfirst=True))
        # We numerically integrate V_prime over y to find V. Afterwards we integrate V over space to find E_grav+E_pot.
        # The numerical integration over y is done by first sorting the y array. As we will integrate over real spaces, the order of the y array does not matter
        # Then we solve the ode dV/dy=V_prime. Scipy will solve this ode and give us the value of the integral for all points
        # specified.
        EGravPot = jnp.asarray(EGravPot)
    if func == 3:
        EGrav = jnp.sum(x ** 2 / 2 + (x + 1) * jnp.exp(-x))
    if func == 4:
        EGrav = jnp.sum((x + x ** 2) / 2 - (1 + 4 * x) ** (3 / 2) / 12)
        EGravPot = jnp.sum(y ** 2 / 2 + 2 * y ** (3 / 2) / 3)
    if func == 5:
        EGrav = jnp.sum(x ** 2 / 2)
    if func == 6:
        EGrav = jnp.sum((2 * x * (jnp.sqrt(4 * x ** 2 + 1) - 2) + jnp.arcsinh(2 * x)) / 8)
    return a0 ** 2 / (4 * jnp.pi * G) * EGrav * cellvolume, -a0 ** 2 / (4 * jnp.pi * G) * EGravPot * cellvolume


@jax.jit
def CurlFreeProj(Ax, Ay, Az):  # Calculates the curl free projection of the vector field A = [Ax,Ay,Az] using FFT's
    A = jnp.array([Ax, Ay, Az])
    del Ax, Ay, Az
    Ahat = jnp.fft.fftn(A, s=shape)
    del A
    intermediatestep = K2inv * KdotProd(Ahat)
    del Ahat
    xyz = jnp.fft.ifftn(jnp.array([intermediatestep * inprodx, intermediatestep * inprody, intermediatestep * inprodz]),
                        s=shape)
    del intermediatestep
    return xyz


@jax.jit
def DivFreeProj(Ax, Ay, Az):  # Calculates the divergence free projection of the vector field A = [Ax,Ay,Az] using FFT's
    A = jnp.array([Ax, Ay, Az])
    del Ax, Ay, Az
    Ahat = jnp.fft.fftn(A, s=shape, axes=(1, 2, 3))
    del A
    intermediatestep = K2inv * KdotProd(Ahat)
    fourier_matrix = jnp.array([Ahat[0] - intermediatestep * inprodx, Ahat[1] - intermediatestep * inprody, Ahat[2] - intermediatestep * inprodz])
    xyz = jnp.fft.ifftn(fourier_matrix, s=shape, axes=(1, 2, 3))
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


def MainLoop(H, NDacc, func, EFE):
    F = NDacc + H
    gM = inpolinv(jnp.linalg.norm(F, axis=0) / a0, func) * F
    gM2 = CurlFreeProj(gM[0], gM[1], gM[2])

    #restoring uniform part (important for EFE)
    if EFE_on:
        gM_mean = jnp.mean(gM, axis=(1, 2, 3))  # shape (3,)
        gM2 = gM2 + gM_mean[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]

    F = inpol(jnp.linalg.norm(gM2, axis=0) / a0, func) * gM2
    H = F - NDacc
    H = DivFreeProj(H[0], H[1], H[2])
    return gM2, H


# %% Creating matrices related to the k vector

Kx2 = jnp.arange(-halfpixels, halfpixels, dtype=datatype)[:, None, None] ** 2
Ky2 = jnp.arange(-halfpixels, halfpixels, dtype=datatype)[:, None] ** 2
Kz2 = jnp.arange(-halfpixels, halfpixels, dtype=datatype) ** 2

# KLM is a matrix where each entry is sum of the index's squared, or the sum of the function values of Kvect of the indices.
K2 = Kx2 + Ky2 + Kz2
del Kx2, Ky2, Kz2
K2 = jnp.roll(K2, halfpixels, axis=0)
K2 = jnp.roll(K2, halfpixels, axis=1)
K2 = jnp.roll(K2, halfpixels, axis=2)
K2 = K2.at[0, 0, 0].set(1)
K2inv = 1 / K2
del K2

# The inproduct matrices are matrices where each entry is the x,y,z index, depending on if it is the x,y,z inproduct matrix.
inprodx = np.zeros([2 * halfpixels, 2 * halfpixels, 2 * halfpixels], dtype=datatype)
inprody = np.zeros([2 * halfpixels, 2 * halfpixels, 2 * halfpixels], dtype=datatype)
inprodz = np.zeros([2 * halfpixels, 2 * halfpixels, 2 * halfpixels], dtype=datatype)
for i in np.roll(np.arange(-halfpixels, halfpixels), halfpixels):
    inprodx[i, :, :] = int(i)
    inprody[:, i, :] = int(i)
    inprodz[:, :, i] = int(i)

inprodx, inprody, inprodz = jnp.array(inprodx), jnp.array(inprody), jnp.array(inprodz)
# %% Simulating and plotting
# Import after Particlelist/constants are defined to avoid circular import with IsothermalClass.

if __name__ == "__main__":
    from IsothermalClass import IsoThermalParticlelist
    if simulate_sphere:
        print('Simulating', name.upper())
        if UseOrbit:
            T = int(np.ceil(orbits * b * 2 * np.pi / (np.sqrt(2 / 3 * np.sqrt(
                G * M * a0)))))  # determined orbit time using radius b and the MOND average velocity sqrt(v2)

            print('Simulating', orbits, 'orbits,', T, 'Myr')  # $\frac{2 \pi b}{\sqrt(\overline{v^2})}=$
        print('Timestep length ', format(int(dt * 10 ** 6), ','), 'yr')
        if EFE_on:
            print('External field effect strength:', round(EFE_M_strength/a0, 4), 'a0')
        else:
            print('No external field')

        free_fall = 0  # 0 is static system, 1 is by shifting the positions each time step by 1/2*g*t^2 (not sure if correct, but may give insights into dynamics as compared to Newton dynamics)
        # 2 is by keeping the center of mass in the middle (was used in simulations), 3 is static system but each timestep the particles are placed back to the origin (not sure if correct).
        particlelist = IsoThermalParticlelist(M / N_particles, b, N_particles)
        # particlelist = RejectionParticlelist(M, b, N_particles)
        posmat, vecmat, accmat, AngMat, MomMat, EkinMat,EpotMat, EgravMat, EMat, COM = particlelist.TimeSim(timesteps, dt,
                                                                                                        itersteps, EFE_M,
                                                                                                        free_fall, regime)
        N_particles = len(posmat)

        ### 3D Cumulative mass and density plot generation ###
        # Setup for density histogram
        number_of_bins = max(int(N_particles / 100), 100)

        rvec = (posmat - np.array([[[halfpixels] * 3] * timesteps] * N_particles)) * celllen
        r = np.sqrt(rvec[:, :, 0] ** 2 + rvec[:, :, 1] ** 2 + rvec[:, :, 2] ** 2)
        rprojz = np.sqrt(rvec[:, :, 0] ** 2 + rvec[:, :, 1] ** 2)
        rprojy = np.sqrt(rvec[:, :, 0] ** 2 + rvec[:, :, 2] ** 2)  # projecting on the three cardinal planes in order to increase the amount of data points.
        rprojx = np.sqrt(rvec[:, :, 1] ** 2 + rvec[:, :, 2] ** 2)
        rproj = np.concatenate([rprojx, rprojy, rprojz])
        if EFE_on:
            rproj = rprojz = np.sqrt(rvec[:, :, 0] ** 2 + rvec[:, :, 1] ** 2) #This is the axis that we see from the center of the galaxy
        m = M / N_particles
        m_arr = np.array([m] * N_particles)
        m_arr_proj = np.array([m] * len(rproj)) / 3 #dividing by three to account for the 3 plane approach
        if EFE_on:
            m_arr_proj = np.array([m] * len(rproj))
        #del rvec
        rs = r[:, 0]
        rf = r[:, -1]
        rsproj = sorted(rproj[:, 0])
        rfproj = sorted(rproj[:, -1])
        #print('Quarter mass radius generated = ', sorted(rs)[int(len(rs) / 4)], 'ly')
        #print('Half mass radius generated = ', sorted(rs)[int(len(rs) / 2)], 'ly')
        #print('Quarter mass radius after simulation ', sorted(rf)[int(len(rf) / 4)], 'ly')
        print('QMR growth over simulation : ', sorted(rf)[int(len(rf) / 4)] - sorted(rs)[int(len(rs) / 4)])
        # Making a cutoff to keep higher resolution in histograms cutoff is same as initial furthest point
        rf_sort = sorted(rf)
        rf_cutoff = []

        for i in range(len(rf)):
            if rf_sort[i] > 10*max(rs):
                break
            else:
                rf_cutoff.append(rf_sort[i])

        # Histogram of initial conditions:
        ms, rsbin_edges = np.histogram(rs, bins=number_of_bins, weights=m_arr, density=False)
        rsbins = 0.5 * (rsbin_edges[:-1] + rsbin_edges[1:])  # bin centers
        rsmean = np.sum(rsbins * ms) / np.sum(ms)
        ms_cum = np.cumsum(ms) / M
        msproj, rsprojbin_edges = np.histogram(rsproj, bins=number_of_bins, weights=m_arr_proj, density=False)
        rsprojbins = 0.5 * (rsprojbin_edges[1:] + rsprojbin_edges[:-1])

        # Histogram after simulating
        mf, rfbin_edges = np.histogram(rf_cutoff, bins=number_of_bins, weights=m_arr[:len(rf_cutoff)], density=False)
        rfbins = 0.5 * (rfbin_edges[:-1] + rfbin_edges[1:])
        mf_cum = np.cumsum(mf) / M
        mfproj, rfprojbin_edges = np.histogram(rfproj, bins=number_of_bins, weights=m_arr_proj, density=False)
        rfprojbins = 0.5 * (rfprojbin_edges[1:] + rfprojbin_edges[:-1])
        plotcum = False
        if plotcum:
            plt.plot(rsbins, ms_cum, label='Starting conditions')
            plt.plot(rfbins, mf_cum, label='After simulation')  # Cumulative plot after simulating
            plt.xlabel('r (ly)')
            plt.ylabel('Fraction of mass contained within a sphere of radius r')
            plt.legend()
            plt.savefig('cumplot.pdf')
            plt.show()

        shellvols = 4 / 3 * np.pi * (rsbin_edges[1:] ** 3 - rsbin_edges[:-1] ** 3)
        shellvolf = 4 / 3 * np.pi * (rfbin_edges[1:] ** 3 - rfbin_edges[:-1] ** 3)
        rhos = ms / shellvols
        rhof = mf / shellvolf

        ringareas = np.pi * (rsprojbin_edges[1:] ** 2 - rsprojbin_edges[:-1] ** 2)
        ringareaf = np.pi * (rfprojbin_edges[1:] ** 2 - rfprojbin_edges[:-1] ** 2)
        sigmas = msproj / ringareas
        sigmaf = mfproj / ringareaf
        sigma0 = sigmas[0]
        if simulate_sphere:
            # plotting simulation data
            plt.plot(rsprojbins, sigmas, ms=2, label='initial condition')
            if not UseOrbit:
                plt.plot(rfprojbins, sigmaf, ms=2, label='after simulating ' + str(T) + ' Myr')
            else:
                plt.plot(rfprojbins, sigmaf, ms=2, label='after simulating ' + str(orbits) + r' $T_{\text{orbit}}$')

    def plotdata(name):
        """
        Load radial magnitude-density (mag/arcsec^2) from CSV,
        convert to mass surface density (Msun/lightyear^2),
        and plot mass density vs radius (linear scale).
        """
        PATH = os.path.dirname(os.path.abspath(__file__))
        filename = PATH + '/' + name + '.csv'
        data = np.loadtxt(filename, delimiter=',')
        # Radius from cluster center
        if name == 'AM1' or name == 'am1':
            R_earth_cluster = 387799.933  # [ly].  source : globular cluster database university of queensland
            M_to_L = 0.8
        if name == 'ngc2419':
            R_earth_cluster = 88500*3.26
            M_to_L = 1.6
        if name == 'ngc5139':
            R_earth_cluster = 5430*3.26
            M_to_L = 2.8
        global r_data
        r_data = [10 ** data[i][0] * R_earth_cluster / 206265 for i in range(len(data))]

        # Magnitude surface density µ (mag/arcsec^2)
        mu = [data[i][1] for i in range(len(data))]

        # Solar absolute magnitudes
        M_sun = 4.83  # V-band magnitude
        const = 19.01  # converts arcsec^2 to ly^2

        # Convert µ → luminosity surface density (Lsun/pc^2)
        luminosity = [10 ** (-0.4 * (mu[i] - M_sun - const)) for i in range(len(mu))]

        # Convert luminosity → mass surface density (Msun/pc^2)
        mass_density = [luminosity[i] * M_to_L for i in range(len(luminosity))]
        global mass_density_ly2
        mass_density_ly2 = [mass_density[i] for i in range(len(mass_density))]

        global mass_integral
        mass_integral = [0]
        for i in range(1, len(mass_density_ly2)):
            mass_integral.append(mass_integral[i-1] + np.pi * mass_density_ly2[i-1] * (r_data[i] ** 2 - r_data[i-1] ** 2))
        mass_integral = np.array(mass_integral)
        #print('Mass integral for ' + str(name) + ' : ' + str(mass_integral[-1]))


        plt.plot(r_data, mass_density_ly2, '.', label=r'Calculated projected density $\sigma$', ms=2)
    if show_data:
        plotdata(name)

    # Setting up fit for plummer approximation
    if fit_to_data:
        r_array = np.linspace(0.05, 2*r_data[-1], 10000)

        if mfit:
            def plummer(r, b, M):
                return M / (np.pi * np.abs(b) ** 2) * (1 + (r / np.abs(b)) ** (3 / 2)) ** (-7 / 3)
            b_fit, M_fit = scipy.optimize.curve_fit(plummer, r_data, mass_density_ly2)[0]
            print('b_fit :', np.abs(b_fit))
            print('M_fit :', M_fit)
            plt.plot(r_array, plummer(r_array, b_fit, M_fit), '--', label=r'$\sigma_{\mathrm{fit}}(r)$')
        else:
            def plummer(r, b):
                return M / (np.pi * b ** 2) * (1 + (r / np.abs(b)) ** (3 / 2)) ** (-7/3)

            b_fit= scipy.optimize.curve_fit(plummer, r_data, mass_density_ly2)[0][0]
            print('b_fit :', np.abs(b_fit))

            plt.plot(r_array, plummer(r_array, b_fit), '--', label=r'$\sigma_{\mathrm{fit}}(r)$')

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'$r$ [ly]')
    plt.ylabel(r'$\sigma \;\; \left[\frac{M_0}{\text{ly}^2}\right]$')
    if show_data:
        plt.ylim(min(mass_density_ly2) / 4)
        None
    plt.legend()
    plt.savefig('Sigma_plot_' + name + '.pdf')
    plt.show()


    #Acc plots from data
    aN = np.array([G * mass_integral[i-1] / (((r_data[i-1]+r_data[i])/2) ** 2) for i in range(1, len(mass_integral))])
    plt.plot((np.array(r_data[1:]) + np.array(r_data[:-1]))/2, aN,'.', label=r'$a_N$ (Newtonian acceleration)', ms=2)
    plt.plot((np.array(r_data[1:]) + np.array(r_data[:-1]))/2, inpolinv(aN/a0, regime)*aN,'.', label=r'$a_{M}$ (MONDian acceleration)', ms=2)
    plt.plot((np.array(r_data[1:]) + np.array(r_data[:-1]))/2, np.sqrt(aN*a0), '.', label=r'$a_{DM}$ (Deep MOND acceleration)', ms=2)
    plt.plot([0, r_data[-1]], [a0, a0], '--', label=r'$a_0$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$r$ [ly]')
    plt.ylabel(r'$a$ $\left[\frac{\text{ly}}{\text{Myr}^2}\right]$')
    plt.legend()
    plt.show()

    #Velocity plots from data
    vN = np.array([np.sqrt(G*mass_integral[i-1]/((r_data[i-1]+r_data[i])/2)) for i in range(1, len(mass_integral))])
    vM = np.sqrt(inpolinv(aN/a0, regime)*aN * (np.array(r_data[1:]) + np.array(r_data[:-1]))/2)
    v0 = np.sqrt(np.sqrt(aN*a0) * (np.array(r_data[1:]) + np.array(r_data[:-1]))/2)
    plt.plot((np.array(r_data[1:]) + np.array(r_data[:-1]))/2, vN, label='Newtonian velocity', ms=2)
    plt.plot((np.array(r_data[1:]) + np.array(r_data[:-1]))/2, vM, label='MONDian velocity', ms=2)
    plt.plot((np.array(r_data[1:]) + np.array(r_data[:-1]))/2, v0, label='Deep mond velocity', ms=2)
    plt.xlabel('r (ly)')
    plt.ylabel(r'Velocity $(\frac{\text{ly}}{\text{Myr}})$')
    plt.legend()
    plt.show()

    #Energy check!
    if simulate_sphere:
        t_arr = np.linspace(0, T, timesteps)
        plt.plot(t_arr, EkinMat, label=r'$E_{\text{kin}}$')
        plt.plot(t_arr, EgravMat, label=r'$E_{\text{grav}}$')
        plt.plot(t_arr, EpotMat, label=r'$E_{\text{pot}}$')
        plt.plot(t_arr, EMat, label=r'$E_{\text{tot}}$')
        #plt.plot(t_arr, np.ones(len(t_arr))*np.sqrt(G*a0*M**3), '--',label=r'$E_{\text{virial}}$')
        plt.xlabel(r'$t$ [Myr]')
        plt.ylabel(r'$E \; [\frac{M_{\odot} \text{ly}^2}{\text{Myr}^2}]$ ')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        plt.legend()
        plt.show()


        plt.plot(posmat[0, :, 0] * size_of_box / (halfpixels * 2), posmat[0, :, 1] * size_of_box / (halfpixels * 2))
        plt.xlabel("$x$ (ly)")
        plt.ylabel("$y$ (ly)")
        plt.xlim(0, size_of_box)
        plt.ylim(0, size_of_box)
        plt.show()


    def plot_v2_vs_r(posmat, vecmat, n_bins=30, label_final=None):
        """
        Plot <v²> vs r on a double-log scale for t=0 and the final timestep.

        Parameters
        ----------
        posmat   : array [N, timesteps, 3]  – positions in grid units
        vecmat   : array [N, timesteps, 3]  – velocities in grid units
        n_bins   : int – number of radial bins
        label_final : str or None – override the legend label for the final state
        """
        centre = np.array([halfpixels, halfpixels, halfpixels])  # grid-unit centre

        def binned_v2(pos_gu, vel_gu):
            """
            pos_gu, vel_gu : [N, 3] in grid units at a single snapshot.
            Returns bin-centre radii (ly) and mean v² (ly²/Myr²) per bin.
            """
            r_gu = np.linalg.norm(np.array(pos_gu) - centre, axis=1)  # grid units
            r_ly = r_gu * celllen  # ly
            v2 = np.sum(np.array(vel_gu) ** 2, axis=1) * celllen ** 2  # (ly/Myr)²

            # Bin edges span the full range; require at least 2 particles per bin
            edges = np.geomspace(r_ly[r_ly > 0].min(), r_ly.max(), n_bins + 1)
            r_bins, v2_bins = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                mask = (r_ly >= lo) & (r_ly < hi)
                if mask.sum() < 2:
                    continue
                r_bins.append(np.sqrt(lo * hi))  # geometric bin centre
                v2_bins.append(v2[mask].mean())

            return np.array(r_bins), np.array(v2_bins)

        # --- t = 0 ---
        r0, v2_0 = binned_v2(posmat[:, 0, :], vecmat[:, 0, :])

        # --- final timestep ---
        r_f, v2_f = binned_v2(posmat[:, -1, :], vecmat[:, -1, :])

        # --- plot ---
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot([r0[0], r0[-1]], [2/3 * np.sqrt(G*M*a0), 2/3 * np.sqrt(G*M*a0)], '-', label=r'$\langle v^2 \rangle = \frac{2}{3} \sqrt{GMa_0}$')
        ax.plot(r0, v2_0, 'o-', ms=4, label=r'$t = 0$')
        final_label = label_final or (
            r'$t = $' + f'{orbits}' + r' $T_{\rm orbit}$' if UseOrbit
            else r'$t = $' + f'{T}' + r' Myr'
        )
        ax.plot(r_f, v2_f, 's--', ms=4, label=final_label)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$r$ [ly]')
        ax.set_ylabel(r'$\langle v^2 \rangle$ $\left[(\mathrm{ly}/\mathrm{Myr})^2\right]$')
        ax.legend()
        fig.tight_layout()
        plt.savefig('v2_vs_r.pdf')
        plt.show()


    plot_v2_vs_r(posmat, vecmat, n_bins=int(number_of_bins/2))

    # Save particle positions and masses for use in plot_potential_t0.py
    np.save('posmat.npy', np.array(posmat))  # shape [N, timesteps, 3], grid units
    np.save('masses.npy', np.array(particlelist.list[:, 0]))  # shape [N], M_sun
    print("Saved posmat.npy and masses.npy")