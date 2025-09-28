# %%
import time
import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from ExampleSystems import *
# %% Simulation parameters and physical constants

# Some of the simulation parameters. You can change halfpixels, which is half the amount of pixels in one dimension of the grid
# You can also change celllen, which is the distance between neighbouring pixels. Some other constants are defined, as this ensures that these calculations are only done once.
halfpixels = 32 #For optimal FFT's, this has to be a power of 2.

size_of_box = 4*10**15 # m
size_of_box = 26738 # au
size_of_box = 1 # ly
datatype=np.float32 #Choose precision

timesteps = 300 # total number of timesteps
# T is total simulated time
T = 0.4*3/10 # Myr
dt = T/timesteps

#Some physical constants
G = 6.674*10**(-11) # m^3/(s^2*kg)
G = 3.942*10**7 # au^3/(kyr^2*M☉)
G = 0.156 # ly^3/(Myr^2*M☉)
a0 = 1.2*10**(-10) # m/s^2
a0 = 0.7978 # au/kyr^2
a0 = 12.614 #ly/Myr^2

CheckBoundary=True #This boolean determines if the code bothers checking if a particle exists the grid
                    #If you know this will not happen, you can turn it off to speed up the code
                    #If a particle crosses the boundary of the grid while this boolean is False, the code gives an error
                    #hence if particles might exit the grid, this should be True

EFE_on = False #External field effect on or off
EFE_M_strength = 1 * a0  # au/kyr^2

EFE_M = (EFE_on, EFE_M_strength);
itersteps = 4; #Number of iterations used for the potential. 
regime = 3 #Which interpolation function should be used? Regime 2,3 don't work currently.
#Different values for regime parameter
#0: Deep MOND
#1: Standard
#2: McGaugh (Slower, EGrav does not work)
#3: Bose-Einstein (Slower)
#4: Verlinde
#5: Newton

#2,3 are slower as either the interpolation function, or its inverse, does not have an expression in terms of elementary functions
# These are therefore calculated using a Newton Raphson method, which is of course slower than using an elementary function 
#For regime 2, the gravitational energy calculation has not been implemented, and EGrav=0 is used everywhere.
MAXNR_ITERATIONS=10 #ONLY USED FOR REGIME=2 OR REGIME=3. Maximum number of iterations the Newton-Rhapson solver does. 


c = 4*jnp.pi*G
shape = (2*halfpixels,2*halfpixels,2*halfpixels)

celllen = size_of_box/(2*halfpixels)
cellleninv = 1/celllen
size = halfpixels*celllen
kstep = jnp.pi/(halfpixels*celllen)
kstep2inv = 1/kstep**2

cellvolume = celllen**3
cellvolumeinv = 1/cellvolume
oversqrt2pi = 1/jnp.sqrt(2*jnp.pi)
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
    return jnp.array(ball)


ball4 = FindBall(4)


def AnalyticGravitationalPotential(r, r0, m, EFE):
    return -jax.scipy.special.erf(jnp.linalg.norm(r - r0) / jnp.sqrt(2)) / jnp.linalg.norm(r - r0) * m * G


def AssignMassGaussShape(density,particlelist,cellvolumeinv,a=1,shape=ball4):  #N is order of the method divided by 2, amount of points used is (2N)^3
    #For a=1, N>3, a=1.5 N>4, a=2 N>5, a=3 N>8. 
  
    a3inv=(1/a)**3
    for i in particlelist: 
        x=i[1]
        y=i[2]
        z=i[3]
      
        for j in shape: 
            cellcoords=(int(i[1])+j[0],int(i[2])+j[1],int(i[3])+j[2])
            weight=oversqrt2pi3*jnp.exp(-((cellcoords[0]-x)**2+(cellcoords[1]-y)**2+(cellcoords[2]-z)**2)/(2*a**2))
            density.at[cellcoords].add(i[0]*weight*cellvolumeinv*a3inv)
    return density


@jax.jit
def AssignMassGaussShapeJAX(density,particlelist,cellvolumeinv,a=1,shape=ball4):  #N is order of the method divided by 2, amount of points used is (2N)^3
    #For a=1, N>3, a=1.5 N>4, a=2 N>5, a=3 N>8. 
  
    a3inv=(1/a)**3

    x=particlelist[:,1]
    y=particlelist[:,2]
    z=particlelist[:,3]
    

    cellcoords= (particlelist[:,jnp.newaxis,1].astype(int)+shape[jnp.newaxis,:,0],particlelist[:,jnp.newaxis,2].astype(int)+shape[jnp.newaxis,:,1],particlelist[:,jnp.newaxis,3].astype(int)+shape[jnp.newaxis,:,2])
    weight=oversqrt2pi3*jnp.exp(-((cellcoords[0]-x[:,jnp.newaxis])**2+(cellcoords[1]-y[:,jnp.newaxis])**2+(cellcoords[2]-z[:,jnp.newaxis])**2)/(2*a**2))

    idx_x = cellcoords[0].ravel()
    idx_y = cellcoords[1].ravel()
    idx_z = cellcoords[2].ravel()
    vals=(particlelist[:,jnp.newaxis,0]*weight*cellvolumeinv*a3inv).ravel()
    density=density.at[idx_x,idx_y,idx_z].add(vals)
    return density

shape_ball = ball4

def AssignAccsGaussShape(accparts,accmat,particlelist2,a=1,shape=ball4):
    a3inv=(1/a)**3

    for k,i in enumerate(particlelist2): 
        x=i[1]
        y=i[2]
        z=i[3]
        
        for j in shape:
            cellcoords=(int(x)+j[0],int(y)+j[1],int(z)+j[2]) 
            weight=oversqrt2pi3*jnp.exp(-((cellcoords[0]-x)**2+(cellcoords[1]-y)**2+(cellcoords[2]-z)**2)/(2*a**2))
            accparts[k,:]+=(accmat[:,cellcoords[0],cellcoords[1],cellcoords[2]])*weight*a3inv
    return accparts



@jax.jit
def AssignAccsGaussShapeJAX(accparts, accmat, particlelist2, a=1, shape=ball4): #Partially written by chatgpt
    a3inv=(1/a)**3

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
    weight = oversqrt2pi3 * jnp.exp(-(dx**2 + dy**2 + dz**2)/(2*a**2)) * a3inv

    # Gather accmat values for all particles
    acc_x = accmat[0, cellcoords_x, cellcoords_y, cellcoords_z]  # (num_particles, num_shape_points)
    acc_y = accmat[1, cellcoords_x, cellcoords_y, cellcoords_z]
    acc_z = accmat[2, cellcoords_x, cellcoords_y, cellcoords_z]

    # Multiply by weights and sum over shape points
    accparts=accparts.at[:, 0].add((acc_x * weight).sum(axis=1))
    accparts=accparts.at[:, 1].add((acc_y * weight).sum(axis=1))
    accparts=accparts.at[:, 2].add((acc_z * weight).sum(axis=1))

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
        return jnp.sum(jnp.diag(self.list[:, 0]) @ jnp.cross(self.list[:, 1:4] - jnp.array([halfpixels] * 3), self.list[:,
                                                                                                          4:7]))  # angular momentum with [halfpixels,halfpixels,halfpixels] as origin.

    def CenterOfMass(self):
        return 1 / jnp.sum(self.list[:, 0]) * (self.list[:, 0] @ self.list[:, 1:4]), 1 / jnp.sum(self.list[:, 0]) * (
                    self.list[:, 0] @ self.list[:, 4:7])

    
    def UpdateAccsMOND(self, particlelist,EFE, shape_ball=ball4, sigma=1, iterlen=4, regime=0):
        density = jnp.zeros((2 * halfpixels, 2 * halfpixels, 2 * halfpixels), dtype=datatype)
        density=AssignMassGaussShapeJAX(density, particlelist, cellvolumeinv, sigma, shape_ball)
        densityfft = jnp.fft.fftn(density)
        potNDmat = Calcpot(densityfft)
        del densityfft
        accNDmat = CalcAccMat(potNDmat)
        if EFE[0] and (EFE[1] == 1):
            accNDmat[2, :, :, :] += -Calculate_gN_gal(EFE_M, EFE_M[1] / a0, regime)
        del potNDmat
        H = jnp.zeros([3, 2 * halfpixels, 2 * halfpixels, 2 * halfpixels], dtype=datatype)
        for i in range(iterlen):
            accMONDmat, H = MainLoop(H, accNDmat, regime, EFE)
        accMONDmatfft = jnp.fft.fftn(accMONDmat, axes=(1, 2, 3))
        del accMONDmat  # the potential in Fourier space is -i*kvec*gvec, where gvec is the acceleration field
        potMONDmatfft = -KdotProd(accMONDmatfft) * K2inv / kstep
        del accMONDmatfft
        potMONDmat = jnp.imag(jnp.fft.ifftn(potMONDmatfft))
        del potMONDmatfft
        accMONDmat = CalcAccMat(potMONDmat)
        Egrav, EGravPot = EGrav(accMONDmat,H+accNDmat,regime)
        EPot = jnp.sum(potMONDmat * density) * cellvolume
        # if regime == 2 or regime == 4:
        #     self.EGrav = EGravPot - self.EPot
        del H
        del accNDmat
        
        del density
        del potMONDmat
        accparts = jnp.zeros((len(particlelist), 3), dtype=datatype)
        accparts=AssignAccsGaussShapeJAX(accparts, accMONDmat, particlelist, sigma, shape_ball)
        return accparts,Egrav,EPot
    UpdateAccsMOND=jax.jit(UpdateAccsMOND,static_argnames=["self","EFE","iterlen","regime"])

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

        accnew,Egrav,Epot = self.UpdateAccsMOND(self.list,EFE, iterlen=4, regime=regime)
        #Jax arrays are immutable, and therefore are copied to a new array when altered.
        #By using JIT, these copy operations should be removed
        @jax.jit 
        def UpdateMats(t,posmat,vecmat,accmat,AngMat,MomMat,EkinMat,EgravMat,EMat,particlelist,AngMom,Ekin,Egrav,Epot,acc):

            posmat=posmat.at[:, t, :].set( particlelist[:, 1:4])
            vecmat=vecmat.at[:, t, :].set( particlelist[:, 4:7])
            accmat=accmat.at[:, t, :].set( acc)

            AngMat=AngMat.at[t, :].set( AngMom)
            MomMat=MomMat.at[t, :].set( jnp.transpose(particlelist[:, 4:7]) @ particlelist[:, 0])
            EkinMat=EkinMat.at[t].set( Ekin)
            EgravMat=EgravMat.at[t].set( Egrav)
            EMat=EMat.at[t].set(Egrav+Epot+Ekin)
            return posmat,vecmat,accmat,AngMat,MomMat,EkinMat,EgravMat,EMat

        if allowNewtonCorrections:
            #TODO: explain the 0.99.
            accnew = self.NewtonCorrection(self.list, accnew, regime, 0.99, 1)

        COM = jnp.zeros([7, timesteps], dtype=datatype)

        for t in range(timesteps):


            posmat,vecmat,accmat,AngMat,MomMat,EkinMat,EgravMat,EMat=UpdateMats(t,posmat,vecmat,accmat,AngMat,MomMat,EkinMat,EgravMat,EMat,self.list,self.AngMom(),self.Ekin(),Egrav,Epot,accnew)

            accold = accnew
            self.list=self.list.at[:, 1:4].add( self.list[:,
                                 4:7] * dt + 0.5 * accold * cellleninv * dt ** 2 ) # Leapfrog without half integer time steps
            if CheckBoundary:
                self.list=CheckBoundaries(self.list)

                accnew,Egrav,Epot = self.UpdateAccsMOND(self.list,EFE, iterlen=itersteps, regime=regime)

                if allowNewtonCorrections:
                    # TODO: explain the 0.99.
                    accnew = self.NewtonCorrection(self.list, accnew, regime, 0.99, 1)
            else:
                try:  # If the particles are outside of the grid this will raise an error. This catches this error
                    # and breaks the loop, ensuring that the data from before the error can be returned.

                    accnew,Egrav,Epot = self.UpdateAccsMOND(self.list,EFE, iterlen=itersteps, regime=regime)

                    if allowNewtonCorrections:
                        # TODO: explain the 0.99.
                        accnew = self.NewtonCorrection(self.list, accnew, regime, 0.99, 1)
                except:  # different ways of handling this exception can be made. For the isothermal sphere for example
                    # the particles will enter
                    print("particle outside box")
                    break
        
            self.list=self.list.at[:, 4:7].add( (accold + accnew) * 0.5 * dt * cellleninv)


            if t % 25 == 0: print("Timesteps done:", t)
        return posmat, vecmat,accmat, AngMat, MomMat, EkinMat, EgravMat, EMat, COM


# %% Functions

@jax.jit
def CheckBoundaries(particlelist): #Set particles that exit the boundary to have 0 mass and 0 velocity, so that they don't affect 
    #the simulation anymore
    cond_func=lambda arr: jnp.logical_or(arr<0,arr>2*halfpixels)
    condition=jnp.logical_or(cond_func(particlelist[:,1]),jnp.logical_or(cond_func(particlelist[:,2]), cond_func(particlelist[:,3])))
    
    mask=~condition
    particlelist=particlelist.at[:,0].set(particlelist[:,0]*mask)
    particlelist=particlelist.at[:,4:].set(particlelist[:,4:]*mask[:,None])
    return particlelist
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
    if func == 2: return FindMu(lambda y: inpolinv(y, func), lambda y: der_inpolinv(y,func), x,max_iterations=MAXNR_ITERATIONS)  # McGaugh
    if func == 3: return 1 - jnp.exp(-x)  # Bose-Einstein
    if func == 4: return 4*x/(1+jnp.sqrt(1+4*x))**2 # Verlinde
    if func == 5: return 1  # Newton


def inpolinv(y, func):  # Inverse interpolation function \nu
    if func == 0: return 1 / jnp.sqrt(y)  # Deepmond
    if func == 1: return jnp.sqrt(1 / 2 + 1 / 2 * jnp.sqrt(1 + 4 / y ** 2))  # Standard
    if func == 2: return 1 / (1 - jnp.exp(-jnp.sqrt(y)))  # McGaugh
    if func == 3: return FindNu(lambda x: inpol(x, func), lambda x: der_inpol(x,func), y,max_iterations=MAXNR_ITERATIONS)  # Bose-Einstein
    if func == 4: return 1 + 1/jnp.sqrt(y) # Verlinde
    if func == 5: return 1  # Newton

#IMPORTANT: If you add interpolation functions or inverse interpolation function where the other is not known
#and you want to use the Newton Raphson method, you should first check that their derivative is never 0.
#Else, if the initial value for the Newton Raphson method gives derivative 0, the method will not converge.

def der_inpol(x,func): #Derivative of interpolation function \mu. Only needed when \nu does not have an explicit expression
    if func==3: return jnp.exp(-x)

def der_inpolinv(y,func): #Derivative of inverse interpolation function \nu. Only needed when \mu does not have an explicit expression
    if func==2: return - jnp.exp(-jnp.sqrt(y))/(2*jnp.sqrt(y)*(1-jnp.exp(-jnp.sqrt(y))))



def NewtonRaphson(func,derfunc,init,rtol=1e-3,max_iterations=20):

    def cond(state):
        currval,initfunc,init,counter=state
        return jnp.logical_and(jnp.all(jnp.abs(currval/initfunc)>rtol), counter<max_iterations)
    
    def body(state):
        currval,initfunc,init,counter=state
        derivative=derfunc(init)
        init=init-currval/derivative
        currval=func(init)
        counter+=1
        return currval,initfunc,init,counter

    initfunc=func(init)
    currval=initfunc
    counter=jnp.array(0)

    currval,initfunc,init,counter=jax.lax.while_loop(cond,body,(currval,initfunc,init,counter))

    return init

def FindMu(nu, der_nu, x, tol=1e-3,max_iterations=20):
    return NewtonRaphson(lambda mu: mu * nu(x * mu) - 1, lambda mu: nu(x * mu)+mu * der_nu(x * mu) * x , rtol=tol,max_iterations=max_iterations)



def FindNu(mu, der_mu, y, tol=1e-3,max_iterations=20):
    return NewtonRaphson(lambda nu: nu * mu(y * nu) - 1, lambda nu: mu(y * nu)+nu * der_mu(y * nu) * y ,jnp.sqrt(1 / y), rtol=tol,max_iterations=max_iterations)


def EGrav(accMONDmat,F,func):
    EGrav = 0
    EGravPot = 0 # Sum of potential energy and gravitational energy
    x = jnp.linalg.norm(accMONDmat,axis=0)/a0
    y = jnp.linalg.norm(F,axis=0)/a0
    if func == 0:
        EGrav = jnp.sum(x**3/3) 
    if func == 1:
        EGrav = jnp.sum((x*jnp.sqrt(1+x**2)-jnp.arcsinh(x))/2)
    if func == 2:
        eps=1e-8
        V_prime = lambda y0,V: y0/(1-np.exp(-np.sqrt(y0))) 
        EGravPot = 0#np.sum(scipy.integrate.odeint(V_prime,eps,np.append(np.array(eps),np.sort(y.flatten())),tfirst=True)) 
        #We numerically integrate V_prime over y to find V. Afterwards we integrate V over space to find E_grav+E_pot. 
        #The numerical integration over y is done by first sorting the y array. As we will integrate over real spaces, the order of the y array does not matter
        #Then we solve the ode dV/dy=V_prime. Scipy will solve this ode and give us the value of the integral for all points
        #specified.
        EGravPot = jnp.asarray(EGravPot)                                                                           
    if func == 3:
        EGrav = jnp.sum(x**2/2+(x+1)*jnp.exp(-x))
    if func == 4:
        EGrav = jnp.sum((x+x**2)/2-(1+4*x)**(3/2)/12)
        EGravPot = jnp.sum(y**2/2 + 2*y**(3/2)/3)
    if func == 5:
        EGrav = jnp.sum(x**2/2)
    return a0**2/(4*jnp.pi*G) * EGrav * cellvolume , -a0**2/(4*jnp.pi*G) * EGravPot * cellvolume
        

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
    xyz = jnp.fft.ifftn(jnp.array([Ahat[0] - intermediatestep * inprodx, Ahat[1] - intermediatestep * inprody,
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
    gM = jnp.nan_to_num(inpolinv(jnp.linalg.norm(F, axis=0) / a0, func)) * F  # if F=0 somewhere, inpolinv(F)=nan, so we use nan to num to set this to 0
    del F
    gM2 = CurlFreeProj(gM[0], gM[1], gM[2])
    del gM
    if EFE[0] and EFE[1] == 1:
        gM2[2, :, :, :] += EFE[2]
    F = inpol(jnp.linalg.norm(gM2, axis=0) / a0, func) * gM2
    H = F - NDacc
    del F
    del NDacc
    H = DivFreeProj(H[0], H[1], H[2])
    return gM2, H


# %% Creating matrices related to the k vector

Kx = jnp.arange(-halfpixels, halfpixels, dtype=datatype)[:, None, None] ** 2
Ky = jnp.arange(-halfpixels, halfpixels, dtype=datatype)[:, None] ** 2
Kz = jnp.arange(-halfpixels, halfpixels, dtype=datatype) ** 2

# KLM is a matrix where each entry is sum of the index's squared, or the sum of the function values of Kvect of the indices.
K2 = Kx + Ky + Kz
del Kx, Ky, Kz
K2 = jnp.roll(K2, halfpixels, axis=0)

K2 = jnp.roll(K2, halfpixels, axis=1)
K2 = jnp.roll(K2, halfpixels, axis=2)
K2=K2.at[0, 0, 0].set(1)
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

inprodx,inprody,inprodz=jnp.array(inprodx),jnp.array(inprody),jnp.array(inprodz)

# %% Simulating and plotting: Two bodies
if __name__=="__main__":
    t_simulation_start = time.time()

    if simulate_two_bodies:
        free_fall = 0  # 0 is static system, 1 is by shifting the positions each time step by 1/2*g*t^2 (not sure if correct, but may give insights into dynamics as compared to Newton dynamics)
    # 2 is by keeping the center of mass in the middle (was used in simulations), 3 is static system but each timestep the particles are placed back to the origin (not sure if correct).


        #m1, rx1, ry1, rz1, vx1, vy1, vz1 = 10, halfpixels * 6 / 8, halfpixels, halfpixels, halfpixels/(100*dt), 0, 0
        #m2, rx2, ry2, rz2, vx2, vy2, vz2 = 20, halfpixels * 9 / 8, halfpixels, halfpixels, -halfpixels/(100*dt),0, 0
        m1=10
        m2=20
        particlelist = TwoBodyCircparticlelist(m1,m2,0.5*halfpixels,0)
       # particlelist= Particlelist([[m1,rx1,ry1,rz1,vx1,vy1,vz1]]) #Use this to test 1 particle sim

        posmat, vecmat, accmat,AngMat, MomMat, EkinMat, EgravMat, EMat, COM= particlelist.TimeSim(timesteps, dt, itersteps, EFE_M, free_fall, regime)

        if free_fall == 3:
            posmat, vecmat, AngMat, MomMat, EkinMat, EgravMat, EMat = COMConverter(
                particlelist, posmat, vecmat, COM)

        # Orbit plot
        t_arr = np.linspace(0, T, timesteps)

        plt.figure(figsize=(7, 7))
        for i in range(particlelist.list.shape[0]):
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

        #acceleration plot
        plt.figure(figsize=(7, 7))
        plt.plot(t_arr,accmat[0,:,0],label="x acceleration of particle 0")
        plt.plot(t_arr,accmat[0,:,1],label="y acceleration of particle 0")
        plt.plot(t_arr,accmat[0,:,2],label="z acceleration of particle 0")
        plt.xlabel("Time (Myr)")
        plt.ylabel("acceleration")
        plt.legend()

        plt.show()

