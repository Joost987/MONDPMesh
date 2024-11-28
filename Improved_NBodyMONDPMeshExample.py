#%%
import time
import numpy as np
import cupy as cp
import cupyx as cpx
import itertools as itert
import math
import scipy
import pyfftw
import matplotlib.pyplot as plt
from numba import cuda

def FindBall(N):
    ball = cp.array([])
    for i in range(-N+1,N):
        for j in range(-N+1,N):
            for k in range(-N+1,N):
                if i**2+j**2+k**2<N**2:
                    
                    ball = cp.append(ball,[i,j,k])
    ball = cp.reshape(ball,(251,3)).astype("int32")
    return ball
ball4 = FindBall(4)

def AnalyticGravitationalPotential(r,r0,m,EFE):
        return -cpx.scipy.special.erf(cp.linalg.norm(r-r0)/cp.sqrt(2))/cp.linalg.norm(r-r0)*m*G 



# Configure CUDA kernels
mystream = cuda.stream()

N_body = 2 # How many bodies are simulated. Needed before simulation to configure

@cuda.jit
def AssignMassGaussShape(density,particlelist,cellvolumeinv,a,shape):
    # In one block, the mass of one particle is assigned to the grid. The number of grid points that we change per particle is len(shape). 
    # Therefore we have tx < len(shape), so each tx stands for the assignment of some mass of the particle to one gridpoint. And we have bx < len(particlelist) because we need to use as many blocks as there are particles.
    # So each bx stands for assigning the mass of one particle onto the grid.
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    if tx < len(shape) and bx < len(particlelist):
        x = particlelist[bx][1]
        y = particlelist[bx][2]
        z = particlelist[bx][3]
        weight = (1/math.sqrt(2*math.pi))**3*math.exp(-(((int(x)+shape[tx][0])-x)**2+((int(y)+shape[tx][1])-y)**2+((int(z)+shape[tx][2])-z)**2)/(2*a**2))
        cuda.atomic.add(density,(int(x)+shape[tx][0],int(y)+shape[tx][1],int(z)+shape[tx][2]),particlelist[bx][0]*weight*a**(-3)*cellvolumeinv)


shape = ball4
tpb_AMGS = len(shape) # Thread Per Block AssignMassGaussShape
bpg_AMGS = N_body # Block Per Grid AssignMassGaussShape
configure_AssignMassGaussShape = AssignMassGaussShape[bpg_AMGS,tpb_AMGS,mystream]


@cuda.jit
def AssignAccsGaussShape(accparts,accmat,particlelist2,a,shape): # The same as assigning mass to the grid, but now we do not assign a mass but we assign the acceleration to the grid.
                                                                 # Note that since this is now three dimensional, we introduce ty < 3
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    if tx < len(shape) and ty < 3 and bx < len(particlelist2):
        x = particlelist2[bx][1]
        y = particlelist2[bx][2]
        z = particlelist2[bx][3]
        weight = (1/math.sqrt(2*math.pi))**3*math.exp(-(((int(x)+shape[tx][0])-x)**2+((int(y)+shape[tx][1])-y)**2+((int(z)+shape[tx][2])-z)**2)/(2*a**2))
        cuda.atomic.add(accparts,(bx,ty),accmat[ty][int(x)+shape[tx][0]][int(y)+shape[tx][1]][int(z)+shape[tx][2]]*weight*a**(-3))

tpb_AAGS = (len(shape),3) # Thread Per Block AssignAccsGaussShape
bpg_AAGS = N_body # Block Per Grid AssignAccsGaussShape
configure_AssignAccsGaussShape = AssignAccsGaussShape[bpg_AAGS,tpb_AAGS,mystream]


#%%
# =============================================================================
# 
# Making classes
# 
# =============================================================================

#First the Particlelist class is made. This is essentially a list of the masses, positions and velocities of the particles.
#together with different functions acting on this physical system. These are functions to find the kinetic energy, angular momentum
# and accelerations on the particles in the system. A function to simulate this system is also included. 
class Particlelist:
    def __init__(self,particlelist):
        #particlelist object should for each particle contain a list with its mass, then the 3 components of its position
        #then the 3 components of its velocity, so [m,rx,ry,rz,vx,vy,vz].
        self.list = cp.array(particlelist,dtype=np.float32)
        self.EPot = 0
        self.EGrav = 0

    def Ekin(self): 
        return cp.dot(1/2*self.list[:,0],cp.diagonal(self.list[:,4:7]@cp.transpose(self.list[:,4:7])))*celllen**2 #take the dot product between 1/2*masses and the velocities squared. 
        #the self.list[:,4:7]@np.transpose(self.list[:,4:7]) part creates a matrix with all of the velocities of each particle multiplied by each other
        #we only want the velocities of each particle squared, so we take the diagonal of this. 

    def ETot(self):
        return self.Ekin()+self.EPot+self.EGrav
        #Note that EPot can only be calculated by UpdateAccsMOND, as the potential is needed.
                                
    def AngMom(self):
        return cp.sum(cp.diag(self.list[:,0])@cp.cross(self.list[:,1:4]-cp.array([halfpixels]*3),self.list[:,4:7])) #angular momentum with [halfpixels,halfpixels,halfpixels] as origin.
    
    def CenterOfMass(self):
        return 1/cp.sum(self.list[:,0])*(self.list[:,0]@self.list[:,1:4]),1/cp.sum(self.list[:,0])*(self.list[:,0]@self.list[:,4:7])
    
    def UpdateAccsMOND(self,EFE,shape = ball4,sigma = 1,iterlen = 4,regime = 0):
        density = cp.zeros((2*halfpixels,2*halfpixels,2*halfpixels),dtype=np.float32)
        configure_AssignMassGaussShape(density,self.list,cellvolumeinv,sigma,shape)
        densityfft = cp.fft.fftn(density)
        potNDmat = CalcPot(densityfft) 
        del densityfft 
        accNDmat = CalcAccMat(potNDmat) 
        if EFE[0] and (EFE[1] == 1): 
            accNDmat[2,:,:,:] += -Calculate_gN_gal(EFE,x)
        del potNDmat 
        H = cp.zeros([3,2*halfpixels,2*halfpixels,2*halfpixels],dtype=np.float32) 
        for i in range(iterlen): 
            accMONDmat, H = MainLoop(H, accNDmat,regime,EFE)
        del H 
        del accNDmat 
        accMONDmatfft = cp.fft.fftn(accMONDmat,axes=(1,2,3)) 
        del accMONDmat  #the potential in Fourier space is -i*kvec*gvec, where gvec is the acceleration field
        potMONDmatfft = -KdotProd(accMONDmatfft)*K2inv/kstep 
        del accMONDmatfft 
        potMONDmat = cp.imag(cp.fft.ifftn(potMONDmatfft))
        del potMONDmatfft 
        accMONDmat = CalcAccMat(potMONDmat)
        self.EGrav = cp.sum(cp.sqrt(1/(cp.linalg.norm(accMONDmat,axis=0)**2/a0**2)+1)*(cp.linalg.norm(accMONDmat,axis=0)**2/a0**2) 
                            + 0.5*(cp.log(cp.abs(cp.sqrt(1/(cp.linalg.norm(accMONDmat,axis=0)**2/a0**2)+1)-1))
                                    -cp.log(cp.abs(cp.sqrt(1/(cp.linalg.norm(accMONDmat,axis=0)**2/a0**2)+1)+1))))/(8*cp.pi*G)*cellvolume*a0**2
        self.EPot = cp.sum(potMONDmat*density)*cellvolume 
        del density 
        del potMONDmat 
        accparts = cp.zeros((len(self.list),3),dtype=np.float32)
        configure_AssignAccsGaussShape(accparts,accMONDmat,self.list,sigma,shape)
        return accparts
    
    def TimeSim(self,T,dt,iterlength,EFE,free_fall,regime = 0):
        posmat = cp.zeros([len(self.list),T,3],dtype=np.float32)
        vecmat = cp.zeros([len(self.list),T,3],dtype=np.float32)

        MomMat = cp.zeros([T,3],dtype=np.float32)
        AngMat = cp.zeros([T,3],dtype=np.float32)
        EkinMat = cp.zeros([T],dtype=np.float32)
        EgravMat = cp.zeros([T],dtype=np.float32)
        EMat = cp.zeros([T],dtype=np.float32)

        accnew = self.UpdateAccsMOND(EFE,iterlen = 4,regime = regime)
        
        COM = cp.zeros([7,T],dtype=np.float32)

        for t in range(T):
            posmat[:,t,:] = self.list[:,1:4]
            vecmat[:,t,:] = self.list[:,4:7]

            AngMat[t,:] = self.AngMom()
            MomMat[t,:] = cp.transpose(self.list[:,4:7])@self.list[:,0]
            EkinMat[t] = self.Ekin()
            EgravMat[t] = self.EGrav
            EMat[t] = self.ETot()
        
            accold = accnew
            self.list[:,1:4] += self.list[:,4:7]*dt+0.5*accold*cellleninv*dt**2 #Leapfrog without half integer time steps
            if (free_fall == 1) and (EFE[0] == True): 
                if t > 0 :                    
                    self.list[:,3] += 0.5*EFE[2]*cellleninv*dt**2 # + 1/2 g t^2
                    self.list[:,6] += EFE[2]*cellleninv*dt # + g t 
            try: #If the particles are outside of the grid this will raise an error. This catches this error
                #and breaks the loop, ensuring that the data from before the error can be returned. 
                accnew = self.UpdateAccsMOND(EFE,iterlen = iterlength,regime = regime)
            except: #different ways of handling this exception can be made. For the isothermal sphere for example
                #the particles will enter 
                print("particle outside box")
                break
            self.list[:,4:7] += (accold+accnew)*0.5*dt*cellleninv
            if (free_fall == 2 or free_fall == 3) and (EFE[0] == True): 
                if free_fall == 2:
                    COM[1:4,t] = self.CenterOfMass()[0]
                    COM[4:7,t] = self.CenterOfMass()[1] 
                    COM[0,t] = self.EPot # Epot shifted from center
                    
                    self.list[:,1:4] += -(self.CenterOfMass()[0]-cp.array([halfpixels,halfpixels,halfpixels]))
                    self.list[:,4:7] += -self.CenterOfMass()[1]
                
                if free_fall == 3:
                    COM[1:4,t] = self.CenterOfMass()[0]
                    COM[4:7,t] = self.CenterOfMass()[1] 
                    COM[0,t] = self.EPot
                    
                    self.list[:,1:4] += -(self.CenterOfMass()[0]-[halfpixels,halfpixels,halfpixels])
                    self.list[:,4:7] += -self.CenterOfMass()[1]
                    if free_fall == 3:
                        self.UpdateAccsMOND(EFE,iterlen = iterlength,regime = regime)
                        E_temp = self.EPot
                        
                        for i in range(1,t):
                            self.list[:,1:4] += -(dt*COM[4:7,i]+COM[1:4,i-1]-[halfpixels,halfpixels,halfpixels])
                            self.UpdateAccsMOND(EFE,iterlen = iterlength,regime = regime)
                            self.list[:,1:4] += dt*COM[4:7,i]+COM[1:4,i-1]-[halfpixels,halfpixels,halfpixels]
                            E_dt_keer_v_i = self.EPot
                            COM[0,t] += E_dt_keer_v_i - E_temp
                    
                    
            if t%25 == 0: print(t)
        return posmat,vecmat,AngMat,MomMat,EkinMat,EgravMat,EMat,COM
    

#Now different classes of physical systems are made. These are specific systems of which analytical solutions in deep MOND are known.
#These also include functions to calculate the analytical accelerations, and if known the analytical potential energy.
#The simulation function is also altered to include a simulation with the exact accelerations.

class TwoBodyParticlelist(Particlelist): #Arbitary two body system
    def __init__(self,m1,m2,rvec1,rvec2,vvec1,vvec2):
        rvec1 += cp.array([halfpixels]*3) # Position vector of particle 1
        rvec2 += cp.array([halfpixels]*3) # Position vector of particle 2  
        self.list = cp.array([[m1,*rvec1,*vvec1],[m2,*rvec2,*vvec2]])
        self.m1 = m1
        self.m2 = m2

    def Analyticalacc(self):
        particle1 = self.list[0]
        particle2 = self.list[1]
        Force = Body2MOND(particle1[1:4],particle2[1:4],particle1[0],particle2[0])*(particle2[1:4]-particle1[1:4])/cp.linalg.norm((particle1[1:4]-particle2[1:4]))
        return [1/particle1[0]*Force,-1/particle2[0]*Force]

    def EPotAna(self):
        return 2/3*cp.sqrt(G*a0)*((self.m1+self.m2)**(3/2)-self.m1**(3/2)-self.m2**(3/2))*cp.log(cp.linalg.norm(self.list[0,1:4]-self.list[1,1:4]))
    
    def TimeSim(self,T,dt,iterlength,regime = 0):
        posmat = cp.zeros([len(self.list),T,3])
        vecmat = cp.zeros([len(self.list),T,3])

        posmat2 = cp.zeros([len(self.list),T,3])
        vecmat2 = cp.zeros([len(self.list),T,3])

        MomMat = cp.zeros([T,3])
        AngMat = cp.zeros([T,3])
        EMat = cp.zeros([T])

        MomMat2 = cp.zeros([T,3])
        AngMat2 = cp.zeros([T,3])
        EMat2 = cp.zeros([T])        

        accnew = self.UpdateAccsMOND(iterlen = 4,regime = regime)
        for t in range(T):
            
            posmat[:,t,:] = self.list[:,1:4]
            vecmat[:,t,:] = self.list[:,4:7]

            AngMat[t,:] = self.AngMom()
            MomMat[t,:] = cp.transpose(self.list[:,4:7])@self.list[:,0]
            EMat[t] = self.ETot()
        
            accold = accnew
            self.list[:,1:4] += self.list[:,4:7]*dt+0.5*accold*cellleninv*dt**2 #Leapfrog without half integer time steps
            try:
                accnew = self.UpdateAccsMOND(iterlen = iterlength,regime = regime)
            except: 
                
                break
            self.list[:,4:7] += (accold+accnew)*0.5*dt*cellleninv

        self.list[:,1:4] = posmat[:,0,:]
        self.list[:,4:7] = vecmat[:,0,:] 
        
        accnew = cp.array(self.Analyticalacc())
        for t in range(T):
            posmat2[:,t,:] = self.list[:,1:4]
            vecmat2[:,t,:] = self.list[:,4:7]

            AngMat2[t,:] = self.AngMom()
            MomMat2[t,:] = cp.transpose(self.list[:,4:7])@self.list[:,0]
            EMat2[t] = self.Ekin()+self.EPotAna()
        
            accold = accnew
            self.list[:,1:4] += self.list[:,4:7]*dt+0.5*accold*cellleninv*dt**2 #Leapfrog without half integer time steps
            accnew = cp.array(self.Analyticalacc())
            self.list[:,4:7] += (accold+accnew)*0.5*dt*cellleninv
            if t%25 == 0: print(t)
        return posmat,vecmat,AngMat,MomMat,EMat,posmat2,vecmat2,AngMat2,MomMat2,EMat2



class TwoBodyCircParticlelist(TwoBodyParticlelist): #Use this to create a two body system in which stable orbits are produced
    def __init__(self,m1,m2,r,phase):
        M = m1+m2
        v = 1/celllen*cp.sqrt(2/3*cp.sqrt(G*a0*(m1+m2))*(1/(1+cp.sqrt(m1/(m1+m2)))+1/(1+cp.sqrt(m2/(m1+m2)))))
        rvec1 = [m2/M*r*cp.cos(phase),m2/M*r*cp.sin(phase),0]
        rvec2 = [-m1/M*r*cp.cos(phase),-m1/M*r*cp.sin(phase),0]
        rvec1 += cp.array([halfpixels]*3)
        rvec2 += cp.array([halfpixels]*3)   
        vvec1 = [-m2*v/M*cp.sin(phase),m2/M*v*cp.cos(phase),0]
        vvec2 = [m1*v/M*cp.sin(phase),-m1/M*v*cp.cos(phase),0]
        self.list = cp.array([[m1,*rvec1,*vvec1],[m2,*rvec2,*vvec2]])
        self.m1 = m1
        self.m2 = m2

class RingParticlelist(Particlelist): #Ring consisting of N particles, with one central particle. Analytical potential is unknown.
    def __init__(self,m0,r2,N,m):
        M = m0+N*m # Total mass
        v = cp.sqrt(2*cp.sqrt(G*a0)/(3*N*m)*(M**(3/2)-m0**(3/2)-N*m**(3/2)))*cellleninv # Circular velocity in deep MOND regime
        particlecentre = cp.array([m0,halfpixels,halfpixels,halfpixels,0,0,0]) # Create the central particle
        particles = [[m,halfpixels+r2*cp.cos(zeta),halfpixels+r2*cp.sin(zeta),halfpixels,-v*cp.sin(zeta),v*cp.cos(zeta),0] for zeta in cp.random.uniform(0,2*cp.pi,N)] # Create particles in the ring
        particles.append(particlecentre)
        particlelist = cp.array(particles)

        self.list = particlelist
        self.m0 = m0 # Mass of central particle
        self.r = r2 # Radius of ring
        self.N = N # Number of particles in the ring
        self.m = m # Mass of particles in ring

    def RingMONDacc(self):
        M = self.m0+self.N*self.m # Total mass
        rhat = -1*cp.transpose(cp.transpose(self.list[:-1,1:4]-self.list[-1,1:4])/cp.linalg.norm(self.list[:-1,1:4]-self.list[-1,1:4],axis = 1)) # Unit direction vector of acceleration (pointing towards origin)
        return 2/3*cp.sqrt(G*a0)/(self.r*celllen)*(M**(3/2)-self.m0**(3/2)-self.N*self.m**(3/2))/(self.m*self.N)*rhat
    
class IsoThermalParticlelist(Particlelist): #Isothermal sphere of N particles in hydrostatic equillibrium. N should be sufficiently high to approximate the thermodynamic limit.
    def __init__(self,m,b,N):
        M = N*m
        self.m = M # Total mass
        self.b = b # Effective radius
        self.N = N # Number of particles
        self.v2 = cp.sqrt(G*a0*self.m)/3*cellleninv**2*2 # Variance in velocity
        [eta1,eta2,eta3] = [cp.random.uniform(low = 0,high = 1,size = N) for i in [0,1,2]]
        [zeta1,zeta2,zeta3] = [cp.random.uniform(low = 0,high = 2*cp.pi,size = N) for i in [0,1,2]]
        xi = cp.random.uniform(low = -1,high = 1,size = N)

        # rvec: position of particle
        # vvec: velocity of particle
        rvec = cp.transpose(cp.array([[halfpixels]*3]*N))+b*(1/cp.sqrt(eta1)-1)**(-2/3)*cp.array([(cp.sqrt(1-xi**2))*cp.cos(zeta1),(cp.sqrt(1-xi**2))*cp.sin(zeta1),xi])
        vvec = (cp.array([cp.sqrt(-1/1.5*self.v2*cp.log(eta2))*cp.cos(zeta2),cp.sqrt(-1/1.5*self.v2*cp.log(eta2))*cp.sin(zeta2),cp.sqrt(-1/1.5*self.v2*cp.log(eta3))*cp.cos(zeta3)]))
        particlelist = [[m,rvec[0,i],rvec[1,i],rvec[2,i],vvec[0,i],vvec[1,i],vvec[2,i]] for i in range(cp.shape(rvec)[1])]

        particlelist2 = []
        for part in particlelist: # All particles inside the grid are selected
            if cp.abs(part[1])<2*halfpixels-4 and cp.abs(part[2])<2*halfpixels-4 and cp.abs(part[3])<2*halfpixels-4: #we need to subtract 4 to account for the smoothing
                particlelist2.append(part)

        particlelist = cp.array(particlelist2)
        self.list = particlelist

    def Analyticalacc(self): # Returns the analytical acceleration
        rvec = self.list[:,1:4] 
        rvec = rvec-cp.array([halfpixels,halfpixels,halfpixels]) # rvec: position of particle

        r = cp.linalg.norm(rvec,axis = cp.where(cp.array(cp.shape(rvec)) == 3)[0][0]) #The axis expression makes sure it takes the norm at the axis where rvec has 3 components
        return -rvec*cp.transpose(cp.array([cp.sqrt(G*self.m*a0/(self.b**3*r))/(1+(r/self.b)**(3/2))]*3))*cellleninv
    
    def EPotAna(self): # Returns the analytical potential energy
        return 2/3*cp.sqrt(G*self.m*a0)*self.m/self.N*cp.sum(cp.log(1+(cp.linalg.norm(self.list[:,1:4]-cp.array([halfpixels]*3),axis = 1)/self.b)**(3/2)))
    
    def EGravAna(self): # Returns the analytical gravitational energy
        pass
    
    def TimeSim(self,T,dt,iterlength):
        posmat = cp.zeros([len(self.list),T,3]) # posmat = position vector
        vecmat = cp.zeros([len(self.list),T,3]) # vecmat = velocity vector

        posmat_a = cp.zeros([len(self.list),T,3]) # posmat_a = analytical position vector
        vecmat_a = cp.zeros([len(self.list),T,3]) # vecmat_a = analytical velocity vector

        MomMat = cp.zeros([T,3]) # MomMat = momentum vector
        AngMat = cp.zeros([T,3]) # AngMat = angular momentum vector
        EMat = cp.zeros([T]) # EMat = energy

        MomMat_a = cp.zeros([T,3]) # MomMat_a = analytical momentum vector
        AngMat_a = cp.zeros([T,3]) # AngMat_a = analytical angular momentum vector
        EMat_a = cp.zeros([T]) # EMat_a = analytical energy      

        accnew = self.UpdateAccsMOND(iterlen = 4)  
        for t in range(T):
            
            posmat[:,t,:] = self.list[:,1:4]
            vecmat[:,t,:] = self.list[:,4:7]

            AngMat[t,:] = self.AngMom()
            MomMat[t,:] = cp.transpose(self.list[:,4:7])@self.list[:,0]
            EMat[t] = self.ETot()
        
            accold = accnew
            self.list[:,1:4] += self.list[:,4:7]*dt+0.5*accold*cellleninv*dt**2 #Leapfrog without half integer time steps
            try:
                accnew = self.UpdateAccsMOND(iterlen = iterlength)
            except: 
                 self.list[:,1:4] = self.list[:,1:4]%(2*halfpixels-4)
                 accnew = self.UpdateAccsMOND(iterlen = iterlength)
            self.list[:,4:7] += (accold+accnew)*0.5*dt*cellleninv

        self.list[:,1:4] = posmat[:,0,:]
        self.list[:,4:7] = vecmat[:,0,:] 
        
        accnew = cp.array(self.Analyticalacc())
        for t in range(T):
            posmat_a[:,t,:] = self.list[:,1:4]
            vecmat_a[:,t,:] = self.list[:,4:7]

            AngMat_a[t,:] = self.AngMom()
            MomMat_a[t,:] = cp.transpose(self.list[:,4:7])@self.list[:,0]
            EMat_a[t] = self.Ekin()+self.EPotAna()+self.EGravAna()
        
            accold = accnew
            self.list[:,1:4] += self.list[:,4:7]*dt+0.5*accold*cellleninv*dt**2 #Leapfrog without half integer time steps
            accnew = cp.array(self.Analyticalacc())
            self.list[:,4:7] += (accold+accnew)*0.5*dt*cellleninv
            if t%25 == 0: print(t)

        return posmat,vecmat,AngMat,MomMat,EMat,posmat_a,vecmat_a,AngMat_a,MomMat_a,EMat_a

#%% Functions

#Mondian acceleration between two bodies
def Body2MOND(x,y,m1,m2):
    M = m1+m2
    return 2/3*cp.sqrt(G*a0)/(celllen*cp.linalg.norm(x[0:2]-y[0:2],axis=0))*(M**(3/2)-m1**(3/2)-m2**(3/2))

#Calculate potential from Fourier transformed density. 
def CalcPot(densityfft):
    potmatfft = -c*densityfft*K2inv/kstep**2 #*kstep2inv #kstep2inv is 1/kstep**2
    del densityfft 
    potmat = cp.fft.ifftn(potmatfft,s=shape) #inverse Fourier Transform
    del potmatfft 
    potmat = cp.real(potmat)
    return potmat

#Use finite differences to calculate the acceleration field on the grid.
def CalcAccMat(potmat): 
    accmat = cp.array([(cp.roll(potmat,1,axis = 0)-cp.roll(potmat,-1,axis = 0))/(2*celllen),(cp.roll(potmat,1,axis = 1)-cp.roll(potmat,-1,axis = 1))/(2*celllen),(cp.roll(potmat,1,axis = 2)-cp.roll(potmat,-1,axis = 2))/(2*celllen)])
    return accmat

def COMConverter(particles,pos,vec,COM): # COMConverter = Center Of Mass Converter. This function converts the coordinates where the COM was kept constant, to the coordinates where the starting position is constant
    posmat = cp.zeros([len(particles.list),len(COM[0]),3])
    vecmat = cp.zeros([len(particles.list),len(COM[0]),3])
    AngMat = cp.zeros(len(COM[0]))
    MomMat = cp.zeros(len(COM[0]))
    EkinMat = cp.zeros(len(COM[0]))
    EMat = cp.zeros(len(COM[0]))
    for t in range(len(COM[0])):
        vecmat[:,t,:] = vec[:,t,:] + cp.sum(COM[4:7,0:t],axis=1)
        posmat[:,t,:] = pos[:,t,:] + cp.sum(COM[1:4,0:t],axis=1) - [(t)*halfpixels,(t)*halfpixels,(t)*halfpixels] 
        for i in range(t):
            posmat[:,t,:] += dt*cp.sum(COM[4:7,0:(t-i)],axis=1)
        EkinMat[t] = 0.5*cp.dot(particles.list[:,0],cp.diagonal(vecmat[:,t,:]@cp.transpose(vecmat[:,t,:])))*celllen**2
        for i in range(1,t):
            EMat[t] += (COM[0,i] - COM[0,i-1])
            print((COM[0,i] - COM[0,i-1]))
        EMat[t] += COM[0,t] 
    return posmat,vecmat,AngMat,MomMat,EkinMat,EMat

def KdotProd(A): #Dot product of a vector field with k vector. K vector is an element of the Fourier transformed domain. 
    return (inprodx*A[0]+inprody*A[1]+inprodz*A[2])

def inpol(x,func): #Interpolation function \mu
    if func == 0: return x # Deepmond
    if func == 1: return x/cp.sqrt(1+x**2) # Standard
    if func == 2: return # McGaugh
    if func == 3: return 1-cp.exp(-x) # Bose-Einstein
    if func == 5: return 1 # Newton
    

def inpolinv(y,func): #Inverse interpolation function \nu
    if func == 0: return 1/cp.sqrt(y) # Deepmond
    if func == 1: return cp.sqrt(1/2+1/2*cp.sqrt(1+4/y**2)) # Standard
    if func == 2: return 1/(1-cp.exp(-cp.sqrt(y))) # McGaugh
    if func == 3: return # Bose-Einstein
    if func == 5: return 1 # Newton

def CurlFreeProj(Ax,Ay,Az): #Calculates the curl free projection of the vector field A = [Ax,Ay,Az] using FFT's
    A = cp.array([Ax,Ay,Az])
    del Ax,Ay,Az 
    Ahat = cp.fft.fftn(A,s=shape)
    del A 
    intermediatestep = K2inv*KdotProd(Ahat)
    del Ahat 
    xyz = cp.fft.ifftn(cp.array([intermediatestep*inprodx,intermediatestep*inprody,intermediatestep*inprodz]),s=shape)
    del intermediatestep 
    return xyz

def DivFreeProj(Ax,Ay,Az): #Calculates the divergence free projection of the vector field A = [Ax,Ay,Az] using FFT's
    A = cp.array([Ax,Ay,Az])
    del Ax,Ay,Az 
    Ahat = cp.fft.fftn(A,s=shape,axes=(1,2,3))
    del A 
    intermediatestep = K2inv*KdotProd(Ahat)
    xyz = cp.fft.ifftn(cp.array([Ahat[0]-intermediatestep*inprodx,Ahat[1]-intermediatestep*inprody,Ahat[2]-intermediatestep*inprodz]),s=shape,axes=(1,2,3))
    del intermediatestep 
    del Ahat 
    return xyz

def Calculate_gN_gal(EFE_M,x,func): # This calculates the field strength which needs to be added to the Newton acceleration field if an external field is simulated using method 3
    # x = gM/a0
    # y = gN/a0
    # mu(x) = inpol(x,func) = y/x
    # nu(y) = inpolinv(y,func) = x/y
    
    error = abs(EFE_M[2]/a0-inpol(x,func)*x) 
    if error < 0.001: return inpol(x,func)*x*a0
    else:
        i = 1
        while (EFE_M[2]/a0-inpol(x*(1+1/i),func)*x*(1+1/i)) < 0 :
            i += 1        
        return Calculate_gN_gal(EFE,x*(1+1/i),func)

def MainLoop(H,NDacc,func,EFE): #This is the iteration loop. This calculates the MOND acceleration field from the Newtonian acceleration field. See thesis for information on why it works.
    #func refers to which interpolation function should be used. 
    F = NDacc+H
    del H 
    gM = inpolinv(cp.linalg.norm(F,axis = 0)/a0,func)*F #might divide by zero
    del F 
    gM2 = CurlFreeProj(gM[0], gM[1], gM[2])
    del gM 
    if EFE[0] and EFE[1] == 1:
        gM2[2,:,:,:] += EFE[2]
    F = inpol(cp.linalg.norm(gM2,axis = 0)/a0,func)*gM2 
    H = F-NDacc
    del F 
    del NDacc 
    H = DivFreeProj(H[0], H[1], H[2])
    return gM2, H
#%% Simulation parameters and physical constants

#Some of the simulation parameters. You can change halfpixels, which is half the amount of pixels in one dimension of the grid
#You can also change celllen, which is the distance between neighbouring pixels. Some other constants are defined, as this ensures that these calculations are only done once.
halfpixels = 64 #For optimal FFT's, this has to be a power of 2. 
shape = (2*halfpixels,2*halfpixels,2*halfpixels)
size_of_box = 4*10**15 # m
size_of_box = 26738 # au
celllen = size_of_box/(2*halfpixels)
cellleninv = 1/celllen
size = halfpixels*celllen
kstep = cp.pi/(halfpixels*celllen) 
kstep2inv = 1/kstep**2

cellvolume = celllen**3
cellvolumeinv = 1/cellvolume
oversqrt2pi = 1/cp.sqrt(2*cp.pi)
oversqrt2pi3 = oversqrt2pi**3

ball4 = FindBall(4)

T = 4000 #total number of timesteps
EndTime = 3600*12*400 # 200 days 
EndTime = 200 # kyr (Alpha Centauri AB has oribtal period of about 80 kyr)
dt = EndTime/T
tarr = cp.linspace(0,(T-1)*dt,T)

#Some physical constants
G = 6.674*10**(-11) # m^3/(s^2*kg)
G = 3.942*10**7 # au^3/(kyr^2*M☉)
c = 4*cp.pi*G
a0 = 1.2*10**(-10) # m/s^2
a0 = 0.7978 # au/kyr^2

# a0 = 1.2*10^-10 m/s^2 = 0.7978 au/kyr^2 = 12.614 Mly/Gyr^2
# Acceleration from Milky Way on Sun = 3.47*10^-9 m/s^2 = 23.07 au/kyr^2 = 364.8 Mly/Gyr^2
# Newtons gravitational constant = 6.674*10^-11 m^3/(s^2*kg) = 3.942*10^7 au^3/(kyr^2*M☉) = 0.235 Mly^3/(Gyr^2*M_MW)

#%% Creating matrices related to the k vector

Kx=cp.arange(-halfpixels,halfpixels,dtype=np.float32)[:,None,None]**2
Ky=cp.arange(-halfpixels,halfpixels,dtype=np.float32)[:,None]**2
Kz=cp.arange(-halfpixels,halfpixels,dtype=np.float32)**2

#KLM is a matrix where each entry is sum of the index's squared, or the sum of the function values of Kvect of the indices. 
K2=Kx+Ky+Kz
del Kx,Ky,Kz 
K2=cp.roll(K2,halfpixels,axis=0)
K2=cp.roll(K2,halfpixels,axis=1)
K2=cp.roll(K2,halfpixels,axis=2)
K2[0,0,0]=1
K2inv=1/K2
del K2


#The inproduct matrices are matrices where each entry is the x,y,z index, depending on if it is the x,y,z inproduct matrix. 
inprodx = cp.zeros([2*halfpixels,2*halfpixels,2*halfpixels],dtype=np.float32)
inprody = cp.zeros([2*halfpixels,2*halfpixels,2*halfpixels],dtype=np.float32)
inprodz = cp.zeros([2*halfpixels,2*halfpixels,2*halfpixels],dtype=np.float32)
for i in cp.roll(cp.arange(-halfpixels,halfpixels),halfpixels):
    inprodx[i,:,:] = int(i)
    inprody[:,i,:] = int(i)
    inprodz[:,:,i] = int(i)

#%% Simulating and plotting: Two bodies
t_simulation_start = time.time()
simulate_two_bodies = False
if simulate_two_bodies: 
    m1,rx1,ry1,rz1,vx1,vy1,vz1 = 1,halfpixels*6/8,halfpixels,halfpixels,halfpixels/300,halfpixels*0.5/300,0
    m2,rx2,ry2,rz2,vx2,vy2,vz2 = 2,halfpixels*9/8,halfpixels,halfpixels,-halfpixels*0.5/300,-halfpixels*0.25/300,0
    
    particlelist = Particlelist([[m1,rx1,ry1,rz1,vx1,vy1,vz1],[m2,rx2,ry2,rz2,vx2,vy2,vz2]])
    
    EFE_on = False
    EFE_M_strength = 1*a0 # au/kyr^2
    EFE_method = 1 # is by adding homogeneous field to Newtonian acceleration field

    EFE_M = [EFE_on,EFE_method,EFE_M_strength]; T = 10; iterlength = 4; regime = 1 
    free_fall = 0 # 0 is static system, 1 is by shifting the positions each time step by 1/2*g*t^2 (not sure if correct, but may give insights into dynamics as compared to Newton dynamics)
                  # 2 is by keeping the center of mass in the middle (was used in simulations), 3 is static system but each timestep the particles are placed back to the origin (not sure if correct).
    
    posmat_cuda,vecmat_cuda,AngMat_cuda,MomMat_cuda,EkinMat_cuda,EMat_cuda,COM_cuda = particlelist.TimeSim(T,dt,iterlength,EFE_M,free_fall,regime)
    posmat,vecmat,AngMat,MomMat,EkinMat,EMat,COM = posmat_cuda.get(),vecmat_cuda.get(),AngMat_cuda.get(),MomMat_cuda.get(),EkinMat_cuda.get(),EMat_cuda.get(),COM_cuda.get()
    
    if free_fall == 3:
        posmat_cuda,vecmat_cuda,AngMat_cuda,MomMat_cuda,EkinMat_cuda,EMat_cuda = COMConverter(particlelist,posmat_cuda,vecmat_cuda,COM_cuda)
        posmat,vecmat,AngMat,MomMat,EkinMat,EMat = posmat_cuda.get(),vecmat_cuda.get(),AngMat_cuda.get(),MomMat_cuda.get(),EkinMat_cuda.get(),EMat_cuda.get()

    
    #Position plot
    plt.figure(figsize=(7,7))
    for i in range(N_body):
        plt.plot(posmat[i,:,0],posmat[i,:,1])
    plt.xlabel("$x$ (pixels)"); plt.ylabel("$y$ (pixels)")
    if free_fall == 0 or free_fall == 1 or free_fall == 2: 
        plt.xlim(0,2*halfpixels); plt.ylim(0,2*halfpixels)
    else:
        plt.xlim(0,2*halfpixels);
    plt.grid()
    plt.show()
    
    #Velocity plot
    plt.figure(figsize=(7,7))
    plt.plot(vecmat[0,:,1],'k.')
    plt.plot(vecmat[1,:,1],'r.')
    plt.xlabel("$T$ (dt)"); plt.ylabel("$v_y$ (pixels/dt)")
    plt.grid()
    plt.show()
    
    #Energy plot
    plt.figure(figsize=(7,7))
    plt.plot(EMat,label="E total")
    plt.plot(EkinMat,label="E kin")
    plt.plot(EMat-EkinMat,label="E Pot",zorder=1)
    plt.xlabel("timesteps"); plt.ylabel("Energy")
    plt.legend()
    plt.show()
    
