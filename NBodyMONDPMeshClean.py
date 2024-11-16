import timeit
import cupyx
import cupy as cp
import scipy
import numpy as np
import pyfftw

# Settings
PI = np.pi
OVERSQRT2PI = 1 / np.sqrt(2 * PI)
OVERSQRT2PI3 = OVERSQRT2PI ** 3
CELLLEN = 3.086e19 *0.45
PIXELCOUNT = 192
G = 6.67e-11
A0 = 1.2e-10
PIXELCOUNT_X = PIXELCOUNT
PIXELCOUNT_Y = PIXELCOUNT
PIXELCOUNT_Z = PIXELCOUNT //3
K_STEP = PI / (CELLLEN*PIXELCOUNT//2)
K_STEP_INV = 1/K_STEP
M_SUN = 1.9891e30
iterlength = 1
mond = False
debug = False
timing = False


def ball_grid_shape(radius: int = 6):
    # Allocate memory by using an upper bound on the number of points
    max_points = (2 * radius - 1) ** 3
    ball= np.zeros((max_points, 3))
    count = 0

    for i in range(-radius + 1, radius):
        for j in range(-radius + 1, radius):
            for k in range(-radius + 1, radius):
                if i ** 2 + j ** 2 + k ** 2 < radius ** 2:
                    ball[count] = [i, j, k]
                    count += 1

    # Resize the array to the correct number of points added
    ball = ball[:count]

    return ball

ball_shape = ball_grid_shape(radius=6)


def put_particles_on_grid(grid_size: tuple[int, int, int], particlelist: cp.ndarray,
                          shape: cp.ndarray = ball_shape,
                          std: float = 1.0) -> cp.ndarray:
    density = cp.zeros(grid_size)
    particlelist = cp.array(particlelist)
    shape = cp.array(shape)
    shape_coords = shape[cp.newaxis, :, :]
    particle_coords = particlelist[:, 1:4][:, cp.newaxis, :]

    # Broadcast shape and particle coordinates
    cellcoords = cp.rint(particle_coords + shape_coords).astype(int)

    # Ensure the coordinates are within bounds
    cellcoords = cp.clip(cellcoords, 0, cp.array(grid_size) - 1)

    # Calculate the weights
    deltas = cellcoords - particle_coords
    sq_distances = cp.sum(deltas ** 2, axis=-1)
    weights = OVERSQRT2PI3 * cp.exp(-sq_distances / (2 * std ** 2))

    # Flatten arrays for cp.add.at
    flat_coords = cellcoords.reshape(-1, 3).T
    flat_weights = (particlelist[:, 0][:, cp.newaxis] * weights).ravel()

    # Add to density
    cupyx.scatter_add(density, tuple(flat_coords), flat_weights)

    a3inv = std ** (-3)
    return (density * a3inv).get()



def potential_grid_to_acceleration_grid(potential):
    N = potential.shape[0]  # Assuming a cubic grid for simplicity
    # Create k-space grid
    kx = np.fft.fftfreq((PIXELCOUNT_X), d=CELLLEN) * 2 * np.pi
    ky= np.fft.fftfreq((PIXELCOUNT_Y), d=CELLLEN) * 2 * np.pi
    kz= np.fft.fftfreq((PIXELCOUNT_Z), d=CELLLEN) * 2 * np.pi
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')

    # Compute the Fourier transform of the potential
    potential_fft = execfftw(fft_object, inputarr, outputarr, potential)

    # Calculate the derivatives in the Fourier domain
    # The derivative in Fourier space is given by multiplying by i*k in each dimension
    gradient_x_fft = potential_fft * 1j * kx
    gradient_y_fft = potential_fft * 1j * ky
    gradient_z_fft = potential_fft * 1j * kz


    force_x = np.real(execfftw(ifft_object, outputarr, inversearr, gradient_x_fft))
    force_y = np.real(execfftw(ifft_object, outputarr, inversearr, gradient_y_fft))
    force_z = np.real(execfftw(ifft_object, outputarr, inversearr, gradient_z_fft))

    # Note the negative sign: force is the negative gradient of the potential
    acceleration_grid = -np.array([force_x, force_y, force_z])

    return acceleration_grid


def potential_grid_to_acceleration_grid2(potential):
    potential = cp.array(potential)
    coefs = cp.array([1 / 280, -4 / 105, 1 / 5, -4 / 5, 0, 4 / 5, -1 / 5, 4 / 105, -1 / 280])

    # Compute derivatives along each axis using CuPy
    acc_x = sum([coefs[i] * cp.roll(potential, -4 + i, axis=0) for i in range(9)]) / (CELLLEN)
    acc_y = sum([coefs[i] * cp.roll(potential, -4 + i, axis=1) for i in range(9)]) / (CELLLEN)
    acc_z = sum([coefs[i] * cp.roll(potential, -4 + i, axis=2) for i in range(9)]) / (CELLLEN)

    # Stack the acceleration components to form a grid
    acceleration_grid = cp.stack([acc_x, acc_y, acc_z])

    return acceleration_grid.get()

def process_particle_chunk(acceleration_grid, shape, std, particles_chunk):
    num_particles = len(particles_chunk)
    forces_chunk = np.zeros((num_particles, 3), dtype=np.float64)

    for k, particle in enumerate(particles_chunk):
        x_pos, y_pos, z_pos = particle[1:4]
        shape_array = shape + np.array([x_pos, y_pos, z_pos], dtype=np.float64)
        offsets = np.round(shape_array).astype(np.int32)

        dx2 = (offsets[:, 0] - x_pos) ** 2
        dy2 = (offsets[:, 1] - y_pos) ** 2
        dz2 = (offsets[:, 2] - z_pos) ** 2

        weights = OVERSQRT2PI3 * np.exp(-(dx2 + dy2 + dz2) / (2 * std ** 2))
        # weights /= np.sum(weights)  # Normalize weights to sum to 1, if required

        for offset, weight in zip(offsets, weights):
            if all(0 <= offset[j] < acceleration_grid.shape[j+1] for j in range(3)):  # j+1 should be j if acceleration_grid has 3 dimensions
                forces_chunk[k, :] += acceleration_grid[:, offset[0], offset[1], offset[2]] * weight

    return forces_chunk

def grid_to_particles(acceleration_grid, particlelist, std: float,
                      shape = ball_shape):
    accparts = cp.zeros((len(particlelist), 3), dtype=cp.float64)
    shape = cp.array(shape)
    acceleration_grid = cp.array(acceleration_grid)

    particlelist = cp.array(particlelist)

    shape_coords = shape[cp.newaxis, :, :]
    particle_coords = particlelist[:, 1:4][:, cp.newaxis, :]

    # Calculate the cell coordinates
    cellcoords = cp.rint(particle_coords + shape_coords).astype(int)

    # Ensure the coordinates are within bounds
    cellcoords = cp.clip(cellcoords, 0, cp.array(acceleration_grid.shape[1:]) - 1)

    # Calculate the weights
    deltas = cellcoords - particle_coords
    sq_distances = cp.sum(deltas ** 2, axis=-1)
    weights = OVERSQRT2PI3 * cp.exp(-sq_distances / (2 * std ** 2))

    # Extract acceleration values
    acc_x = acceleration_grid[0, cellcoords[..., 0], cellcoords[..., 1], cellcoords[..., 2]]
    acc_y = acceleration_grid[1, cellcoords[..., 0], cellcoords[..., 1], cellcoords[..., 2]]
    acc_z = acceleration_grid[2, cellcoords[..., 0], cellcoords[..., 1], cellcoords[..., 2]]

    # Calculate the contributions
    accparts[:, 0] = cp.sum(acc_x * weights, axis=1)
    accparts[:, 1] = cp.sum(acc_y * weights, axis=1)
    accparts[:, 2] = cp.sum(acc_z * weights, axis=1)

    a3inv = std ** (-3)
    return (accparts * a3inv).get()


def inpol(x, inpol):
    if inpol == 1:
        return x
    if inpol == 2:
        return x/cp.sqrt(1+x**2)


def inpolinv(x, inpol):
    if inpol == 1:
        return cp.nan_to_num(1/cp.sqrt(x))
    if inpol == 2:
        return cp.sqrt(0.5 + 0.5 * cp.sqrt(1 + 4 / (x ** 2)))




def CurlFreeProj(Ax, Ay, Az):  # Calculates the curl free projection of the vector field A=[Ax,Ay,Az] using FFT's
    A = cp.array([Ax, Ay, Az])
    Ahat = cp.array(exec3fftw(fft_object3, inputarr3, outputarr3, A.get()))
    intermediatestep = cp.array(cp.array(KLM_INV2) * KdotProd(Ahat))
    xyz = exec3fftw(ifft_object3, outputarr3, inversearr3,
                    (cp.array([intermediatestep * inprodx, intermediatestep * inprody, intermediatestep * inprodz])).get())

    return cp.array(xyz)



def DivFreeProj(Ax, Ay, Az):  # Calculates the divergence free projection of the vector field A=[Ax,Ay,Az] using FFT's
    A = cp.array([Ax, Ay, Az])
    Ahat = cp.array(exec3fftw(fft_object3, inputarr3, outputarr3, A.get()))
    intermediatestep = cp.array(KLM_INV2) * KdotProd(Ahat)
    xyz = exec3fftw(ifft_object3, outputarr3, inversearr3, (cp.array(
        [Ahat[0] - intermediatestep * inprodx, Ahat[1] - intermediatestep * inprody,
         Ahat[2] - intermediatestep * inprodz])).get())

    return cp.array(xyz)


def KdotProd(
        A):  # Dot product of a vector field with k vector. K vector is an element of the Fourier transformed domain.
    return (inprodx * A[0] + inprody * A[1] + inprodz * A[2])

class detected(Exception):
    pass

def correct_potential(potential_grid, particles):
    cutoff = 10
    Nx, Ny, Nz = potential_grid.shape
    grid_positions = np.indices((Nx, Ny, Nz)).transpose(1, 2, 3, 0)

    for particle in particles:
        mass, px, py, pz, _, _, _ = particle

        # Define influence region based on the cutoff distance
        x_min = max(int(px - cutoff), 0)
        x_max = min(int(px + cutoff + 1), Nx)
        y_min = max(int(py - cutoff), 0)
        y_max = min(int(py + cutoff + 1), Ny)
        z_min = max(int(pz - cutoff), 0)
        z_max = min(int(pz + cutoff + 1), Nz)

        # Extracting local grid positions and calculating distances
        local_grid_positions = grid_positions[x_min:x_max, y_min:y_max, z_min:z_max]
        distance_vectors = local_grid_positions - np.array([px, py, pz])
        distances = np.linalg.norm(distance_vectors, axis=3)

        # Create a mask to apply corrections only within the cutoff and avoid division by zero
        influence_mask = (distances < cutoff)
        valid_distances = distances[influence_mask]

        # Calculating the gravitational potential using the Gaussian smoothed formula

        correction = G * mass * (scipy.special.erf(valid_distances/2)/((valid_distances*CELLLEN)**2))

        # Applying the correction
        correction_array = np.zeros_like(distances)
        correction_array[influence_mask] = correction
        potential_grid[x_min:x_max, y_min:y_max, z_min:z_max] += correction_array


    return potential_grid

def interpolate_potential(potential, particlelist):


    std = 1
    potparts = cp.zeros((len(particlelist)), dtype=cp.float64)
    potential = cp.array(potential)

    particlelist = cp.array(particlelist)

    shape = cp.array(ball_shape)  # Influence range
    shape_coords = shape[cp.newaxis, :, :]
    particle_coords = particlelist[:, 1:4][:, cp.newaxis, :]

    # Compute cell coordinates
    cellcoords = cp.rint(particle_coords + shape_coords).astype(int)
    cellcoords = cp.clip(cellcoords, 0, cp.array(potential.shape) - 1)

    # Distance and weight calculations
    deltas = cellcoords - particle_coords
    sq_distances = cp.sum(deltas ** 2, axis=-1)
    weights = cp.exp(-sq_distances / (2 * std ** 2)) * OVERSQRT2PI3

    # Extract potential values from the grid
    potparts_x = potential[cellcoords[..., 0], cellcoords[..., 1], cellcoords[..., 2]]

    # Exclude self-energy and sum for the total potential energy contributions
    interaction_potential_contributions = cp.sum(potparts_x * weights, axis=1)

    # Sum the contributions to get the total potential energy, excluding self-interactions
    total_potential_energy = cp.sum(interaction_potential_contributions*particlelist[:,0])

    return total_potential_energy




counter = 1
def get_accelerations(std: float, particlelist, iterlen = iterlength, inpolfunc = 2, method=3):
    if debug: print("GRID ASSIGNMENT STARTED")
    if timing: start_time = timeit.default_timer()
    mass_grid = put_particles_on_grid(grid_size=(PIXELCOUNT_X, PIXELCOUNT_Y, PIXELCOUNT_Z), shape=ball_shape,
                                          std=std, particlelist=particlelist)
    density_grid = mass_grid / CELLLEN ** 3
    if timing:
        end_time = timeit.default_timer()
        print("GRID: " + str(end_time-start_time))

    if debug: print("GRID ASSIGNMENT ENDED")


    if debug: print("NEWTONIAN STARTED")
    if timing: start_time = timeit.default_timer()
    density_fft = execfftw(fft_object, inputarr, outputarr, density_grid)
    potential_fft = - 4* PI * G * density_fft * KLM_INV
    potential_fft[PIXELCOUNT_X//2, PIXELCOUNT_Y//2, PIXELCOUNT_Z//2] = 0


    potential = np.real(execfftw(ifft_object, outputarr, inversearr, potential_fft))

    potential -= np.max(potential)

    Epot = np.sum(density_grid*potential)*CELLLEN**3 #- 2*G*M_SUN**2 * np.sqrt(2/(3*np.pi)) / CELLLEN

    acceleration_grid_ND = potential_grid_to_acceleration_grid2(potential)


    Ekin = 1/(8*np.pi*G)*(np.linalg.norm(acceleration_grid_ND))**2 * CELLLEN**3



    if mond:
        if debug: print("MOND STARTED")
        acceleration_grid_ND = cp.array(acceleration_grid_ND)
        H = cp.zeros([3, PIXELCOUNT_X, PIXELCOUNT_Y, PIXELCOUNT_Z])
        for i in range(iterlen):
            F = acceleration_grid_ND + H
            acceleration_grid_MOND = inpolinv(cp.linalg.norm(F, axis=0) / A0, inpolfunc) * F
            acceleration_grid_MOND = CurlFreeProj(acceleration_grid_MOND[0], acceleration_grid_MOND[1], acceleration_grid_MOND[2])
            acceleration_grid_MOND = cp.real(acceleration_grid_MOND)
            F = inpol(cp.linalg.norm(acceleration_grid_MOND, axis=0) / A0, inpolfunc) * acceleration_grid_MOND
            H = F - acceleration_grid_ND
            H = DivFreeProj(H[0], H[1], H[2])
            H = cp.real(H)


        accMONDmatfft = cp.array(exec3fftw(fft_object3, inputarr3, outputarr3, acceleration_grid_MOND.get()))

        potMONDmatfft = -KdotProd(accMONDmatfft) * cp.array(KLM_INV2) / K_STEP



        potMONDmat = cp.array(np.imag(execfftw(ifft_object, outputarr, inversearr, potMONDmatfft.get())))
        acceleration_grid_MOND = potential_grid_to_acceleration_grid2(potMONDmat)
        norm = (np.linalg.norm(acceleration_grid_MOND)**2)/A0**2
        Ekin = A0**2/(8*np.pi*G) * (np.sqrt(norm)*np.sqrt(norm+1)-np.arcsinh(np.sqrt(norm))) * CELLLEN**3
        Epot = np.sum(density_grid * potMONDmat.get()) * CELLLEN ** 3
        print(Ekin)

    if mond:
        accelerations = grid_to_particles(acceleration_grid_MOND, particlelist=particlelist, std=std)
    else:
        accelerations= grid_to_particles(acceleration_grid_ND, particlelist=particlelist, std=std)





    if method == 3:

        particle_positions = np.array([p[1:4] for p in particlelist])
        particle_masses = np.array([p[0] for p in particlelist])

        num_particles = len(particlelist)

        for index1 in range(num_particles):
            distance_to_origin = np.linalg.norm(
                particle_positions[index1] - np.array([PIXELCOUNT_X // 2, PIXELCOUNT_Y // 2, PIXELCOUNT_Z // 2]))
            if distance_to_origin > 7: continue

            distance_vecs = particle_positions - particle_positions[index1]
            distances = np.linalg.norm(distance_vecs, axis=1)

            valid_indices = (distances >0) & (distances < 8)  #& (distance_to_origin > 2) & (distances <= 4) & (distance_to_origin <= 10)

            masses = particle_masses[valid_indices]
            distance_vecs = distance_vecs[valid_indices]
            distances = distances[valid_indices]

            if distances.size > 0:


                distance_factors = correction_factor(distances, std) * distance_vecs.T / distances
                correction = np.sum(distance_factors * G * masses / CELLLEN, axis=1)

                correction2 = np.sum(G * masses / ((distances**2) * distances) * distance_vecs.T / CELLLEN ** 2, axis=1)

                accelerations[index1] -= correction
                accelerations[index1] += correction2

        # Ewald correction
        if debug: print("EWALD STARTED")
        for index1, particle1 in enumerate(particlelist):
            correction = np.zeros(3)
            correction2 = np.zeros(3)
            for index2, particle2 in enumerate(particlelist):
                if index1 == index2: continue
                distance_vec = particle1[1:4] - particle2[1:4]
                distance = np.linalg.norm(distance_vec)
                if distance > 4: continue


                correction += correction_factor(distance, 1) * distance_vec / distance * G * particle2[0] / CELLLEN
                correction2 += (-G * particle2[0] / (distance + 0.001) ** 3 * distance_vec / CELLLEN ** 2)

            accelerations[index1] += correction

            accelerations[index1] += correction2
    if debug: print("EWALD ENDED")

    if timing:
        end_time = timeit.default_timer()
        print("Corrections: " + str(end_time-start_time))



    return accelerations, Epot, Ekin




def correction_factor(R, std):
    return (scipy.special.erf(R/(2*std))/R**2 - np.exp(-R**2/(4*std**2))/(np.sqrt(np.pi)*R*std)) / CELLLEN

def time_loop(number_of_steps, dt, std: float, particlelist, method=3):
    pos = [[] for _ in range(number_of_steps)]
    vel = [[] for _ in range(number_of_steps)]
    Ekin = np.zeros(number_of_steps)
    Epot = np.zeros(number_of_steps)
    Eveld = np.zeros(number_of_steps)

    accelerations, Epot[0], Ekin[0] = get_accelerations(std=std, particlelist=particlelist, method=method)
    for t in range(number_of_steps):



        print(f"{t}/{number_of_steps - 1}")

        pos[t] = particlelist[:, 1:4].copy()
        vel[t] = particlelist[:, 4:7].copy()



        # Update positions
        if debug: print("UPDATE POSITIONS STARTED")


        old_accelerations = accelerations.copy()

        particlelist[:, 1:4] += particlelist[:, 4:7] * dt + 0.5 * accelerations[:] * (dt ** 2) / CELLLEN

        if debug: print("UPDATE POSITIONS ENDED")


        accelerations, Epot[t], Ekin[t] = get_accelerations(std=std, particlelist=particlelist, method=method)

        if debug: print("UPDATE ACCELERATIONS STARTED")
        particlelist[:, 4:7] = np.add(particlelist[:, 4:7], 0.5 * (old_accelerations[:] + accelerations[:]) * dt / CELLLEN)
        if debug: print("UPDATE ACCELERATION ENDED")
        Eveld[t] = Ekin[t]


        Ekin[t] = np.sum((1/2*particlelist[:, 0]*(np.linalg.norm(particlelist[:, 4:7], axis=1)*CELLLEN)**2))

        print(Ekin[t]+Epot[t]+Eveld[t])





    return pos, Ekin, Epot, Eveld




def run_simulation(particlelist, N, std, dt, method=1):

    particlelist = particlelist
    end_positions, Ekin, Epot, Eveld = time_loop(number_of_steps=N, dt=dt, std=std, particlelist=particlelist, method=method)
    np.save("positions_disk_galaxy.npy", end_positions)
    np.save("Ekin_disk_galaxy.npy", Ekin)
    np.save("Epot_disk_galaxy.npy", Epot)






def execfftw(fft_object, inputarr, outputarr, arr):
    inputarr[:, :, :] = arr
    fftarr = fft_object()

    return fftarr.copy()


def exec3fftw(fft_object, inputarr, outputarr, arr):
    inputarr[:, :, :, :] = arr
    fftarr = fft_object()

    return fftarr.copy()


if __name__ == "__main__":
    def DiscrK(k, k_step):
        return (k / k_step) ** 2



    DiscrKvect = np.vectorize(DiscrK)

    k_step_x = 2 * np.pi/(CELLLEN*PIXELCOUNT_X)
    k_step_y = 2 * np.pi / (CELLLEN * PIXELCOUNT_Y)
    k_step_z = 2 * np.pi / (CELLLEN * PIXELCOUNT_Z)

    k_indices_x = np.arange(-PIXELCOUNT_X // 2, PIXELCOUNT_X // 2)
    k_indices_y = np.arange(-PIXELCOUNT_Y // 2, PIXELCOUNT_Y // 2)
    k_indices_z = np.arange(-PIXELCOUNT_Z//2, PIXELCOUNT_Z//2)


    # Apply the DiscrK function with k-space step size for each dimension
    K = DiscrKvect(k_indices_x[:, None, None], 1/k_step_x)
    L = DiscrKvect(k_indices_y[None, :, None], 1/k_step_y)
    M = DiscrKvect(k_indices_z[None, None, :], 1/k_step_z)

    # KLM is a matrix where each entry is sum of the index's squared, or the sum of the function values of Kvect of the indices.

    KLM = K + L + M

    KLM[PIXELCOUNT_X // 2, PIXELCOUNT_Y // 2, PIXELCOUNT_Z // 2] = 1
    KLM = np.roll(KLM, PIXELCOUNT_X // 2, axis=0)
    KLM = np.roll(KLM, PIXELCOUNT_Y // 2, axis=1)
    KLM = np.roll(KLM, PIXELCOUNT_Z // 2, axis=2)
    KLM_INV = 1 / KLM


    def DiscrK2(k):
        return k ** 2


    DiscrKvect2 = np.vectorize(DiscrK2)
    K2 = DiscrKvect2(np.arange(-PIXELCOUNT_X // 2, PIXELCOUNT_X // 2)[:, None, None])

    L2 = DiscrKvect2(np.arange(-PIXELCOUNT_Y // 2, PIXELCOUNT_Y // 2)[:, None])

    M2= DiscrKvect2(np.arange(-PIXELCOUNT_Z // 2, PIXELCOUNT_Z // 2))

    # KLM is a matrix where each entry is sum of the index's squared, or the sum of the function values of Kvect of the indices.

    KLM2 = K2 + L2 + M2


    KLM2[PIXELCOUNT_X // 2, PIXELCOUNT_Y // 2, PIXELCOUNT_Z // 2] = 1
    KLM2 = np.roll(KLM2, PIXELCOUNT_X // 2, axis=0)
    KLM2 = np.roll(KLM2, PIXELCOUNT_Y // 2, axis=1)
    KLM2 = np.roll(KLM2, PIXELCOUNT_Z // 2, axis=2)
    KLM_INV2 = 1 / KLM2



    shape_input  = (PIXELCOUNT_X, PIXELCOUNT_Y, PIXELCOUNT_Z)
    shape_output = (PIXELCOUNT_X, PIXELCOUNT_Y, PIXELCOUNT_Z)

    pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"
    inputarr = pyfftw.empty_aligned(shape_input, dtype="complex64")
    outputarr = pyfftw.empty_aligned(shape_output, dtype="complex64")
    inversearr = pyfftw.empty_aligned(shape_output, dtype="complex64")
    fft_object = pyfftw.FFTW(inputarr, outputarr, axes=(0, 1, 2), threads=6, direction="FFTW_FORWARD")
    ifft_object = pyfftw.FFTW(outputarr, inversearr, direction="FFTW_BACKWARD", axes=(0, 1, 2), threads=6)

    # Now the same is done for the vector FFT.
    shape2_input = (3, shape_input[0], shape_input[1], shape_input[2])
    shape2_output = (3, shape_output[0], shape_output[1], shape_output[2])
    inputarr3 = pyfftw.empty_aligned(shape2_input, dtype="complex64")
    outputarr3 = pyfftw.empty_aligned(shape2_output, dtype="complex64")
    inversearr3 = pyfftw.empty_aligned(shape2_output, dtype="complex64")
    fft_object3 = pyfftw.FFTW(inputarr3, outputarr3, axes=(1, 2, 3), threads=6, direction="FFTW_FORWARD")
    ifft_object3 = pyfftw.FFTW(outputarr3, inversearr3, direction="FFTW_BACKWARD", axes=(1, 2, 3), threads=6)

    inprodx = cp.array(
        [[[i for k in range(-PIXELCOUNT_Z // 2, PIXELCOUNT_Z // 2)] for j in range(-PIXELCOUNT_Y // 2, PIXELCOUNT_Y // 2)] for i
         in
         range(-PIXELCOUNT_X // 2, PIXELCOUNT_X // 2)])
    inprodx = cp.roll(inprodx, PIXELCOUNT_X // 2, axis=0)
    inprody = cp.array(
        [[[j for k in range(-PIXELCOUNT_Z // 2, PIXELCOUNT_Z // 2)] for j in range(-PIXELCOUNT_Y // 2, PIXELCOUNT_Y // 2)] for i
         in
         range(-PIXELCOUNT_X // 2, PIXELCOUNT_X // 2)])
    inprody = cp.roll(inprody, PIXELCOUNT_Y // 2, axis=1)
    inprodz = cp.array(
        [[[k for k in range(-PIXELCOUNT_Z // 2, PIXELCOUNT_Z // 2)] for j in range(-PIXELCOUNT_Y // 2, PIXELCOUNT_Y // 2)] for i
         in
         range(-PIXELCOUNT_X // 2, PIXELCOUNT_X // 2)])
    inprodz = cp.roll(inprodz, PIXELCOUNT_Z // 2, axis=2)
