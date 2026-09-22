"""
3D Electromagnetic Particle-In-Cell (EM-PIC) Simulation
Topic: Localized Plasma Expansion in a Background Magnetic Field

Description:
This script implements a fully 3D electromagnetic PIC simulation to study the expansion
of a localized plasma cloud (electrons and ions) into a vacuum under a constant 
background magnetic field. 

Key Features:
- Charge Deposition: 3D Cloud-In-Cell (CIC) with periodic boundaries.
- Initial Field Solver: 3D FFT-based Poisson solver mapped to a staggered Yee grid.
- Particle Pusher: 3D Boris algorithm for robust integration in EM fields.
- Field Interpolation (Gather): Trilinear interpolation tailored for the Yee grid offsets.
- Current Deposition (Scatter): Charge-conserving Villasenor-Buneman (VB) trajectory method.
- EM Field Update (Maxwell): FDTD leapfrog integration using Faraday's and Ampere's laws.
- Performance Optimization: Core computational bottlenecks are accelerated using Numba (@njit).
"""

import numpy as np
from numba import njit
import matplotlib.pyplot as plt


@njit
def deposit_charge_3d_cic(x, y, z, q, Nx, Ny, Nz, dx, dy, dz):
    """3D Cloud-In-Cell (CIC) charge deposition."""
    # Initialize the charge density array with zeros
    rho = np.zeros((Nx, Ny, Nz), dtype=np.float64)
    N_particles = x.shape[0]

    for p in range(N_particles):
        # Continuous position on the grid
        xi = x[p] / dx
        yi = y[p] / dy
        zi = z[p] / dz

        # Find the floor cell indices
        i = int(xi)
        j = int(yi)
        k = int(zi)

        # Fractional distances for interpolation weights
        fx = xi - i
        fy = yi - j
        fz = zi - k

        # Apply periodic boundary conditions for indices
        i0 = i % Nx
        i1 = (i + 1) % Nx
        j0 = j % Ny
        j1 = (j + 1) % Ny
        k0 = k % Nz
        k1 = (k + 1) % Nz

        # Volumetric weighting over the 8 adjacent nodes of the cube
        w000 = (1.0 - fx) * (1.0 - fy) * (1.0 - fz)
        w100 = fx * (1.0 - fy) * (1.0 - fz)
        w010 = (1.0 - fx) * fy * (1.0 - fz)
        w110 = fx * fy * (1.0 - fz)
        w001 = (1.0 - fx) * (1.0 - fy) * fz
        w101 = fx * (1.0 - fy) * fz
        w011 = (1.0 - fx) * fy * fz
        w111 = fx * fy * fz

        # Scatter/deposit charge to the nodes
        rho[i0, j0, k0] += q * w000
        rho[i1, j0, k0] += q * w100
        rho[i0, j1, k0] += q * w010
        rho[i1, j1, k0] += q * w110
        rho[i0, j0, k1] += q * w001
        rho[i1, j0, k1] += q * w101
        rho[i0, j1, k1] += q * w011
        rho[i1, j1, k1] += q * w111

    # Divide the total accumulated charge by the cell volume to get charge density
    rho = rho / (dx * dy * dz)
    return rho


def solve_poisson_3d_yee(rho, dx, dy, dz):
    """Solves Poisson's equation using 3D FFT and maps the E-field to a Yee grid."""
    Nx, Ny, Nz = rho.shape

    # 3D Fast Fourier Transform of the charge density
    rho_k = np.fft.fftn(rho)

    kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
    kz = np.fft.fftfreq(Nz, d=dz) * 2 * np.pi

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k_squared = KX ** 2 + KY ** 2 + KZ ** 2

    # Avoid division by zero for the DC mode (k=0)
    k_squared[0, 0, 0] = 1.0

    # Calculate potential in Fourier space (Phi(k) = rho(k) / k^2)
    phi_k = rho_k / k_squared
    phi_k[0, 0, 0] = 0.0

    # Inverse FFT to obtain potential in real space
    phi = np.real(np.fft.ifftn(phi_k))

    # --- Calculate electric field on the staggered Yee grid ---

    # Shift potential array in 3 directions (periodic boundary conditions)
    phi_ip1 = np.roll(phi, shift=-1, axis=0)
    phi_jp1 = np.roll(phi, shift=-1, axis=1)
    phi_kp1 = np.roll(phi, shift=-1, axis=2)

    # Ex at staggered position (i+0.5, j, k)
    Ex = -(phi_ip1 - phi) / dx

    # Ey at staggered position (i, j+0.5, k)
    Ey = -(phi_jp1 - phi) / dy

    # Ez at staggered position (i, j, k+0.5)
    Ez = -(phi_kp1 - phi) / dz

    return Ex, Ey, Ez


@njit
def boris_pusher_em_3d(v, Ex_p, Ey_p, Ez_p, Bx_p, By_p, Bz_p, q_m, dt):
    """
    Boris Pusher algorithm for 3D Cartesian space.
    q_m: Charge-to-mass ratio (q/m) of the particle.
    """
    N_particles = v.shape[0]

    for p in range(N_particles):
        # 1. First half-step electric field acceleration (v-)
        um_x = v[p, 0] + 0.5 * q_m * Ex_p[p] * dt
        um_y = v[p, 1] + 0.5 * q_m * Ey_p[p] * dt
        um_z = v[p, 2] + 0.5 * q_m * Ez_p[p] * dt

        # 2. Magnetic Rotation in 3D
        # Calculate vector t in 3 directions
        t_x = 0.5 * q_m * Bx_p[p] * dt
        t_y = 0.5 * q_m * By_p[p] * dt
        t_z = 0.5 * q_m * Bz_p[p] * dt
        t_mag2 = t_x**2 + t_y**2 + t_z**2

        # Calculate vector s in 3 directions
        s_scale = 2.0 / (1.0 + t_mag2)
        s_x = t_x * s_scale
        s_y = t_y * s_scale
        s_z = t_z * s_scale

        # First cross product: v' = v- + (v- x t)
        up_x = um_x + (um_y * t_z - um_z * t_y)
        up_y = um_y + (um_z * t_x - um_x * t_z)
        up_z = um_z + (um_x * t_y - um_y * t_x)

        # Second cross product: v+ = v- + (v' x s)
        uplus_x = um_x + (up_y * s_z - up_z * s_y)
        uplus_y = um_y + (up_z * s_x - up_x * s_z)
        uplus_z = um_z + (up_x * s_y - up_y * s_x)

        # 3. Second half-step electric field acceleration
        v[p, 0] = uplus_x + 0.5 * q_m * Ex_p[p] * dt
        v[p, 1] = uplus_y + 0.5 * q_m * Ey_p[p] * dt
        v[p, 2] = uplus_z + 0.5 * q_m * Ez_p[p] * dt


@njit
def trilinear_interpolate(x_p, y_p, z_p, grid, dx, dy, dz, off_x, off_y, off_z, Nx, Ny, Nz):
    """Helper function for trilinear interpolation on a 3D grid with specific offsets."""
    # Logical coordinates with grid offset (accommodating Yee grid staggering)
    cx = (x_p / dx) - off_x
    cy = (y_p / dy) - off_y
    cz = (z_p / dz) - off_z

    # Determine floor indices
    i = int(np.floor(cx))
    j = int(np.floor(cy))
    k = int(np.floor(cz))

    # Intra-cell fractional distances (weights)
    hx = cx - i
    hy = cy - j
    hz = cz - k

    # Periodic boundary conditions for interpolation
    i0, i1 = i % Nx, (i + 1) % Nx
    j0, j1 = j % Ny, (j + 1) % Ny
    k0, k1 = k % Nz, (k + 1) % Nz

    # Trilinear interpolation between the 8 enclosing nodes
    return ((1 - hx) * (1 - hy) * (1 - hz) * grid[i0, j0, k0] +
            hx * (1 - hy) * (1 - hz) * grid[i1, j0, k0] +
            (1 - hx) * hy * (1 - hz) * grid[i0, j1, k0] +
            hx * hy * (1 - hz) * grid[i1, j1, k0] +
            (1 - hx) * (1 - hy) * hz * grid[i0, j0, k1] +
            hx * (1 - hy) * hz * grid[i1, j0, k1] +
            (1 - hx) * hy * hz * grid[i0, j1, k1] +
            hx * hy * hz * grid[i1, j1, k1])


@njit
def gather_fields_3d(x, y, z, Ex, Ey, Ez, Bx, By, Bz, dx, dy, dz, Nx, Ny, Nz):
    """Interpolates macroscopic fields to individual particle positions."""
    N = x.shape[0]
    Ex_p, Ey_p, Ez_p = np.zeros(N), np.zeros(N), np.zeros(N)
    Bx_p, By_p, Bz_p = np.zeros(N), np.zeros(N), np.zeros(N)

    for p in range(N):
        # Standard Yee grid offsets
        # Ex: (0.5, 0, 0), Ey: (0, 0.5, 0), Ez: (0, 0, 0.5)
        # Bx: (0, 0.5, 0.5), By: (0.5, 0, 0.5), Bz: (0.5, 0.5, 0)
        
        Ex_p[p] = trilinear_interpolate(x[p], y[p], z[p], Ex, dx, dy, dz, 0.5, 0.0, 0.0, Nx, Ny, Nz)
        Ey_p[p] = trilinear_interpolate(x[p], y[p], z[p], Ey, dx, dy, dz, 0.0, 0.5, 0.0, Nx, Ny, Nz)
        Ez_p[p] = trilinear_interpolate(x[p], y[p], z[p], Ez, dx, dy, dz, 0.0, 0.0, 0.5, Nx, Ny, Nz)

        Bx_p[p] = trilinear_interpolate(x[p], y[p], z[p], Bx, dx, dy, dz, 0.0, 0.5, 0.5, Nx, Ny, Nz)
        By_p[p] = trilinear_interpolate(x[p], y[p], z[p], By, dx, dy, dz, 0.5, 0.0, 0.5, Nx, Ny, Nz)
        Bz_p[p] = trilinear_interpolate(x[p], y[p], z[p], Bz, dx, dy, dz, 0.5, 0.5, 0.0, Nx, Ny, Nz)

    return Ex_p, Ey_p, Ez_p, Bx_p, By_p, Bz_p


# ---------------------------------------------------------
# Villasenor-Buneman (VB) 3D Current Deposition Functions
# ---------------------------------------------------------

@njit
def deposit_current_segment_periodic_3d(xA, yA, zA, xB, yB, zB, q, dt, dx, dy, dz, Jx, Jy, Jz, Nx, Ny, Nz):
    """Deposits current for a single straight-line segment within one cell."""
    x_mid = 0.5 * (xA + xB)
    y_mid = 0.5 * (yA + yB)
    z_mid = 0.5 * (zA + zB)

    i = int(np.floor(x_mid / dx))
    j = int(np.floor(y_mid / dy))
    k = int(np.floor(z_mid / dz))

    Fx = (x_mid - i * dx) / dx
    Fy = (y_mid - j * dy) / dy
    Fz = (z_mid - k * dz) / dz

    delta_x = xB - xA
    delta_y = yB - yA
    delta_z = zB - zA

    coef = q / (dt * dx * dy * dz)

    # Direct application of periodic boundary conditions for current nodes
    i0 = i % Nx; i1 = (i + 1) % Nx
    j0 = j % Ny; j1 = (j + 1) % Ny
    k0 = k % Nz; k1 = (k + 1) % Nz

    # Deposit Jx (depends on y and z grid fractions)
    cx = coef * delta_x
    Jx[i0, j0, k0] += cx * (1.0 - Fy) * (1.0 - Fz)
    Jx[i0, j1, k0] += cx * Fy * (1.0 - Fz)
    Jx[i0, j0, k1] += cx * (1.0 - Fy) * Fz
    Jx[i0, j1, k1] += cx * Fy * Fz

    # Deposit Jy (depends on x and z grid fractions)
    cy = coef * delta_y
    Jy[i0, j0, k0] += cy * (1.0 - Fx) * (1.0 - Fz)
    Jy[i1, j0, k0] += cy * Fx * (1.0 - Fz)
    Jy[i0, j0, k1] += cy * (1.0 - Fx) * Fz
    Jy[i1, j0, k1] += cy * Fx * Fz

    # Deposit Jz (depends on x and y grid fractions)
    cz = coef * delta_z
    Jz[i0, j0, k0] += cz * (1.0 - Fx) * (1.0 - Fy)
    Jz[i1, j0, k0] += cz * Fx * (1.0 - Fy)
    Jz[i0, j1, k0] += cz * (1.0 - Fx) * Fy
    Jz[i1, j1, k0] += cz * Fx * Fy

@njit
def deposit_current_trajectory_periodic_3d(x_old, y_old, z_old, x_new, y_new, z_new, q, dt, dx, dy, dz, Jx, Jy, Jz, Nx, Ny, Nz):
    """Splits a particle's trajectory across cell boundaries and deposits current for each segment."""
    x_curr = x_old
    y_curr = y_old
    z_curr = z_old

    # Maximum possible cell boundary crossings in a 3D single step is 7
    for _ in range(7): 
        if np.abs(x_curr - x_new) < 1e-12 and np.abs(y_curr - y_new) < 1e-12 and np.abs(z_curr - z_new) < 1e-12:
            break

        i = int(np.floor(x_curr / dx))
        j = int(np.floor(y_curr / dy))
        k = int(np.floor(z_curr / dz))

        delta_x = x_new - x_curr
        delta_y = y_new - y_curr
        delta_z = z_new - z_curr

        # Determine next intersection boundaries
        if delta_x > 1e-12:
            x_bound = (i + 1) * dx
        elif delta_x < -1e-12:
            x_bound = i * dx
            if np.abs(x_curr - x_bound) < 1e-12:
                x_bound = (i - 1) * dx
        else:
            x_bound = np.inf

        if delta_y > 1e-12:
            y_bound = (j + 1) * dy
        elif delta_y < -1e-12:
            y_bound = j * dy
            if np.abs(y_curr - y_bound) < 1e-12:
                y_bound = (j - 1) * dy
        else:
            y_bound = np.inf

        if delta_z > 1e-12:
            z_bound = (k + 1) * dz
        elif delta_z < -1e-12:
            z_bound = k * dz
            if np.abs(z_curr - z_bound) < 1e-12:
                z_bound = (k - 1) * dz
        else:
            z_bound = np.inf

        # Fractional step parameter t to reach boundaries
        tx = (x_bound - x_curr) / delta_x if delta_x != 0 else np.inf
        ty = (y_bound - y_curr) / delta_y if delta_y != 0 else np.inf
        tz = (z_bound - z_curr) / delta_z if delta_z != 0 else np.inf

        t_min = min(tx, ty, tz, 1.0)

        if t_min <= 0:
            t_min = 1e-6

        x_next = x_curr + t_min * delta_x
        y_next = y_curr + t_min * delta_y
        z_next = z_curr + t_min * delta_z

        if t_min == 1.0:
            x_next = x_new
            y_next = y_new
            z_next = z_new

        deposit_current_segment_periodic_3d(x_curr, y_curr, z_curr, x_next, y_next, z_next, q, dt, dx, dy, dz, Jx, Jy, Jz, Nx, Ny, Nz)

        x_curr = x_next
        y_curr = y_next
        z_curr = z_next

@njit
def deposit_current_particles_vb_periodic_3d(x_old, y_old, z_old, x_new, y_new, z_new, q, dt, dx, dy, dz, Jx, Jy, Jz, Nx, Ny, Nz):
    """Wrapper function to iterate current deposition over all particles."""
    for p in range(len(x_old)):
        deposit_current_trajectory_periodic_3d(x_old[p], y_old[p], z_old[p], x_new[p], y_new[p], z_new[p], q, dt, dx, dy, dz, Jx, Jy, Jz, Nx, Ny, Nz)

# -------------------------------------------------
# Magnetic Field Update (Faraday's Law - 3D)
# -------------------------------------------------
def update_b_field_faraday_3d(Bx, By, Bz, Ex, Ey, Ez, dt_step, dx, dy, dz):
    """
    Updates the B-field using the curl of E.
    Negative shift denotes a forward finite difference mapped to the staggered grid.
    """
    Bx -= dt_step * ( (np.roll(Ez, shift=-1, axis=1) - Ez) / dy - (np.roll(Ey, shift=-1, axis=2) - Ey) / dz )
    By -= dt_step * ( (np.roll(Ex, shift=-1, axis=2) - Ex) / dz - (np.roll(Ez, shift=-1, axis=0) - Ez) / dx )
    Bz -= dt_step * ( (np.roll(Ey, shift=-1, axis=0) - Ey) / dx - (np.roll(Ex, shift=-1, axis=1) - Ex) / dy )

    return Bx, By, Bz


def plot_fields_and_particles_3d(x_e, y_e, z_e, x_i, y_i, z_i, Ex, Bz, step, Lx, Ly, Lz):
    """Visualizes 2D mid-plane slices of fields and a 3D scatter plot of particles."""
    plt.clf()

    # Find the middle index in the z-direction to plot a 2D slice
    Nz = Ex.shape[2]
    mid_z = Nz // 2

    # 1. Plot Ex field slice in the x-y plane
    plt.subplot(1, 3, 1)
    plt.imshow(Ex[:, :, mid_z].T, origin='lower', extent=[0, Lx, 0, Ly], cmap='RdBu', aspect='auto')
    plt.colorbar(label='$E_x$')
    plt.title(f'Slice $E_x$ (z={mid_z}) - Step {step}')
    plt.xlabel('$x$')
    plt.ylabel('$y$')

    # 2. Plot Bz field slice in the x-y plane
    plt.subplot(1, 3, 2)
    plt.imshow(Bz[:, :, mid_z].T, origin='lower', extent=[0, Lx, 0, Ly], cmap='PRGn', aspect='auto')
    plt.colorbar(label='$B_z$')
    plt.title(f'Slice $B_z$ (z={mid_z}) - Step {step}')
    plt.xlabel('$x$')
    plt.ylabel('$y$')

    # 3. 3D Particle Scatter Plot (first 2000 particles to save rendering time)
    ax = plt.subplot(1, 3, 3, projection='3d')
    N_plot = min(2000, len(x_e))

    # Plot Electrons
    ax.scatter(x_e[:N_plot], y_e[:N_plot], z_e[:N_plot],
               s=1, color='blue', alpha=0.5, label='Electrons')
    # Plot Ions
    ax.scatter(x_i[:N_plot], y_i[:N_plot], z_i[:N_plot],
               s=1, color='red', alpha=0.5, label='Ions')

    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(0, Lz)
    ax.set_title('3D Particle Positions')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

    plt.tight_layout()
    plt.pause(0.01)


# ==========================================================
# 1. Initial Setup and Simulation Parameters
# ==========================================================

Nx, Ny, Nz = 64, 64, 64        # Number of grid cells in each direction
Lx, Ly, Lz = 1.0, 1.0, 1.0     # Physical length of the simulation box in each direction

dx = Lx / Nx                   # Grid spacing in x
dy = Ly / Ny                   # Grid spacing in y
dz = Lz / Nz                   # Grid spacing in z

dt = 0.005                     # Time step (must satisfy CFL condition)
n0 = 1
Np_per_cell = 10
N_particles = int(Np_per_cell * (Nx) * (Ny) * (Nz))
weight = n0 * (Lx * Ly * Lz) / N_particles

cx, cy, cz = Lx/2, Ly/2, Lz/2  # Center coordinates for the initial plasma cloud
radius = Lx / 8.0              # Radius of the localized plasma cloud

mass_ratio = 100               # Artificial ion-to-electron mass ratio

q_e = -1 * weight
m_e = weight
q_m_e = q_e / m_e

q_i = +1 * weight
m_i = mass_ratio * weight
q_m_i = q_i / m_i

c, eps0 = 1.0, 1.0             # Speed of light and vacuum permittivity

# Initial positions (x, y, z) for electrons inside a localized Gaussian cloud
x_e, y_e, z_e = (np.random.normal(cx, radius, N_particles) % Lx, np.random.normal(cy, radius, N_particles) % Ly,
                 np.random.normal(cz, radius, N_particles) % Lz)

vth_e = 0.05 * c               # Electron thermal velocity
vx_e, vy_e, vz_e = (np.random.normal(0, vth_e, N_particles), np.random.normal(0, vth_e, N_particles),
                    np.random.normal(0, vth_e, N_particles))

v_e = np.stack((vx_e, vy_e, vz_e), axis=-1)

# Ions initially co-located with electrons (quasi-neutral plasma)
x_i = x_e.copy()
y_i = y_e.copy()
z_i = z_e.copy()

vth_i = vth_e / np.sqrt(mass_ratio)

vx_i = np.random.normal(0, vth_i, N_particles)
vy_i = np.random.normal(0, vth_i, N_particles)
vz_i = np.random.normal(0, vth_i, N_particles)

v_i = np.stack((vx_i, vy_i, vz_i), axis=-1)


# Electric Field arrays (Ex, Ey, Ez)
Ex = np.zeros((Nx, Ny, Nz))
Ey = np.zeros((Nx, Ny, Nz))
Ez = np.zeros((Nx, Ny, Nz))

# Magnetic Field arrays (Bx, By, Bz)
Bx = np.zeros((Nx, Ny, Nz))
By = np.zeros((Nx, Ny, Nz))
Bz = np.zeros((Nx, Ny, Nz))

# Current Density arrays (Separate and Total)
Jx_e, Jy_e, Jz_e = np.zeros((Nx, Ny, Nz)), np.zeros((Nx, Ny, Nz)), np.zeros((Nx, Ny, Nz))
Jx_i, Jy_i, Jz_i = np.zeros((Nx, Ny, Nz)), np.zeros((Nx, Ny, Nz)), np.zeros((Nx, Ny, Nz))
Jx, Jy, Jz = np.zeros((Nx, Ny, Nz)), np.zeros((Nx, Ny, Nz)), np.zeros((Nx, Ny, Nz))

# Apply Constant Background Magnetic Field
# B is a 3D array initialized once
Bz += 5  # Add an external constant field (Bz = 5) to the entire grid

N_steps = 10000

# ==========================================================
# 2. Leapfrog Synchronization (Rewind Half Step)
# ==========================================================

# Calculate initial charge density (electrons + ions)
rho_e = deposit_charge_3d_cic(x_e, y_e, z_e, q_e, Nx, Ny, Nz, dx, dy, dz)
rho_i = deposit_charge_3d_cic(x_i, y_i, z_i, q_i, Nx, Ny, Nz, dx, dy, dz)
rho = rho_e + rho_i
rho = rho - np.mean(rho)  # Neutralize small numerical inaccuracies for the FFT Poisson solver

# Calculate initial electrostatic field
Ex, Ey, Ez = solve_poisson_3d_yee(rho, dx, dy, dz)


# Rewind velocities by half a time step (t = -dt/2) for both species
Ex_p_e, Ey_p_e, Ez_p_e, Bx_p_e, By_p_e, Bz_p_e = gather_fields_3d(x_e, y_e, z_e, Ex, Ey, Ez, Bx, By, Bz, dx, dy, dz,
                                                                  Nx, Ny, Nz)

Ex_p_i, Ey_p_i, Ez_p_i, Bx_p_i, By_p_i, Bz_p_i = gather_fields_3d(x_i, y_i, z_i, Ex, Ey, Ez, Bx, By, Bz, dx, dy, dz,
                                                                  Nx, Ny, Nz)

boris_pusher_em_3d(v_e, Ex_p_e, Ey_p_e, Ez_p_e, Bx_p_e, By_p_e, Bz_p_e, q_m_e, -0.5 * dt)
boris_pusher_em_3d(v_i, Ex_p_i, Ey_p_i, Ez_p_i, Bx_p_i, By_p_i, Bz_p_i, q_m_i, -0.5 * dt)

# Rewind the magnetic field to half-step (t = -dt/2) to sync with velocities
Bx, By, Bz = update_b_field_faraday_3d(Bx, By, Bz, Ex, Ey, Ez, -0.5 * dt, dx, dy, dz)

plt.figure(figsize=(15, 4))

# ==========================================================
# 3. Main Simulation Loop
# ==========================================================

for step in range(1, N_steps + 1):

    # ---------------------------------------------------------
    #  Magnetic Field Update: First Half-Step (Faraday)
    #  Advances B from (t - dt/2) to time (t) to synchronize with E
    # ---------------------------------------------------------
    Bx, By, Bz = update_b_field_faraday_3d(Bx, By, Bz, Ex, Ey, Ez, 0.5 * dt, dx, dy, dz)

    # ---------------------------------------------------------
    #  Interpolate Fields to Particle Positions (Gather)
    # ---------------------------------------------------------
    Ex_p_e, Ey_p_e, Ez_p_e, Bx_p_e, By_p_e, Bz_p_e = gather_fields_3d(x_e, y_e, z_e, Ex, Ey, Ez, Bx, By, Bz, dx, dy, dz,
                                                                      Nx, Ny, Nz)
    Ex_p_i, Ey_p_i, Ez_p_i, Bx_p_i, By_p_i, Bz_p_i = gather_fields_3d(x_i, y_i, z_i, Ex, Ey, Ez, Bx, By, Bz, dx, dy, dz,
                                                                      Nx, Ny, Nz)

    # Store old positions for current deposition (trajectory splitting)
    x_old_e = np.copy(x_e)
    y_old_e = np.copy(y_e)
    z_old_e = np.copy(z_e)

    x_old_i = np.copy(x_i)
    y_old_i = np.copy(y_i)
    z_old_i = np.copy(z_i)

    # ---------------------------------------------------------
    #  Particle Pusher (Update Velocities and Positions)
    # ---------------------------------------------------------
    boris_pusher_em_3d(v_e, Ex_p_e, Ey_p_e, Ez_p_e, Bx_p_e, By_p_e, Bz_p_e, q_m_e, dt)
    boris_pusher_em_3d(v_i, Ex_p_i, Ey_p_i, Ez_p_i, Bx_p_i, By_p_i, Bz_p_i, q_m_i, dt)

    x_e += v_e[:, 0] * dt
    y_e += v_e[:, 1] * dt
    z_e += v_e[:, 2] * dt

    x_i += v_i[:, 0] * dt
    y_i += v_i[:, 1] * dt
    z_i += v_i[:, 2] * dt

    # ---------------------------------------------------------
    #  Current Deposition (Scatter - VB Method)
    # ---------------------------------------------------------

    # Reset current arrays
    Jx.fill(0.0); Jy.fill(0.0); Jz.fill(0.0)
    Jx_e.fill(0.0); Jy_e.fill(0.0); Jz_e.fill(0.0)
    Jx_i.fill(0.0); Jy_i.fill(0.0); Jz_i.fill(0.0)

    # Deposit currents for electrons and ions
    deposit_current_particles_vb_periodic_3d(x_old_e, y_old_e, z_old_e, x_e, y_e, z_e, q_e, dt, dx, dy, dz, Jx_e, Jy_e, Jz_e, Nx, Ny, Nz)
    deposit_current_particles_vb_periodic_3d(x_old_i, y_old_i, z_old_i, x_i, y_i, z_i, q_i, dt, dx, dy, dz, Jx_i, Jy_i, Jz_i, Nx, Ny, Nz)

    Jx = Jx_e + Jx_i
    Jy = Jy_e + Jy_i
    Jz = Jz_e + Jz_i

    # Apply periodic boundary conditions to particle coordinates
    x_e = x_e % Lx
    y_e = y_e % Ly
    z_e = z_e % Lz

    x_i = x_i % Lx
    y_i = y_i % Ly
    z_i = z_i % Lz

    # ---------------------------------------------------------
    #  Magnetic Field Update: Second Half-Step (Faraday)
    #  Advances B from time (t) to (t + dt/2)
    # ---------------------------------------------------------
    Bx, By, Bz = update_b_field_faraday_3d(Bx, By, Bz, Ex, Ey, Ez, 0.5 * dt, dx, dy, dz)

    # ---------------------------------------------------------
    #  Electric Field Update (Ampere's Law - 3D)
    # ---------------------------------------------------------
    # Curl of B to update E (using positive shift for backward finite difference)
    Ex += c ** 2 * dt * ((Bz - np.roll(Bz, shift=1, axis=1)) / dy - (By - np.roll(By, shift=1, axis=2)) / dz) - (dt / eps0) * Jx
    Ey += c ** 2 * dt * ((Bx - np.roll(Bx, shift=1, axis=2)) / dz - (Bz - np.roll(Bz, shift=1, axis=0)) / dx) - (dt / eps0) * Jy
    Ez += c ** 2 * dt * ((By - np.roll(By, shift=1, axis=0)) / dx - (Bx - np.roll(Bx, shift=1, axis=1)) / dy) - (dt / eps0) * Jz

    # ---------------------------------------------------------
    #  Diagnostics: Plot Data (every 1000 steps)
    # ---------------------------------------------------------
    print(step)
    if step % 1000 == 0:
        plot_fields_and_particles_3d(x_e, y_e, z_e, x_i, y_i, z_i, Ex, Bz, step, Lx, Ly, Lz)

plt.show()
