"""
plot_potential_t0.py
====================
1. EFE sanity checks at t=0 (prints diagnostic numbers).
2. Contour plot of the MOND potential in the x-z plane (shows EFE asymmetry along z).
3. Surface density plot projected onto x-z plane (shows EFE asymmetry along z).

Run from the directory that contains NBodyMONDPMeshClean.py and IsothermalClass.py.
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# NBodyMONDPMeshClean no longer imports IsothermalClass at module level,
# so there is no circular import and both can be imported directly.
import NBodyMONDPMeshClean as sim
from IsothermalClass import IsoThermalParticlelist

Particlelist            = sim.Particlelist
AssignMassGaussShapeJAX = sim.AssignMassGaussShapeJAX
Calcpot                 = sim.Calcpot
CalcAccMat              = sim.CalcAccMat
MainLoop                = sim.MainLoop
CurlFreeProj            = sim.CurlFreeProj
inpolinv                = sim.inpolinv
inpol                   = sim.inpol
KdotProd                = sim.KdotProd
halfpixels              = sim.halfpixels
celllen                 = sim.celllen
cellleninv              = sim.cellleninv
cellvolume              = sim.cellvolume
cellvolumeinv           = sim.cellvolumeinv
G                       = sim.G
a0                      = sim.a0
kstep                   = sim.kstep
K2inv                   = sim.K2inv
EFE_M                   = sim.EFE_M
EFE_on                  = sim.EFE_on
EFE_M_strength          = sim.EFE_M_strength
M                       = sim.M
N_particles             = sim.N_particles
b                       = sim.b
regime                  = sim.regime
itersteps               = sim.itersteps
size_of_box             = sim.size_of_box
datatype                = sim.datatype
ball4                   = sim.ball4
orbits                  = sim.orbits
# ── 1.  Build the initial particle list ─────────────────────────────────────
print("Building initial particle list …")
particles = IsoThermalParticlelist(M / N_particles, b, N_particles)
plist = particles.list

# ── 2.  Newtonian potential & acceleration ───────────────────────────────────
print("Computing Newtonian potential …")
density = jnp.zeros((2*halfpixels,)*3, dtype=datatype)
density = AssignMassGaussShapeJAX(density, plist, cellvolumeinv, a=1, shape=ball4)
densityfft = jnp.fft.fftn(density)
potND  = Calcpot(densityfft)
accND  = CalcAccMat(potND)

# ── 3.  Add external Newtonian field ─────────────────────────────────────────
if EFE_M[0]:
    g_N_e      = jnp.array(EFE_M[1], dtype=datatype)[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
    accND_total = accND + g_N_e
else:
    accND_total = accND

# ── 4.  MOND iteration ───────────────────────────────────────────────────────
print(f"Running {itersteps} MOND iterations …")
H = jnp.zeros([3, 2*halfpixels, 2*halfpixels, 2*halfpixels], dtype=datatype)
for _ in range(itersteps):
    gM2, H = MainLoop(H, accND_total, regime, EFE_M)

# Recover MOND potential
accMONDfft = jnp.fft.fftn(gM2, axes=(1, 2, 3))
potMONDfft = -KdotProd(accMONDfft) * K2inv / kstep
potMOND    = jnp.imag(jnp.fft.ifftn(potMONDfft))

# ── 5.  EFE SANITY CHECKS ────────────────────────────────────────────────────
print("\n" + "="*60)
print("EFE SANITY CHECKS")
print("="*60)

efe_mag  = abs(EFE_M_strength)
print(f"\n[1] External field |g_N_e| = {efe_mag:.4f} ly/Myr²")
print(f"    a0              = {a0:.4f} ly/Myr²")
print(f"    |g_N_e| / a0   = {efe_mag/a0:.4f}  "
      f"({'sub-MOND' if efe_mag/a0 < 1 else 'super-MOND'} regime)")

acc_norm = np.array(jnp.linalg.norm(accND, axis=0))
centre   = halfpixels
b_px     = int(round(b / celllen))
shell_mask = np.zeros_like(acc_norm, dtype=bool)
cx = cy = cz = centre
for ix in range(2*halfpixels):
    for iy in range(2*halfpixels):
        for iz in range(2*halfpixels):
            r2 = (ix-cx)**2 + (iy-cy)**2 + (iz-cz)**2
            if (b_px-1)**2 <= r2 <= (b_px+1)**2:
                shell_mask[ix, iy, iz] = True

a_int_at_b = float(np.mean(acc_norm[shell_mask])) if shell_mask.any() else float('nan')
print(f"\n[2] Mean |g_N_int| at r=b  = {a_int_at_b:.4f} ly/Myr²")
print(f"    |g_N_int(b)| / a0      = {a_int_at_b/a0:.4f}")
print(f"    |g_N_int(b)| / |g_N_e| = {a_int_at_b/efe_mag:.4f}  "
      f"({'EFE significant' if efe_mag > a_int_at_b else 'EFE weak'})")

acc_mond_norm = np.array(jnp.linalg.norm(gM2, axis=0))
ratio_at_b = (float(np.mean(acc_mond_norm[shell_mask])) /
              max(a_int_at_b, 1e-30)) if shell_mask.any() else float('nan')
print(f"\n[3] Mean |g_MOND| / |g_N_int| at r=b = {ratio_at_b:.4f}")

g_N_e_vec = np.array(EFE_M[1])
g_N_e_mag = np.linalg.norm(g_N_e_vec)
nu_gNe    = float(inpolinv(g_N_e_mag / a0, regime))
g_M_e_mag = nu_gNe * g_N_e_mag
print(f"\n[4] EFE direction vector   : {EFE_M[1]}")
print(f"    ν(|g_N_e|/a0)          = {nu_gNe:.4f}")
print(f"    |g_M_e|                = {g_M_e_mag:.4f} ly/Myr²")

z_lo = int(halfpixels - b_px)
z_hi = int(halfpixels + b_px)
pot_np = np.array(potMOND)
pot_lo = float(pot_np[centre, centre, z_lo])
pot_hi = float(pot_np[centre, centre, z_hi])
asym   = abs(pot_lo - pot_hi) / (0.5*abs(pot_lo + pot_hi) + 1e-30)
print(f"\n[5] Potential asymmetry along z at r=b:")
print(f"    Φ(0,0,-b) = {pot_lo:.4e},  Φ(0,0,+b) = {pot_hi:.4e}")
print(f"    Relative asymmetry = {asym:.4e}  "
      f"({'detectable' if asym > 1e-3 else 'very small or absent'})")
print("\n" + "="*60)

# ── Shared coordinate axes ────────────────────────────────────────────────────
H_px    = halfpixels
px      = np.arange(2*H_px) * celllen
px_cent = px - px[H_px]          # centred coordinates in ly
zoom    = 10 * b
theta   = np.linspace(0, 2*np.pi, 300)

# ── 6.  CONTOUR PLOT of potential in the x-z plane (y = centre) ──────────────
print("\nGenerating potential contour plot (x-z plane) …")

pot_xz = pot_np[:, H_px, :]      # shape (2H, 2H): x along axis 0, z along axis 1

fig, ax = plt.subplots(figsize=(6, 6))

# Choose contour levels symmetric around the central value, finely spaced
pot_zoom_mask = (np.abs(px_cent) <= zoom)
pot_region    = pot_xz[np.ix_(pot_zoom_mask, pot_zoom_mask)]
vmin, vmax    = np.percentile(pot_region, 2), np.percentile(pot_region, 98)
levels        = np.linspace(vmin, vmax, 30)

cf = ax.contourf(px_cent, px_cent, pot_xz.T, levels=levels, cmap='RdBu_r')
cs = ax.contour( px_cent, px_cent, pot_xz.T, levels=levels[::3], colors='k',
                 linewidths=0.5, alpha=0.6)
ax.clabel(cs, fmt='%.1f', fontsize=6, inline=True)

cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r'$\Phi_{\rm MOND}\ \left[\frac{{\rm ly}^2}{{\rm Myr}^2}\right]$', fontsize=11)

# Circle at r = b for reference
ax.plot(b*np.cos(theta), b*np.sin(theta), 'k--', lw=1.2,
        label=rf'$r = b = {b:.1f}$ ly')
#Arrow indicating EFE direction
ax.annotate('', xy=(0, 0.85*zoom), xytext=(0, 0.55*zoom),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.text(0.05*zoom, 0.72*zoom, r'$\mathbf{g}_{N,e}$', color='green', fontsize=11)

ax.set_xlim(-zoom, zoom)
ax.set_ylim(-zoom, zoom)
ax.set_xlabel(r'$x$ [ly]', fontsize=12)
ax.set_ylabel(r'$z$ [ly]', fontsize=12)
ax.set_title(r'MOND potential $\Phi(x,\,y{=}0,\,z)$ at $t=0$', fontsize=13)
ax.set_aspect('equal')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('potential_t0_xz_contour.pdf', dpi=150)
print("Saved → potential_t0_xz_contour.pdf")
plt.show()

# ── 7.  SURFACE DENSITY in the x-z plane ────────────────────────────────────
# Compute directly from the particle list by binning x and z positions.
# Each particle contributes mass m to its (x, z) bin; dividing by bin area
# gives surface density Sigma(x, z) in M_sun/ly^2.
print("\nGenerating surface density plot (x-z plane) …")

plist_np = np.array(plist)                          # (N, 7): [m, x, y, z, vx, vy, vz]
mass_pp  = plist_np[:, 0]                           # mass per particle
x_ly     = (plist_np[:, 1] - halfpixels) * celllen  # x positions in ly, centred
z_ly     = (plist_np[:, 3] - halfpixels) * celllen  # z positions in ly, centred

n_bins   = 64                                       # resolution of the surface density map
bin_edges = np.linspace(-zoom, zoom, n_bins + 1)
bin_area  = (2 * zoom / n_bins) ** 2               # ly^2 per bin

# Weighted 2D histogram: total mass per bin
sigma_hist, _, _ = np.histogram2d(
    x_ly, z_ly,
    bins=[bin_edges, bin_edges],
    weights=mass_pp,
)
sigma_hist /= bin_area                              # M_sun / ly^2

fig, ax = plt.subplots(figsize=(6, 6))

sigma_plot = sigma_hist.copy()
sigma_plot[sigma_plot <= 0] = np.nan

bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
vmin_s = np.nanpercentile(sigma_plot, 5)
vmax_s = np.nanpercentile(sigma_plot, 99.5)

im = ax.imshow(
    np.log10(sigma_plot).T,
    origin='lower',
    extent=[bin_edges[0], bin_edges[-1], bin_edges[0], bin_edges[-1]],
    cmap='inferno',
    vmin=np.log10(max(vmin_s, 1e-30)),
    vmax=np.log10(vmax_s),
    interpolation='bilinear',
)

# Overlay contours to make asymmetry legible
levels_s = np.linspace(np.log10(max(vmin_s, 1e-30)), np.log10(vmax_s), 12)
ax.contour(bin_centres, bin_centres, np.log10(sigma_plot).T,
           levels=levels_s, colors='white', linewidths=0.5, alpha=0.5)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r'$\log_{10} (\sigma(x,z))\; \left[\frac{M_\odot}{{\rm ly}^2}\right]$',
               fontsize=11)

# Circle at r = b
ax.plot(b*np.cos(theta), b*np.sin(theta), 'w--', lw=1.2,
        label=rf'$r = b = {b:.1f}$ ly')
# EFE direction arrow
ax.annotate('', xy=(0, 0.85*zoom), xytext=(0, 0.55*zoom),
            arrowprops=dict(arrowstyle='->', color='cyan', lw=2))
ax.text(0.05*zoom, 0.72*zoom, r'$\mathbf{g}_{N,e}$', color='cyan', fontsize=11)

ax.set_xlim(-zoom, zoom)
ax.set_ylim(-zoom, zoom)
ax.set_xlabel(r'$x$ [ly]', fontsize=12)
ax.set_ylabel(r'$z$ [ly]', fontsize=12)
ax.set_title(r'Surface density $\sigma(x,z)$ at $t=0$', fontsize=13)
ax.set_aspect('equal')
ax.legend(fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig('surface_density_t0_xz.pdf', dpi=150)
print("Saved → surface_density_t0_xz.pdf")
plt.show()

# ── 8.  SURFACE DENSITY at the final timestep ────────────────────────────────
import os

if os.path.exists('posmat.npy') and os.path.exists('masses.npy'):
    print("\nGenerating surface density plot at final timestep …")

    posmat_np = np.load('posmat.npy')  # [N, timesteps, 3], grid units
    masses_np = np.load('masses.npy')  # [N], M_sun

    # Final timestep positions, converted to ly centred on box centre
    x_ly_f = (posmat_np[:, -1, 0] - halfpixels) * celllen
    z_ly_f = (posmat_np[:, -1, 2] - halfpixels) * celllen

    sigma_f, _, _ = np.histogram2d(
        x_ly_f, z_ly_f,
        bins=[bin_edges, bin_edges],
        weights=masses_np,
    )
    sigma_f /= bin_area

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)


    def plot_sigma(ax, sigma, title):
        s = sigma.copy()
        s[s <= 0] = np.nan
        vmin_p = np.nanpercentile(s, 5)
        vmax_p = np.nanpercentile(s, 99.5)
        im = ax.imshow(
            np.log10(s).T,
            origin='lower',
            extent=[bin_edges[0], bin_edges[-1], bin_edges[0], bin_edges[-1]],
            cmap='inferno',
            vmin=np.log10(max(vmin_p, 1e-30)),
            vmax=np.log10(vmax_p),
            interpolation='bilinear',
        )
        levels_p = np.linspace(np.log10(max(vmin_p, 1e-30)), np.log10(vmax_p), 12)
        ax.contour(bin_centres, bin_centres, np.log10(s).T,
                   levels=levels_p, colors='white', linewidths=0.5, alpha=0.5)
        #ax.plot(b * np.cos(theta), b * np.sin(theta), 'w--', lw=1.2,
                #label=rf'$r = b = {b:.1f}$ ly')
        ax.annotate('', xy=(0, 0.85 * zoom), xytext=(0, 0.55 * zoom),
        arrowprops=dict(arrowstyle='->', color='cyan', lw=2))
        ax.text(0.05 * zoom, 0.72 * zoom, r'$\mathbf{g}_{N,e}$', color='cyan', fontsize=11)
        ax.set_xlim(-zoom, zoom)
        ax.set_ylim(-zoom, zoom)
        ax.set_xlabel(r'$x$ [ly]', fontsize=12)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  framealpha=0.8)
        return im


    im0 = plot_sigma(axes[0], sigma_hist, r'$\sigma(x,z)$ at $t=0$')
    im1 = plot_sigma(axes[1], sigma_f, r'$\sigma(x,z)$ at $t=$' + str(orbits) + r'$T_{\text{orbit}}$')

    axes[0].set_ylabel(r'$z$ [ly]', fontsize=12)
    cb = fig.colorbar(im1, ax=axes.tolist(), fraction=0.02, pad=0.02)
    cb.set_label(r'$\log_{10}(\sigma) \; \left[\frac{M_\odot}{{\rm ly}^2}\right]$', fontsize=11)

    plt.savefig('surface_density_comparison_xz.pdf', dpi=150, bbox_inches='tight')
    print("Saved → surface_density_comparison_xz.pdf")
    plt.show()
else:
    print("\nNo posmat.npy/masses.npy found — run NBodyMONDPMeshClean.py first to generate simulation data.")