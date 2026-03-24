"""
2D Lattice Boltzmann Method (LBM) — Flow around a NACA 4-digit airfoil
=======================================================================
Scheme  : D2Q9  (9-velocity, 2-dimensional)
Collision: BGK single-relaxation-time
BC      : Zou-He inlet/outlet, bounce-back on airfoil surface
Libraries: numpy, matplotlib  (pure open-source, no special LBM package)

Usage
-----
    python lbm_naca.py

Outputs
-------
    lbm_naca_flow.png  — four-panel figure (velocity magnitude, vorticity,
                         pressure, streamlines)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import time

# ──────────────────────────────────────────────────────────────────────────────
# D2Q9 lattice constants
# ──────────────────────────────────────────────────────────────────────────────
# Velocity directions (cx, cy) and weights w
NQ = 9
CX = np.array([ 0,  1,  0, -1,  0,  1, -1, -1,  1], dtype=np.int32)
CY = np.array([ 0,  0,  1,  0, -1,  1,  1, -1, -1], dtype=np.int32)
W  = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
# Opposite direction index (for bounce-back)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

# ──────────────────────────────────────────────────────────────────────────────
# Simulation parameters
# ──────────────────────────────────────────────────────────────────────────────
NX       = 520          # grid width  (x)
NY       = 180          # grid height (y)
CHORD    = 90           # airfoil chord length in lattice units
NACA     = "2412"       # NACA 4-digit designation
AoA_DEG  = 8.0          # angle of attack (degrees)
U_INF    = 0.08         # inlet velocity (lattice units; keep < 0.15 for stability)
RE       = 500          # Reynolds number  →  ν = U·C / Re
NU       = U_INF * CHORD / RE
TAU      = 3*NU + 0.5   # relaxation time
OMEGA    = 1.0 / TAU
N_STEPS  = 8_000        # total time steps
SAVE_STEP= N_STEPS - 1  # which step to visualise

print(f"D2Q9 LBM  |  grid {NX}×{NY}  |  chord {CHORD}  |  NACA {NACA}")
print(f"Re={RE}   U_inf={U_INF}   nu={NU:.5f}   tau={TAU:.4f}   omega={OMEGA:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# NACA 4-digit airfoil geometry
# ──────────────────────────────────────────────────────────────────────────────
def naca4_points(naca_str, n_pts=400):
    """Return (x, y_upper, y_lower) for a NACA 4-digit profile, x in [0,1]."""
    m  = int(naca_str[0]) / 100.0   # max camber
    p  = int(naca_str[1]) / 10.0    # max-camber position
    t  = int(naca_str[2:]) / 100.0  # thickness

    beta = np.linspace(0, np.pi, n_pts)
    x    = (1 - np.cos(beta)) / 2   # cosine spacing → denser near LE/TE

    # thickness distribution
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
              + 0.2843*x**3 - 0.1015*x**4)

    # camber line
    yc = np.where(x < p,
                  m/p**2 * (2*p*x - x**2),
                  m/(1-p)**2 * ((1-2*p) + 2*p*x - x**2))

    dyc = np.where(x < p,
                   2*m/p**2 * (p - x),
                   2*m/(1-p)**2 * (p - x))
    theta = np.arctan(dyc)

    xu = x  - yt*np.sin(theta)
    yu = yc + yt*np.cos(theta)
    xl = x  + yt*np.sin(theta)
    yl = yc - yt*np.cos(theta)
    return xu, yu, xl, yl

def build_airfoil_mask(NX, NY, chord, naca_str, aoa_deg, x_offset, y_offset):
    """Return boolean mask (NY, NX) — True inside the airfoil."""
    xu, yu, xl, yl = naca4_points(naca_str, n_pts=600)
    aoa = np.radians(aoa_deg)

    def rotate(x, y):
        xr =  x*np.cos(aoa) + y*np.sin(aoa)
        yr = -x*np.sin(aoa) + y*np.cos(aoa)
        return xr, yr

    # rotate & scale to lattice units
    xur, yur = rotate(xu - 0.25, yu); xur = xur*chord + x_offset; yur = yur*chord + y_offset
    xlr, ylr = rotate(xl - 0.25, yl); xlr = xlr*chord + x_offset; ylr = ylr*chord + y_offset

    # Rasterise: for every column x, find upper/lower y extent
    mask = np.zeros((NY, NX), dtype=bool)
    # build polygon (upper surface CW + lower surface CCW)
    poly_x = np.concatenate([xur, xlr[::-1]])
    poly_y = np.concatenate([yur, ylr[::-1]])

    # Scanline fill
    xi = np.arange(NX)
    yi = np.arange(NY)
    # Use matplotlib path for rasterisation (pure Python, no extra deps)
    from matplotlib.path import Path
    path = Path(np.column_stack([poly_x, poly_y]))
    xg, yg = np.meshgrid(xi, yi)
    pts = np.column_stack([xg.ravel(), yg.ravel()])
    inside = path.contains_points(pts).reshape(NY, NX)
    return inside

# Position airfoil: leading-edge at ~25% from inlet, vertically centred
x_off = int(NX * 0.28)
y_off = NY // 2
print("Building airfoil mask …", end=" ", flush=True)
SOLID = build_airfoil_mask(NX, NY, CHORD, NACA, AoA_DEG, x_off, y_off)
print(f"done  ({SOLID.sum()} solid nodes)")

# ──────────────────────────────────────────────────────────────────────────────
# Equilibrium distribution
# ──────────────────────────────────────────────────────────────────────────────
def feq(rho, ux, uy):
    """Compute D2Q9 equilibrium distribution. Returns array (NQ, NY, NX)."""
    uu = ux*ux + uy*uy
    feq_out = np.empty((NQ, NY, NX))
    for q in range(NQ):
        cu = CX[q]*ux + CY[q]*uy
        feq_out[q] = W[q] * rho * (1 + 3*cu + 4.5*cu*cu - 1.5*uu)
    return feq_out

# ──────────────────────────────────────────────────────────────────────────────
# Initialise fields
# ──────────────────────────────────────────────────────────────────────────────
rho = np.ones((NY, NX))
ux  = np.full((NY, NX), U_INF)
uy  = np.zeros((NY, NX))
f   = feq(rho, ux, uy)

# ──────────────────────────────────────────────────────────────────────────────
# Main LBM loop
# ──────────────────────────────────────────────────────────────────────────────
print(f"Running {N_STEPS} steps …")
t0 = time.time()

for step in range(N_STEPS):

    # 1. Macroscopic quantities
    rho = f.sum(axis=0)
    ux  = (CX[:, None, None] * f).sum(axis=0) / rho
    uy  = (CY[:, None, None] * f).sum(axis=0) / rho

    # 2. Zou-He inlet BC (left wall, x=0): fixed velocity
    #    Use density extrapolation + momentum BC
    x = 0
    rho[:, x] = (f[0, :, x] + f[2, :, x] + f[4, :, x]
                 + 2*(f[3, :, x] + f[6, :, x] + f[7, :, x])) / (1 - U_INF)
    ux[:, x]  = U_INF
    uy[:, x]  = 0.0
    f[1, :, x] = f[3, :, x] + (2/3)*rho[:, x]*U_INF
    f[5, :, x] = f[7, :, x] - 0.5*(f[2, :, x]-f[4, :, x]) + (1/6)*rho[:, x]*U_INF
    f[8, :, x] = f[6, :, x] + 0.5*(f[2, :, x]-f[4, :, x]) + (1/6)*rho[:, x]*U_INF

    # 3. Collision (BGK)
    fstar = f - OMEGA * (f - feq(rho, ux, uy))

    # 4. Streaming
    fnew = np.empty_like(fstar)
    for q in range(NQ):
        fnew[q] = np.roll(np.roll(fstar[q], CX[q], axis=1), CY[q], axis=0)

    # 5. Bounce-back on solid nodes
    for q in range(NQ):
        fnew[q][SOLID] = fstar[OPP[q]][SOLID]

    # 6. Outlet BC (right wall, x=NX-1): zero-gradient / do-nothing
    fnew[:, :, -1] = fnew[:, :, -2]

    # 7. Free-slip top/bottom walls
    # top row (y=NY-1): reflect y-component
    fnew[4, -1, :] = fnew[2, -1, :]
    fnew[7, -1, :] = fnew[5, -1, :]
    fnew[8, -1, :] = fnew[6, -1, :]
    # bottom row (y=0)
    fnew[2,  0, :] = fnew[4,  0, :]
    fnew[5,  0, :] = fnew[7,  0, :]
    fnew[6,  0, :] = fnew[8,  0, :]

    f = fnew

    if (step+1) % 1000 == 0:
        elapsed = time.time() - t0
        mlups = NX * NY * (step+1) / elapsed / 1e6
        print(f"  step {step+1:5d}/{N_STEPS}   {elapsed:5.1f}s   {mlups:.2f} MLUPS")

print(f"Simulation done in {time.time()-t0:.1f}s")

# ──────────────────────────────────────────────────────────────────────────────
# Post-processing
# ──────────────────────────────────────────────────────────────────────────────
# Final macroscopic fields
rho_f = f.sum(axis=0)
ux_f  = (CX[:, None, None] * f).sum(axis=0) / rho_f
uy_f  = (CY[:, None, None] * f).sum(axis=0) / rho_f

# Mask solid nodes for plotting
ux_plot = np.where(SOLID, np.nan, ux_f)
uy_plot = np.where(SOLID, np.nan, uy_f)
rho_plot= np.where(SOLID, np.nan, rho_f)

# Velocity magnitude
umag = np.sqrt(ux_plot**2 + uy_plot**2)

# Vorticity  ω_z = ∂uy/∂x − ∂ux/∂y  (central differences)
dvdx = (np.roll(uy_plot, -1, axis=1) - np.roll(uy_plot, 1, axis=1)) / 2
dudy = (np.roll(ux_plot, -1, axis=0) - np.roll(ux_plot, 1, axis=0)) / 2
vort = dvdx - dudy
vort = np.where(SOLID, np.nan, vort)

# Pressure  p = cs² ρ  (cs² = 1/3)
pres = rho_plot / 3.0

# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────
print("Plotting …")

fig, axes = plt.subplots(2, 2, figsize=(18, 10),
                          facecolor="#0d0d0d")
fig.suptitle(f"LBM Flow — NACA {NACA}  |  AoA = {AoA_DEG}°  |  Re = {RE}",
             fontsize=15, color="white", y=0.97)

cmap_vel  = "inferno"
cmap_vort = "RdBu_r"
cmap_pres = "coolwarm"
cmap_stream = "viridis"

extent = [0, NX, 0, NY]

def add_airfoil_patch(ax):
    from matplotlib.colors import ListedColormap
    ax.imshow(SOLID.astype(float), origin="lower", extent=extent,
              cmap=ListedColormap(["none", "#cccccc"]),
              vmin=0, vmax=1, zorder=5)

def styled_ax(ax, title):
    ax.set_facecolor("#111111")
    ax.set_title(title, color="white", fontsize=11, pad=6)
    ax.tick_params(colors="gray", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")

# ── Panel 1: Velocity magnitude ───────────────────────────────────────────────
ax = axes[0, 0]
styled_ax(ax, "Velocity Magnitude  |U|")
im = ax.imshow(umag, origin="lower", extent=extent,
               cmap=cmap_vel, vmin=0, vmax=U_INF*1.8)
add_airfoil_patch(ax)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.ax.yaxis.set_tick_params(color="gray", labelcolor="gray", labelsize=7)
cb.set_label("lattice u", color="gray", fontsize=8)

# ── Panel 2: Vorticity ────────────────────────────────────────────────────────
ax = axes[0, 1]
styled_ax(ax, "Vorticity  ωz = ∂uy/∂x − ∂ux/∂y")
vmax = np.nanpercentile(np.abs(vort), 99)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = ax.imshow(vort, origin="lower", extent=extent,
               cmap=cmap_vort, norm=norm)
add_airfoil_patch(ax)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.ax.yaxis.set_tick_params(color="gray", labelcolor="gray", labelsize=7)
cb.set_label("ωz", color="gray", fontsize=8)

# ── Panel 3: Pressure ─────────────────────────────────────────────────────────
ax = axes[1, 0]
styled_ax(ax, "Pressure  p = ρ/3")
pmin = np.nanpercentile(pres, 1)
pmax = np.nanpercentile(pres, 99)
pmid = np.nanmedian(pres)
norm_p = TwoSlopeNorm(vmin=pmin, vcenter=pmid, vmax=pmax)
im = ax.imshow(pres, origin="lower", extent=extent,
               cmap=cmap_pres, norm=norm_p)
add_airfoil_patch(ax)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.ax.yaxis.set_tick_params(color="gray", labelcolor="gray", labelsize=7)
cb.set_label("p (lattice)", color="gray", fontsize=8)

# ── Panel 4: Streamlines ──────────────────────────────────────────────────────
ax = axes[1, 1]
styled_ax(ax, "Streamlines (coloured by |U|)")
X, Y = np.meshgrid(np.arange(NX), np.arange(NY))
# Replace NaN with 0 for streamplot
ux_s = np.where(np.isnan(ux_plot), 0, ux_plot)
uy_s = np.where(np.isnan(uy_plot), 0, uy_plot)
speed = np.sqrt(ux_s**2 + uy_s**2)
ax.streamplot(X, Y, ux_s, uy_s,
              color=speed, cmap=cmap_stream,
              linewidth=0.6, density=2.0, arrowsize=0.7,
              norm=plt.Normalize(0, U_INF*1.8))
add_airfoil_patch(ax)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = "lbm_naca_flow.png"
plt.savefig(out_path, dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out_path}")
