import os
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, io
from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem
import ufl
from basix.ufl import element, mixed_element

comm = MPI.COMM_WORLD
rank = comm.rank

# ======================
# ====== CONFIG ========
# ======================
MESH_FILE = "quadrato_mesh.msh"  # stesso folder dello script

TAG_COLD   = 10        # sinistra -> 293 K
TAG_HOT    = 12        # destra   -> 303 K
ADIAB_TAGS = [11, 13]  # top, bottom
WALL_TAGS  = [10, 11, 12, 13]  # no-slip ovunque

T_cold = 293.0
T_hot  = 303.0

# Proprietà aria
rho0  = 1.225
mu    = 1.8e-5
nu    = mu / rho0
k     = 0.025
cp    = 1005.0
alpha = k / (rho0 * cp)
beta  = 3.0e-3
g     = 9.81
T_ref = T_cold

# ======================
# ===== Mesh & info ====
# ======================
if rank == 0 and not os.path.exists(MESH_FILE):
    raise FileNotFoundError(f"File mesh non trovato: {MESH_FILE}")

mesh, cell_tags, facet_tags = gmshio.read_from_msh(MESH_FILE, comm, gdim=2)
if rank == 0:
    print(f"TAG: mesh letta da {MESH_FILE}")

# <<< RIMOSSO controllo triangolo/quadrilatero con meshio >>>
# Nuovo: ricavo il tipo cella direttamente dal mesh di dolfinx
cell_str = mesh.ufl_cell().cellname()  # "triangle" oppure "quadrilateral"
if rank == 0:
    print(f"TAG: cell_str = {cell_str}")
    if facet_tags is not None and facet_tags.values.size > 0:
        vals, counts = np.unique(facet_tags.values, return_counts=True)
        print("TAG: facet tags (valore: count):")
        for v, c in zip(vals.tolist(), counts.tolist()):
            print(f"  - {v}: {c}")

# Scala e Rayleigh
xcoord = mesh.geometry.x
Lx = float(xcoord[:, 0].max() - xcoord[:, 0].min())
Ly = float(xcoord[:, 1].max() - xcoord[:, 1].min())
H  = max(Lx, Ly)
dx = ufl.dx
dT = T_hot - T_cold
Ra_est = g * beta * dT * (H**3) / (nu * alpha)
if rank == 0:
    print(f"TAG: Rayleigh stimato ≈ {Ra_est:.2e} (H={H:.1g} m)")
RA_CAP = 5e5
buoy_scale = min(1.0, RA_CAP / max(Ra_est, 1e-16))
if rank == 0 and buoy_scale < 1.0:
    print(f"TAG: cap buoyancy attivo: scala={buoy_scale:.3f}")
g_vec = ufl.as_vector((0.0, -g))

# ======================
# ===== Spazi FEM =====
# ======================
dim = mesh.geometry.dim
Ve = element("Lagrange", cell_str, 2, shape=(dim,))   # u P2
Qe = element("Lagrange", cell_str, 1)                 # p P1
Te = element("Lagrange", cell_str, 2)                 # T P2

W_stokes = fem.functionspace(mesh, mixed_element([Ve, Qe]))  # (u,p)
V_u = W_stokes.sub(0)   # sottospazio velocità
V_p = W_stokes.sub(1)   # sottospazio pressione

# Spazi collassati per i valori
V0s, _ = V_u.collapse()
Q1s, _ = V_p.collapse()
Th     = fem.functionspace(mesh, Te)

# ======================
# === Boundary sets ===
# ======================
facet_dim = mesh.topology.dim - 1
def _find(tag):
    try:
        return facet_tags.find(int(tag))
    except Exception:
        return np.array([], dtype=np.int32)

cold_facets   = _find(TAG_COLD)
hot_facets    = _find(TAG_HOT)
adiab_facets  = np.concatenate([_find(t) for t in ADIAB_TAGS]) if ADIAB_TAGS else np.array([], dtype=np.int32)
noslip_facets = np.concatenate([_find(t) for t in WALL_TAGS])  if WALL_TAGS  else np.array([], dtype=np.int32)

if rank == 0:
    print(f"TAG: #cold={cold_facets.size}, #hot={hot_facets.size}, #adiab={adiab_facets.size}, #noslip={noslip_facets.size}")

# ======================
# ====== BC: T =========
# ======================
Theta = ufl.TrialFunction(Th); sT = ufl.TestFunction(Th)
bc_list_T = []
if cold_facets.size > 0:
    dofs_Tc = fem.locate_dofs_topological(Th, facet_dim, cold_facets).astype(np.int32).ravel()
    bc_list_T.append(fem.dirichletbc(PETSc.ScalarType(T_cold), dofs_Tc, Th))
if hot_facets.size > 0:
    dofs_Th = fem.locate_dofs_topological(Th, facet_dim, hot_facets).astype(np.int32).ravel()
    bc_list_T.append(fem.dirichletbc(PETSc.ScalarType(T_hot), dofs_Th, Th))
# top/bottom: adiabatiche -> naturale

# ======================
# ====== BC: u =========
# ======================
u0s = fem.Function(V0s); u0s.x.array[:] = 0.0
bc_list_Stokes = []
if noslip_facets.size > 0:
    dofs_us_raw = fem.locate_dofs_topological((V_u, V0s), facet_dim, noslip_facets)
    dofs_us = [np.array(x, dtype=np.int32).ravel() for x in (dofs_us_raw if isinstance(dofs_us_raw, (list, tuple)) else [dofs_us_raw])]
    bc_list_Stokes.append(fem.dirichletbc(u0s, dofs_us, V_u))

# ======================
# ====== BC: p =========
# ======================
# p = 0 in un punto (angolo in basso a sinistra)
x_min = xcoord[:,0].min()
y_min = xcoord[:,1].min()
def at_corner(xp, tol=1e-12):
    return (np.isclose(xp[0], x_min, atol=tol) & np.isclose(xp[1], y_min, atol=tol))

p0c = fem.Constant(mesh, PETSc.ScalarType(0.0))
dofs_p0s = fem.locate_dofs_geometrical((V_p, Q1s), at_corner)
dofs_p0s = (dofs_p0s.astype(np.int32).ravel() if isinstance(dofs_p0s, np.ndarray) else np.array(dofs_p0s, dtype=np.int32).ravel())
if dofs_p0s.size == 0:
    dofs_p0s = np.array([0], dtype=np.int32)
bc_list_Stokes.append(fem.dirichletbc(p0c, dofs_p0s, V_p))

# ==========================
# ===== Conduzione (T) =====
# ==========================
a_T = alpha * ufl.inner(ufl.grad(Theta), ufl.grad(sT)) * dx
L_T = fem.Constant(mesh, PETSc.ScalarType(0.0)) * sT * dx
prob_T = LinearProblem(a_T, L_T, bcs=bc_list_T,
                       petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-12})
T_h = prob_T.solve(); T_h.name = "Temperature"
if rank == 0: print("TAG: fase conduzione OK")

# ==================================
# === Stokes + Boussinesq (lin.) ===
# ==================================
U, P = ufl.TrialFunctions(W_stokes)
v, q  = ufl.TestFunctions(W_stokes)

gamma_div = PETSc.ScalarType(1e-3)
eta_brink = fem.Constant(mesh, PETSc.ScalarType(1e-8))

a_S = (
    ufl.inner(2*nu*ufl.sym(ufl.grad(U)), ufl.sym(ufl.grad(v))) * dx
    + eta_brink * ufl.inner(U, v) * dx
    - ufl.inner(P, ufl.div(v)) * dx
    + ufl.inner(ufl.div(U), q) * dx
    + gamma_div * ufl.inner(ufl.div(U), ufl.div(v)) * dx
)
L_S = buoy_scale * beta * (T_h - T_ref) * ufl.inner(g_vec, v) * dx

stokes = LinearProblem(a_S, L_S, bcs=bc_list_Stokes,
                       petsc_options={"ksp_type": "gmres", "pc_type": "lu", "ksp_rtol": 1e-12})
w_SP = stokes.solve()
if rank == 0: print("TAG: fase Stokes OK")

# ==========================
# === Estrazione & I/O  ====
# ==========================
def split_from(Vmix, wmix, i, name):
    Vsub, map_sub = Vmix.sub(i).collapse()
    f = fem.Function(Vsub, name=name)
    f.x.array[:] = wmix.x.array[map_sub]
    return f

u_h = split_from(W_stokes, w_SP, 0, "Velocity")  # P2
p_h = split_from(W_stokes, w_SP, 1, "Pressure")  # P1

V1 = fem.functionspace(mesh, element("Lagrange", cell_str, 1, shape=(dim,)))
Q1 = fem.functionspace(mesh, element("Lagrange", cell_str, 1))
T1 = fem.functionspace(mesh, element("Lagrange", cell_str, 1))

u_viz = fem.Function(V1, name="Velocity");    u_viz.interpolate(u_h)
p_viz = fem.Function(Q1, name="Pressure");    p_viz.interpolate(p_h)
T_viz = fem.Function(T1, name="Temperature"); T_viz.interpolate(T_h)

with io.XDMFFile(mesh.comm, "results.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_viz)
    xdmf.write_function(p_viz)
    xdmf.write_function(T_viz)

if rank == 0:
    print("TAG: scritto results.xdmf (u,p,T)")
