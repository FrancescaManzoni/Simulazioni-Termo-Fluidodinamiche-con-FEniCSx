

import os
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import mesh as dmesh, fem, io
from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem
from basix.ufl import element, mixed_element

comm = MPI.COMM_WORLD
rank = comm.rank
TAG = (lambda s: print(s, flush=True)) if rank == 0 else (lambda s: None)

# ---- proprietà aria
rho0 = 1.225
mu   = 1.8e-5
nu   = mu / rho0
k    = 0.025
cp   = 1005.0
alpha = k/(rho0*cp)
beta  = 3.0e-3
g     = 9.81
g_vec = ufl.as_vector((0.0, -g))

# ---- tempo
t_end = 50.0        
dt    = 0.01
nsteps = int(round(t_end/dt))
save_every = 10    # ogni 0.1 s con dt=0.01

# ---- mesh (usa dimensioni native)
msh_name = "script_quadrato_0.1.msh"
if rank == 0 and not os.path.exists(msh_name):
    raise FileNotFoundError(msh_name)
ret = gmshio.read_from_msh(msh_name, comm, gdim=2)
mesh = ret if isinstance(ret, dmesh.Mesh) else ret[0]

x = mesh.geometry.x
xmin, ymin = float(np.min(x[:, 0])), float(np.min(x[:, 1]))
xmax, ymax = float(np.max(x[:, 0])), float(np.max(x[:, 1]))
H = max(xmax - xmin, ymax - ymin)
TAG(f"TAG: dimensioni mesh = {xmax-xmin:.6f} x {ymax-ymin:.6f} m")

dx = ufl.dx
dim = mesh.topology.dim
facet_dim = dim - 1
tol = 1e-12
def on_bottom(p): return np.isclose(p[1], ymin, atol=tol)
def on_top(p):    return np.isclose(p[1], ymax, atol=tol)

# ---- spazi (soluzione)
cell = mesh.ufl_cell().cellname()
Ve = element("Lagrange", cell, 2, shape=(dim,))
Qe = element("Lagrange", cell, 1)
Te = element("Lagrange", cell, 2)
W  = fem.functionspace(mesh, mixed_element([Ve, Qe]))
V, _ = W.sub(0).collapse()
Q, _ = W.sub(1).collapse()
Th = fem.functionspace(mesh, Te)

# ---- spazi (visualizzazione CG1) -> richiesti da XDMF
V1 = fem.functionspace(mesh, element("Lagrange", cell, 1, shape=(dim,)))
Q1 = fem.functionspace(mesh, element("Lagrange", cell, 1))
T1 = fem.functionspace(mesh, element("Lagrange", cell, 1))
u_viz = fem.Function(V1, name="Velocity")
p_viz = fem.Function(Q1, name="Pressure")
T_viz = fem.Function(T1, name="Temperature")

# ---- BC velocità (no-slip)
u_zero = fem.Function(V); u_zero.x.array[:] = 0.0
facets_all = dmesh.locate_entities_boundary(
    mesh, facet_dim,
    lambda p: np.isclose(p[0], xmin, atol=tol) |
              np.isclose(p[0], xmax, atol=tol) |
              np.isclose(p[1], ymin, atol=tol) |
              np.isclose(p[1], ymax, atol=tol))
dofs_u = fem.locate_dofs_topological((W.sub(0), V), facet_dim, facets_all)
bc_u = fem.dirichletbc(u_zero, dofs_u, W.sub(0))

# ---- BC temperatura: fondo caldo, TOP ISOTERMA 293 K (sx/dx naturali -> adiabatiche)
T_bottom = 298.0
T_top    = 293.0
T_init   = 293.0
T_ref    = 293.0

facets_bottom = dmesh.locate_entities_boundary(mesh, facet_dim, on_bottom)
dofs_Tb = fem.locate_dofs_topological(Th, facet_dim, facets_bottom)
bc_T_bottom = fem.dirichletbc(PETSc.ScalarType(T_bottom), dofs_Tb, Th)

facets_top = dmesh.locate_entities_boundary(mesh, facet_dim, on_top)
dofs_Ttop = fem.locate_dofs_topological(Th, facet_dim, facets_top)
bc_T_top  = fem.dirichletbc(PETSc.ScalarType(T_top), dofs_Ttop, Th)

# ---- gauge pressione (p=0 all'angolo in basso a sinistra)
def at_corner(p): return np.isclose(p[0], xmin, atol=tol) & np.isclose(p[1], ymin, atol=tol)
p_pin = fem.Function(Q); p_pin.x.array[:] = 0.0
dofs_p0 = fem.locate_dofs_geometrical((W.sub(1), Q), at_corner)
bc_p0 = fem.dirichletbc(p_pin, dofs_p0, W.sub(1))

# ---- variabili & iniziali
U, P = ufl.TrialFunctions(W)
v, q = ufl.TestFunctions(W)
u_n = fem.Function(V); u_n.x.array[:] = 0.0
p_n = fem.Function(Q); p_n.x.array[:] = 0.0
Tn  = fem.Function(Th); Tn.x.array[:] = T_init

# ---- forme (segno corretto: -beta*(T-Tref)*g)
gamma_div = PETSc.ScalarType(1e-8)
eta_brink = PETSc.ScalarType(1e-10)

a_mom = (
    (1.0/dt)*ufl.inner(U, v)*dx
    + ufl.inner(ufl.grad(U)*u_n, v)*dx
    + 2*nu*ufl.inner(ufl.sym(ufl.grad(U)), ufl.sym(ufl.grad(v)))*dx
    - ufl.inner(P, ufl.div(v))*dx
    + ufl.inner(ufl.div(U), q)*dx
    + gamma_div*ufl.inner(ufl.div(U), ufl.div(v))*dx
    + eta_brink*ufl.inner(U, v)*dx
)
L_mom = (1.0/dt)*ufl.inner(u_n, v)*dx - beta*ufl.inner((Tn - T_ref)*g_vec, v)*dx

Theta = ufl.TrialFunction(Th)
sT    = ufl.TestFunction(Th)
a_T = (1.0/dt)*ufl.inner(Theta, sT)*dx \
    + alpha*ufl.inner(ufl.grad(Theta), ufl.grad(sT))*dx \
    + ufl.inner(u_n, ufl.grad(Theta))*sT*dx
L_T = (1.0/dt)*ufl.inner(Tn, sT)*dx

opts_stokes = {"ksp_type": "gmres", "pc_type": "lu",    "ksp_rtol": 1e-8}
opts_temp   = {"ksp_type": "cg",    "pc_type": "hypre", "ksp_rtol": 1e-10}
problem_NS = LinearProblem(a_mom, L_mom, bcs=[bc_u, bc_p0], petsc_options=opts_stokes)
problem_T  = LinearProblem(a_T,   L_T,   bcs=[bc_T_bottom, bc_T_top], petsc_options=opts_temp)

# ---- XDMF
xdmf = io.XDMFFile(mesh.comm, "results_transient.xdmf", "w")
xdmf.write_mesh(mesh)
def write_viz(u_field, p_field, T_field, time_value):
    u_viz.interpolate(u_field)
    p_viz.interpolate(p_field)
    T_viz.interpolate(T_field)
    xdmf.write_function(u_viz, time_value)
    xdmf.write_function(p_viz, time_value)
    xdmf.write_function(T_viz, time_value)

# stato iniziale
write_viz(u_n, p_n, Tn, 0.0)
TAG("TAG: start time stepping")

def classify_ra(Ra):
    if Ra < 1e6:  return "laminare"
    if Ra < 1e8:  return "di transizione"
    return "tendenzialmente turbolento"

for n in range(1, nsteps+1):
    t = n*dt

    # Stokes-Boussinesq
    w_sol = problem_NS.solve()
    Uh, map_u = W.sub(0).collapse()
    Ph, map_p = W.sub(1).collapse()
    u_out = fem.Function(Uh); u_out.x.array[:] = w_sol.x.array[map_u]
    p_out = fem.Function(Ph); p_out.x.array[:] = w_sol.x.array[map_p]

    # Temperatura
    T_out = problem_T.solve()

    # Avanza
    tmp = fem.Function(V); tmp.interpolate(u_out); u_n.x.array[:] = tmp.x.array
    Tn.x.array[:] = T_out.x.array
    p_n.interpolate(p_out)

    # Ricrea problemi (dipendono da u_n,Tn)
    a_mom = (
        (1.0/dt)*ufl.inner(U, v)*dx
        + ufl.inner(ufl.grad(U)*u_n, v)*dx
        + 2*nu*ufl.inner(ufl.sym(ufl.grad(U)), ufl.sym(ufl.grad(v)))*dx
        - ufl.inner(P, ufl.div(v))*dx
        + ufl.inner(ufl.div(U), q)*dx
        + gamma_div*ufl.inner(ufl.div(U), ufl.div(v))*dx
        + eta_brink*ufl.inner(U, v)*dx
    )
    L_mom = (1.0/dt)*ufl.inner(u_n, v)*dx - beta*ufl.inner((Tn - T_ref)*g_vec, v)*dx
    problem_NS = LinearProblem(a_mom, L_mom, bcs=[bc_u, bc_p0], petsc_options=opts_stokes)

    a_T = (1.0/dt)*ufl.inner(Theta, sT)*dx \
        + alpha*ufl.inner(ufl.grad(Theta), ufl.grad(sT))*dx \
        + ufl.inner(u_n, ufl.grad(Theta))*sT*dx
    L_T = (1.0/dt)*ufl.inner(Tn, sT)*dx
    problem_T = LinearProblem(a_T, L_T, bcs=[bc_T_bottom, bc_T_top], petsc_options=opts_temp)

    # Output ogni 0.1 s + Re/Ra
    if n % save_every == 0 or n == nsteps:
        write_viz(u_n, p_n, Tn, t)
        dT_cur = float(np.max(Tn.x.array) - np.min(Tn.x.array))
        U_est  = float(np.sqrt(max(g*beta*dT_cur*H, 0.0)))
        Re_est = U_est*H/nu
        Ra_est = g*beta*dT_cur*H**3/(nu*alpha)
        TAG(f"t={t:6.1f}s | ΔT≈{dT_cur:6.2f} K  U*≈{U_est:7.4f} m/s  Re≈{Re_est:7.2f}  Ra≈{Ra_est:8.2e}  → {classify_ra(Ra_est)}")

xdmf.close()
TAG("TAG: simulazione completata → results_transient.xdmf")
