#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# === Configurazione ===
from mpi4py import MPI
from petsc4py import PETSc  
import numpy as np
import ufl, os

from dolfinx import fem, mesh as dmesh
from dolfinx.io import XDMFFile, gmshio
from dolfinx.fem.petsc import LinearProblem
from basix.ufl import mixed_element

# === Parametri simulazione ===
MESH_FILE = "script _cilindro.msh"  
P_IN, P_OUT = 8.0, 0.0              
EPS_P = 1e-10                      

comm = MPI.COMM_WORLD
rank = comm.rank

# === Lettura mesh  ===========
if rank == 0 and not os.path.exists(MESH_FILE):
    raise FileNotFoundError(f"File non trovato: {MESH_FILE}")
comm.barrier()

# 1) Leggi la mesh dal file .msh
ret = gmshio.read_from_msh(MESH_FILE, comm, gdim=2)
mesh = ret if isinstance(ret, dmesh.Mesh) else ret[0]

# 2) Bounding box per riconoscere i bordi per coordinate
X = mesh.geometry.x
xmin, ymin = float(np.min(X[:, 0])), float(np.min(X[:, 1]))
xmax, ymax = float(np.max(X[:, 0])), float(np.max(X[:, 1]))
tol = 1e-12
dim = mesh.topology.dim
fdim = dim - 1

def on_inlet(p):   return np.isclose(p[0], xmin, atol=tol)
def on_outlet(p):  return np.isclose(p[0], xmax, atol=tol)
def on_bottom(p):  return np.isclose(p[1], ymin, atol=tol)
def on_top(p):     return np.isclose(p[1], ymax, atol=tol)

# 3) Individua i facet di ciascun bordo
inlet_facets  = dmesh.locate_entities_boundary(mesh, fdim, on_inlet)
outlet_facets = dmesh.locate_entities_boundary(mesh, fdim, on_outlet)
walls_facets  = dmesh.locate_entities_boundary(
    mesh, fdim, lambda p: on_bottom(p) | on_top(p)
)

# 4) Crea Meshtags: 10=inlet, 11=walls, 12=outlet
facet_indices = np.concatenate([inlet_facets, walls_facets, outlet_facets]).astype(np.int32)
facet_values  = np.concatenate([
    np.full(inlet_facets.size,  10, dtype=np.int32),
    np.full(walls_facets.size,  11, dtype=np.int32),
    np.full(outlet_facets.size, 12, dtype=np.int32),
])
# ordinamento richiesto da DOLFINx
order = np.argsort(facet_indices)
facet_tags = dmesh.meshtags(mesh, fdim, facet_indices[order], facet_values[order])

# 5) Misura di bordo
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

# alias ID per chiarezza
inlet_id, walls_id, outlet_id = 10, 11, 12

if rank == 0:
    print(f"Mesh: {mesh.topology.index_map(mesh.topology.dim).size_local} celle,"
          f" {mesh.geometry.x.shape[0]} nodi")
    print(f"Bordi: inlet={inlet_facets.size}, walls={walls_facets.size}, outlet={outlet_facets.size}")

# === Spazi finiti e condizioni al contorno ================================
try:
    V = fem.functionspace(mesh, ("Lagrange", 2, (mesh.geometry.dim,)))
except Exception:
    V = fem.functionspace(mesh, ("CG", 2, (mesh.geometry.dim,)))
try:
    Q = fem.functionspace(mesh, ("Lagrange", 1))
except Exception:
    Q = fem.functionspace(mesh, ("CG", 1))

W = fem.functionspace(mesh, mixed_element([V.ufl_element(), Q.ufl_element()]))

V0, _ = W.sub(0).collapse()
u0 = fem.Function(V0); u0.x.array[:] = 0.0
dofs_walls = fem.locate_dofs_topological((W.sub(0), V0), fdim, facet_tags.find(walls_id))
bc_walls = fem.dirichletbc(u0, dofs_walls, W.sub(0))
bcs = [bc_walls]

# === Formulazione variazionale Stokes (trazione di pressione) =============
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)

a = (ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
     - p * ufl.div(v) * ufl.dx
     + q * ufl.div(u) * ufl.dx
     + EPS_P * p * q * ufl.dx)

n = ufl.FacetNormal(mesh)
L = (- P_IN  * ufl.dot(n, v) * ds(inlet_id)
     - P_OUT * ufl.dot(n, v) * ds(outlet_id))

# === Risoluzione lineare ===================================================
problem = LinearProblem(
    a, L, bcs=bcs,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"}  # niente petsc_options_prefix
)
Uh = problem.solve()

u_h = Uh.sub(0).copy(); u_h.name = "u"
p_h = Uh.sub(1).copy(); p_h.name = "p"

# === Export XDMF (grado coerente con la geometria) ========================
mesh_deg = getattr(mesh.geometry, "degree", 1)
try:
    Vout = fem.functionspace(mesh, ("Lagrange", mesh_deg, (mesh.geometry.dim,)))
except Exception:
    Vout = fem.functionspace(mesh, ("CG", mesh_deg, (mesh.geometry.dim,)))
try:
    Qout = fem.functionspace(mesh, ("Lagrange", mesh_deg))
except Exception:
    Qout = fem.functionspace(mesh, ("CG", mesh_deg))

u_vis = fem.Function(Vout); u_vis.name = "u"; u_vis.interpolate(u_h)
p_vis = fem.Function(Qout); p_vis.name = "p"; p_vis.interpolate(p_h)

out_name = "poiseuille_results.xdmf"
with XDMFFile(mesh.comm, out_name, "w") as xdmf:
    u_vis.x.scatter_forward(); p_vis.x.scatter_forward()
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_vis, t=0.0)
    xdmf.write_function(p_vis, t=0.0)

if rank == 0:
    print(f"✅ Output scritto: {out_name}")

