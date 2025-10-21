#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh as dmesh, fem
from dolfinx.io import gmshio, XDMFFile
from dolfinx.fem.petsc import LinearProblem
from ufl import TrialFunction, TestFunction, dx, grad, inner

MESH_FILE = "mesh_trapezio.msh"
T_FREDDA = 300.0
T_CALDA  = 600.0
TAG_FREDDO = 5      # sinistra (x = xmin)
TAG_CALDO  = 7      # destra   (x = xmax)

comm = MPI.COMM_WORLD
rank = comm.rank

# --- Lettura mesh ---
if rank == 0 and not os.path.exists(MESH_FILE):
    raise FileNotFoundError(f"File non trovato: {MESH_FILE}")
comm.barrier()

ret = gmshio.read_from_msh(MESH_FILE, comm, gdim=2)
domain = ret if isinstance(ret, dmesh.Mesh) else ret[0]

# --- Marcatura bordi per coordinate ---
X = domain.geometry.x
xmin, ymin = float(np.min(X[:, 0])), float(np.min(X[:, 1]))
xmax, ymax = float(np.max(X[:, 0])), float(np.max(X[:, 1]))
tol = 1e-12
fdim = domain.topology.dim - 1

def on_left(p):  return np.isclose(p[0], xmin, atol=tol)
def on_right(p): return np.isclose(p[0], xmax, atol=tol)

left_facets  = dmesh.locate_entities_boundary(domain, fdim, on_left)
right_facets = dmesh.locate_entities_boundary(domain, fdim, on_right)

facet_indices = np.concatenate([left_facets, right_facets]).astype(np.int32)
facet_values  = np.concatenate([
    np.full(left_facets.size,  TAG_FREDDO, dtype=np.int32),
    np.full(right_facets.size, TAG_CALDO,  dtype=np.int32),
])
order = np.argsort(facet_indices)
facet_tags = dmesh.meshtags(domain, fdim, facet_indices[order], facet_values[order])

if rank == 0:
    cells = domain.topology.index_map(domain.topology.dim).size_local
    print(f"Mesh: {cells} celle, {domain.geometry.x.shape[0]} nodi")
    print(f"Facce sinistra={left_facets.size}, destra={right_facets.size}")

# --- Spazio e BC ---
V = fem.functionspace(domain, ("Lagrange", 1))
Tfredda = fem.Constant(domain, PETSc.ScalarType(T_FREDDA))
Tcalda  = fem.Constant(domain, PETSc.ScalarType(T_CALDA))

left_dofs  = fem.locate_dofs_topological(V, fdim, facet_tags.find(TAG_FREDDO))
right_dofs = fem.locate_dofs_topological(V, fdim, facet_tags.find(TAG_CALDO))
bc_left  = fem.dirichletbc(Tfredda, left_dofs,  V)
bc_right = fem.dirichletbc(Tcalda,  right_dofs, V)
bcs = [bc_left, bc_right]

# --- Problema termico stazionario ---
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v)) * dx
L = fem.Constant(domain, PETSc.ScalarType(0.0)) * v * dx

problem = LinearProblem(a, L, bcs=bcs)
uh = problem.solve()
uh.name = "T"

# --- Output XDMF (niente 'io.', uso direttamente XDMFFile) ---
with XDMFFile(comm, "soluzione_termica.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

if rank == 0:
    print("✅ Output scritto: soluzione_termica.xdmf")
