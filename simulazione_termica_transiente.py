# simulazione_termica_transiente.py
import os
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io, mesh as dmesh
from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem
from ufl import TrialFunction, TestFunction, dx, grad, inner

# -----------------------
# Parametri principali
# -----------------------
MESH_FILE = "mesh_trapezio.msh"

# Manteniamo gli stessi ID che usavi prima, ma li creeremo noi per coordinate
TAG_FREDDO = 5   # lato sinistro (x = xmin)
TAG_CALDO  = 7   # lato destro  (x = xmax)

T_FREDDA, T_CALDA = 300.0, 600.0
T_INIT = 300.0

dt_value = 0.1
t_final  = 100.0
theta = 0.5
WRITE_EVERY = 5

# -----------------------
# Lettura mesh + marcatura bordi per coordinate
# -----------------------
comm = MPI.COMM_WORLD
rank = comm.rank

if rank == 0 and not os.path.exists(MESH_FILE):
    raise FileNotFoundError(f"File non trovato: {MESH_FILE}")
comm.barrier()

# Legge la mesh dal .msh (2D)
ret = gmshio.read_from_msh(MESH_FILE, comm, gdim=2)
mesh = ret if isinstance(ret, dmesh.Mesh) else ret[0]

# Bounding box per riconoscere i bordi
X = mesh.geometry.x
xmin, ymin = float(np.min(X[:, 0])), float(np.min(X[:, 1]))
xmax, ymax = float(np.max(X[:, 0])), float(np.max(X[:, 1]))
tol = 1e-12
fdim = mesh.topology.dim - 1

def on_left(p):   return np.isclose(p[0], xmin, atol=tol)   # freddo
def on_right(p):  return np.isclose(p[0], xmax, atol=tol)   # caldo

left_facets  = dmesh.locate_entities_boundary(mesh, fdim, on_left)
right_facets = dmesh.locate_entities_boundary(mesh, fdim, on_right)

# Se vuoi, potresti creare anche top/bottom, ma qui non servono per le BC

# Crea Meshtags con i due insiemi (ordinati come richiesto da dolfinx)
facet_indices = np.concatenate([left_facets, right_facets]).astype(np.int32)
facet_values  = np.concatenate([
    np.full(left_facets.size,  TAG_FREDDO, dtype=np.int32),
    np.full(right_facets.size, TAG_CALDO,  dtype=np.int32),
])
order = np.argsort(facet_indices)
facet_tags = dmesh.meshtags(mesh, fdim, facet_indices[order], facet_values[order])

# -----------------------
# Spazi, BC e problema 
# -----------------------
V = fem.functionspace(mesh, ("Lagrange", 1))

def facets_by_tag(tag: int):
    facets = facet_tags.find(tag)
    if facets.size == 0:
        raise RuntimeError(f"Nessuna faccia con tag {tag} trovata nella mesh!")
    return facets

Tfredda = fem.Constant(mesh, PETSc.ScalarType(T_FREDDA))
Tcalda  = fem.Constant(mesh, PETSc.ScalarType(T_CALDA))

left_facets  = facets_by_tag(TAG_FREDDO)
right_facets = facets_by_tag(TAG_CALDO)

left_dofs  = fem.locate_dofs_topological(V, fdim, left_facets)
right_dofs = fem.locate_dofs_topological(V, fdim, right_facets)

bc_left  = fem.dirichletbc(Tfredda, left_dofs,  V)
bc_right = fem.dirichletbc(Tcalda,  right_dofs, V)
bcs = [bc_left, bc_right]

u, v = TrialFunction(V), TestFunction(V)

u_n = fem.Function(V, name="T")
u_n.x.array[:] = T_INIT

dt = fem.Constant(mesh, PETSc.ScalarType(dt_value))
alpha = fem.Constant(mesh, PETSc.ScalarType(1.0))  # diffusività unitaria

a = (u*v + dt*alpha*theta*inner(grad(u), grad(v))) * dx
L = (u_n*v - dt*alpha*(1-theta)*inner(grad(u_n), grad(v))) * dx

problem = LinearProblem(a, L, bcs=bcs)

# -----------------------
# Time stepping + output 
# -----------------------
t = 0.0
istep = 0
with io.XDMFFile(mesh.comm, "soluzione_termica_transiente.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_n, t)
    while t < t_final - 1e-14:
        uh = problem.solve()
        u_n.x.array[:] = uh.x.array[:]
        t += float(dt.value)
        istep += 1
        if istep % WRITE_EVERY == 0 or t >= t_final:
            xdmf.write_function(u_n, t)
