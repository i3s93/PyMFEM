'''
   MFEM example 1

   See c++ version in the MFEM library for more detail 
'''
from mfem import path
import mfem.ser as mfem
from mfem.ser import intArray
from os.path import expanduser, join
import numpy as np

order = 1
static_cond = False
meshfile = expanduser(join(path, 'data', 'star.mesh'))
mesh = mfem.Mesh(meshfile, 1,1)

dim = mesh.Dimension()

#   3. Refine the mesh to increase the resolution. In this example we do
#      'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
#      largest number that gives a final mesh with no more than 50,000
#      elements.
ref_levels = int(np.floor(np.log(50000./mesh.GetNE())/np.log(2.)/dim))
for x in range(ref_levels):
   mesh.UniformRefinement();

#5. Define a finite element space on the mesh. Here we use vector finite
#   elements, i.e. dim copies of a scalar finite element space. The vector
#   dimension is specified by the last argument of the FiniteElementSpace
#   constructor. For NURBS meshes, we use the (degree elevated) NURBS space
#   associated with the mesh nodes.
if order > 0:
    fec = mfem.H1_FECollection(order, dim)
elif mesh.GetNodes():
    fec = mesh.GetNodes().OwnFEC()
    prinr( "Using isoparametric FEs: " + str(fec.Name()));
else:
    order = 1
    fec = mfem.H1_FECollection(order, dim)
fespace = mfem.FiniteElementSpace(mesh, fec)
print('Number of finite element unknowns: '+
       str(fespace.GetTrueVSize()))
# 5. Determine the list of true (i.e. conforming) essential boundary dofs.
#    In this example, the boundary conditions are defined by marking all
#    the boundary attributes from the mesh as essential (Dirichlet) and
#    converting them to a list of true dofs.
ess_tdof_list = intArray()
if mesh.bdr_attributes.Size()>0:
    ess_bdr = intArray([1]*mesh.bdr_attributes.Max())
    ess_bdr = intArray(mesh.bdr_attributes.Max())
    ess_bdr.Assign(1)
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)
#6. Set up the linear form b(.) which corresponds to the right-hand side of
#   the FEM linear system, which in this case is (1,phi_i) where phi_i are
#   the basis functions in the finite element fespace.
b = mfem.LinearForm(fespace)
one = mfem.ConstantCoefficient(1.0)
b.AddDomainIntegrator(mfem.DomainLFIntegrator(one))
b.Assemble();
#7. Define the solution vector x as a finite element grid function
#   corresponding to fespace. Initialize x with initial guess of zero,
#   which satisfies the boundary conditions.
x = mfem.GridFunction(fespace);
x.Assign(0.0)
#8. Set up the bilinear form a(.,.) on the finite element space
#   corresponding to the Laplacian operator -Delta, by adding the Diffusion
#   domain integrator.
a = mfem.BilinearForm(fespace);
a.AddDomainIntegrator(mfem.DiffusionIntegrator(one))
#9. Assemble the bilinear form and the corresponding linear system,
#   applying any necessary transformations such as: eliminating boundary
#   conditions, applying conforming constraints for non-conforming AMR,
#   static condensation, etc.
if static_cond: a.EnableStaticCondensation()
a.Assemble();

A = mfem.SparseMatrix()
B = mfem.Vector()
X = mfem.Vector()
a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
print("Size of linear system: " + str(A.Size()))
# 10. Solve 
M = mfem.GSSmoother(A)
mfem.PCG(A, M, B, X, 1, 200, 1e-12, 0.0);
# 11. Recover the solution as a finite element grid function.
a.RecoverFEMSolution(X, b, x)
# 12. Save the refined mesh and the solution. This output can be viewed later
#     using GLVis: "glvis -m refined.mesh -g sol.gf".
mesh.PrintToFile('refined.mesh', 8)
x.SaveToFile('sol.gf', 8)