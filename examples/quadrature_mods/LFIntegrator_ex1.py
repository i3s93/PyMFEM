'''
   MFEM example 1 (converted from ex1.cpp)

   See c++ version in the MFEM library for more detail

   How to run:
      python <arguments>

   Example of arguments:
      ex1.py -m star.mesh
      ex1.py -m square-disc.mesh
      ex1.py -m escher.mesh
      ex1.py -m fichera.mesh
      ex1.py -m square-disc-p3.mesh -o 3
      ex1.py -m square-disc-nurbs.mesh -o -1
      ex1.py -m disc-nurbs.mesh -o -1
      ex1.py -m pipe-nurbs.mesh -o -1
      ex1.py -m star-surf.mesh
      ex1.py -m square-disc-surf.mesh
      ex1.py -m inline-segment.mesh
      ex1.py -m amr-quad.mesh
      ex1.py -m amr-hex.mesh
      ex1.py -m fichera-amr.mesh
      ex1.py -m mobius-strip.mesh
      ex1.py -m mobius-strip.mesh -o -1 -sc

   Description:  This example code demonstrates the use of MFEM to define a
                 simple finite element discretization of the Laplace problem
                 -Delta u = 1 with homogeneous Dirichlet boundary conditions.

'''
import os
from os.path import expanduser, join
import numpy as np

import mfem.ser as mfem


def run(order=1, static_cond=False,
        meshfile='', visualization=False,
        device='cpu', pa=False):
    '''
    run ex1
    '''
    device = mfem.Device(device)
    device.Print()

    mesh = mfem.Mesh(meshfile, 1, 1)
    dim = mesh.Dimension()

    #   3. Refine the mesh to increase the resolution. In this example we do
    #      'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
    #      largest number that gives a final mesh with no more than 50,000
    #      elements.
    ref_levels = int(
        np.floor(
            np.log(
                50000. /
                mesh.GetNE()) /
            np.log(2.) /
            dim))
    for x in range(ref_levels):
        mesh.UniformRefinement()

    # 5. Define a finite element space on the mesh. Here we use vector finite
    #   elements, i.e. dim copies of a scalar finite element space. The vector
    #   dimension is specified by the last argument of the FiniteElementSpace
    #   constructor. For NURBS meshes, we use the (degree elevated) NURBS space
    #   associated with the mesh nodes.
    if order > 0:
        fec = mfem.H1_FECollection(order, dim)
    elif mesh.GetNodes():
        fec = mesh.GetNodes().OwnFEC()
        print("Using isoparametric FEs: " + str(fec.Name()))
    else:
        order = 1
        fec = mfem.H1_FECollection(order, dim)
    fespace = mfem.FiniteElementSpace(mesh, fec)
    print('Number of finite element unknowns: ' +
          str(fespace.GetTrueVSize()))

    # 5. Determine the list of true (i.e. conforming) essential boundary dofs.
    #    In this example, the boundary conditions are defined by marking all
    #    the boundary attributes from the mesh as essential (Dirichlet) and
    #    converting them to a list of true dofs.
    ess_tdof_list = mfem.intArray()
    if mesh.bdr_attributes.Size() > 0:
        ess_bdr = mfem.intArray([1] * mesh.bdr_attributes.Max())
        ess_bdr = mfem.intArray(mesh.bdr_attributes.Max())
        ess_bdr.Assign(1)
        fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)

    # 6. Set up the linear form b(.) which corresponds to the right-hand side of
    #   the FEM linear system, which in this case is (1,phi_i) where phi_i are
    #   the basis functions in the finite element fespace.
    b = mfem.LinearForm(fespace)
    one = mfem.ConstantCoefficient(1.0)
    b.AddDomainIntegrator(mfem.DomainLFIntegrator(one))
    b.Assemble()







    # Test: First, get the integration rules based on the order and geometry
    # This will fill a list with various rules based on the different element
    # geometries, in which a particular element geometry can be accessed.
    int_rules = [mfem.IntRules.Get(i, order) for i in range(mfem.Geometry.NumGeom)]

    # Set the number of DoF according to the FE space
    num_dof = fespace.GetNDofs()

    # Create an array to hold the global data for the RHS
    # This is basically just the linear form
    global_vector = mfem.Vector(np.zeros([num_dof]))

    #------------------------------------------------------------------------------
    # Test: Storage for the quadrature points and weights on the mesh
    #------------------------------------------------------------------------------
    num_elem = mesh.GetNE()

    # Set the order of the integration rule and number of quadrature points
    intorder = 2*fespace.GetFE(0).GetOrder()
    num_quad_pts = mfem.IntRules.Get(fespace.GetFE(0).GetGeomType(), intorder).GetNPoints()

    # Quadrature data at physical points
    quad_wts = np.zeros([num_elem, num_quad_pts])
    quad_pts = np.zeros([num_elem, num_quad_pts, 3]) 

    # Shape functions at the physical points
    # We assume the number of active shape functions are identical across elements
    shape_at_quad_pts = np.zeros([num_elem, num_quad_pts, fespace.GetFE(0).GetDof()])

    #------------------------------------------------------------------------------

    # Next, loop over the mesh elements and perform certain element-wise operations
    for i in range(mesh.GetNE()):


        # Get the particular element
        element = fespace.GetFE(i)
        dof = element.GetDof()

        # Get the indices for the DoF on this element
        # This tells us which entries we write to in the global vector
        vdofs = fespace.GetElementVDofs(i)
        vdofs = mfem.intArray(vdofs)

        # Get the element's transformation, which will maps the ir's reference points
        Tr = mesh.GetElementTransformation(i)
        ir = mfem.IntRules.Get(element.GetGeomType(), intorder)

        # Storage for the basis functions and local dof on this element
        # This should always be smaller than the size of the global vector
        shape = mfem.Vector(np.zeros([dof]))
        local_vector = mfem.Vector(np.zeros([dof]))

        for j in range(ir.GetNPoints()):

            # Get the integration point from the rule
            ip = ir.IntPoint(j)

            # Set an integration point in the element transformation
            Tr.SetIntPoint(ip)

            # Transform the reference integration point to a physical location
            transip = mfem.Vector(np.zeros([3]))
            Tr.Transform(ip, transip)

            # Next, evaluate all the basis functions at this integration point
            element.CalcPhysShape(Tr, shape)

            # Compute the adjusted quadrature weight (volume factors)
            wt = ip.weight*Tr.Weight()

            # Store the contributions of the shape functions in the local vector
            # No overload for the operator *= with float types, so we cast as a 
            # Numpy array (shallow copy)
            local_vector += mfem.Vector(wt*shape.GetDataArray())
            
            # Store the relevant quadrature data and shape function data
            # Again, we assume that the number of active dof on each element
            # is the same across elements
            quad_wts[i,j] = wt
            quad_pts[i,j,:] = transip.GetDataArray()
            shape_at_quad_pts[i,j,:] = shape.GetDataArray()


        # Check the updates for the local vector
        #print("local_vector =", local_vector.GetDataArray(), "\n")
        
        # Accumulate the local vector and the relevant values of the global vector 
        global_sub_vector = mfem.Vector()
        global_vector.GetSubVector(vdofs, global_sub_vector)
        global_sub_vector += local_vector

        # Set the values in the relevant entries of the global vector
        global_vector.SetSubVector(vdofs, global_sub_vector)
    
        # Check that the values are being set in the global vector
        #print("global_vector =", global_vector.GetDataArray(), "\n")



    # Test: Define my own linear form for the RHS based on the above function
    # This would test if the discrepancies are due to the 'FormLinearSystem' 
    # method, which performs additional manipulations
    my_b = mfem.LinearForm(fespace)

    # Don't use the MFEM assembly commands... use mine instead
    #my_b.AddDomainIntegrator(mfem.DomainLFIntegrator(one))
    #my_b.Assemble()

    # See if we can overwrite the data in the 'elemvect' attribute
    # This is essentially the same thing as what our loops above do
    my_b.Assign(global_vector)

    # 7. Define the solution vector x as a finite element grid function
    #   corresponding to fespace. Initialize x with initial guess of zero,
    #   which satisfies the boundary conditions.
    x = mfem.GridFunction(fespace)
    x.Assign(0.0)

    # 8. Set up the bilinear form a(.,.) on the finite element space
    #   corresponding to the Laplacian operator -Delta, by adding the Diffusion
    #   domain integrator.
    a = mfem.BilinearForm(fespace)
    if pa:
        a.SetAssemblyLevel(mfem.AssemblyLevel_PARTIAL)
    a.AddDomainIntegrator(mfem.DiffusionIntegrator(one))

    # 9. Assemble the bilinear form and the corresponding linear system,
    #   applying any necessary transformations such as: eliminating boundary
    #   conditions, applying conforming constraints for non-conforming AMR,
    #   static condensation, etc.
    if static_cond:
        a.EnableStaticCondensation()
    a.Assemble()

    A = mfem.OperatorPtr()
    B = mfem.Vector()
    X = mfem.Vector()

    a.FormLinearSystem(ess_tdof_list, x, b, A, X, B)
    print("Size of linear system: " + str(A.Height()))

    # Build the linear system with the updated linear form
    my_B = mfem.Vector()
    a.FormLinearSystem(ess_tdof_list, x, my_b, A, X, my_B)

    # Covert the MFEM Vectors to Numpy arrays for norms
    B_array = B.GetDataArray()
    my_B_array = my_B.GetDataArray()

    # Compare the output of the two RHS vectors
    print("B =", B_array,"\n")
    print("my_B =", my_B_array,"\n")

    # Relative error against the MFEM output in the 2-norm
    rel_err_1 = np.linalg.norm(my_B_array - B_array, 1)/np.linalg.norm(B_array, 1)
    rel_err_2 = np.linalg.norm(my_B_array - B_array, 2)/np.linalg.norm(B_array, 2)
    rel_err_inf = np.linalg.norm(my_B_array - B_array, np.inf)/np.linalg.norm(B_array, np.inf)

    print("Relative error in the rhs (1-norm):", rel_err_1, "\n")
    print("Relative error in the rhs (2-norm):", rel_err_2, "\n")
    print("Relative error in the rhs (inf-norm):", rel_err_inf, "\n")









    # 10. Solve
    #if pa:
    #    if mfem.UsesTensorBasis(fespace):
    #        M = mfem.OperatorJacobiSmoother(a, ess_tdof_list)
    #        mfem.PCG(A, M, B, X, 1, 4000, 1e-12, 0.0)
    #    else:
    #        mfem.CG(A, B, X, 1, 400, 1e-12, 0.0)
    #else:
        #AA = mfem.OperatorHandle2SparseMatrix(A)
    #    AA = A.AsSparseMatrix()
    #    M = mfem.GSSmoother(AA)
    #    mfem.PCG(A, M, B, X, 1, 200, 1e-12, 0.0)

    # 11. Recover the solution as a finite element grid function.
    #a.RecoverFEMSolution(X, b, x)

    # 12. Save the refined mesh and the solution. This output can be viewed later
    #     using GLVis: "glvis -m refined.mesh -g sol.gf".
    #mesh.Print('refined.mesh', 8)
    #x.Save('sol.gf', 8)

    # 13. Send the solution by socket to a GLVis server.
    #if (visualization):
    #    sol_sock = mfem.socketstream("localhost", 19916)
    #    sol_sock.precision(8)
    #    sol_sock.send_solution(mesh, x)


if __name__ == "__main__":
    from mfem.common.arg_parser import ArgParser

    parser = ArgParser(description='Ex1 (Laplace Problem)')
    parser.add_argument('-m', '--mesh',
                        default='star.mesh',
                        action='store', type=str,
                        help='Mesh file to use.')
    parser.add_argument('-vis', '--visualization',
                        action='store_true',
                        help='Enable GLVis visualization')
    parser.add_argument('-o', '--order',
                        action='store', default=1, type=int,
                        help="Finite element order (polynomial degree) or -1 for isoparametric space.")
    parser.add_argument('-sc', '--static-condensation',
                        action='store_true',
                        help="Enable static condensation.")
    parser.add_argument("-pa", "--partial-assembly",
                        action='store_true',
                        help="Enable Partial Assembly.")
    parser.add_argument("-d", "--device",
                        default="cpu", type=str,
                        help="Device configuration string, see Device::Configure().")

    args = parser.parse_args()
    parser.print_options(args)

    order = args.order
    static_cond = args.static_condensation

    meshfile = expanduser(
        join(os.path.dirname(__file__), '../../', 'data', args.mesh))
    visualization = args.visualization
    device = args.device
    pa = args.partial_assembly

    run(order=order,
        static_cond=static_cond,
        meshfile=meshfile,
        visualization=visualization,
        device=device,
        pa=pa)







