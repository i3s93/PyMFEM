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

# Import any of the MFEM Python modules
import mfem.ser as mfem
from mfem.common.arg_parser import ArgParser


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
    ref_levels = int(np.floor(np.log(50000./mesh.GetNE())/np.log(2.)/dim))

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
          str(fespace.GetTrueVSize()),"\n")

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

    #-----------------------------------------------------------------------------
    # Version with blocking 
    #-----------------------------------------------------------------------------

    # Create a dictionary for MFEM objects to expand their scope
    # This will gradually get larger as we'll add objects to it
    
    # Specify a partition of the (global) list of elements
    num_blocks = 4 # How many blocks do you want to use? 
    elements_per_block = mesh.GetNE()//num_blocks # Block size
    leftover_blocks = mesh.GetNE() % num_blocks
    
    # Adjust the number of elements if the block size doesn't divide the elements evenly
    block_sizes = elements_per_block*np.ones([num_blocks], dtype=np.int64)
    block_sizes[0:leftover_blocks] += 1
    
    print("Number of blocks: " + str(num_blocks))
    print("Number of left-over blocks: " + str(leftover_blocks))
    print("Elements per block:", block_sizes,"\n")

    # Set the number of DoF according to the FE space
    num_dof = fespace.GetNDofs()
  
    # Create an array to hold the global data for the RHS
    # This is just the linear form
    global_array = np.zeros([num_dof])

    # To use the blocking scheme, we'll
    # also use another set of arrays that
    # hold partial sums of this global array
    block_global_array = np.zeros([num_blocks, num_dof])

    # Next we loop over each block which partitions the element indices
    # Each block will form a task, so we can control granularity by choosing
    # the number of blocks carefully. In most cases, this will be the number of
    # devices. It may be possible to use more blocks than devices if we use a
    # round-robin style of mapping to devices
    for i in range(num_blocks):

        # Need the offset for the element indices owned by this block
        # This is the sum of all block sizes that came before it
        s_idx = np.sum(block_sizes[:i])
        e_idx = s_idx + block_sizes[i]

        # Next, loop over the mesh elements on this block and perform quadrature evaluations
        for j in range(s_idx, e_idx):

            # Get the particular element
            element = fespace.GetFE(j)
            dof = element.GetDof()
    
            # Get the indices for the DoF on this element
            # This tells us which entries we write to in the global array
            vdofs = fespace.GetElementVDofs(j)
            vdofs = np.asarray(vdofs) # Convert the list to a Numpy array
    
            # Get the element's transformation, which will maps the ir's reference points
            Tr = mesh.GetElementTransformation(j)
            intorder = 2*order
            ir = mfem.IntRules.Get(element.GetGeomType(), intorder)
    
            # Storage for the basis functions and local dof on this element
            # This should always be smaller than the size of the global array
            shape = mfem.Vector(np.zeros([dof]))
            local_array = np.zeros([dof])
    
            for k in range(ir.GetNPoints()):
    
                # Get the integration point from the rule
                ip = ir.IntPoint(k)
    
                # Set an integration point in the element transformation
                Tr.SetIntPoint(ip)
    
                # Transform the reference integration point to a physical location
                transip = mfem.Vector(np.zeros([3]))
                Tr.Transform(ip, transip)
    
                # Next, evaluate all the basis functions at this integration point
                element.CalcPhysShape(Tr, shape)
    
                # Compute the adjusted quadrature weight (volume factors)
                wt = ip.weight*Tr.Weight()
    
                # Store the contributions of the shape functions in the local array
                local_array += wt*shape.GetDataArray()

            # Accumulate the local array into the relevant entries of the block-wise global vector
            block_global_array[i,vdofs] += local_array[:]

    print("(Before sum) block global array = ", block_global_array[:,:10], "\n")

    # Perform the reduction across the blocks and store in the global_array
    global_array = np.sum(block_global_array, axis=0)

    print("(After sum) block global array = ", block_global_array[:,:10], "\n")

    print("global array = ", global_array[:10], "\n")

    # Define my own linear form for the RHS based on the above function
    # The 'FormLinearSystem' method, which performs additional manipulations
    # that simplify the RHS for the resulting linear system
    my_b = mfem.LinearForm(fespace)
    my_b.Assign(global_array)

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
    print("Size of linear system: " + str(A.Height()),"\n")

    # Build the linear system with my linear form
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

    print("Relative error in the rhs (1-norm):", rel_err_1)
    print("Relative error in the rhs (2-norm):", rel_err_2)
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







