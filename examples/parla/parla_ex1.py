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

# Import the MFEM Python modules
import mfem.ser as mfem
from mfem.common.arg_parser import ArgParser

# Parla modules with associated decorators
from parla import Parla

# Import for placing tasks on the cpu
from parla.cpu import cpu

# Import for the spawn decorator and TaskSpace declaration
from parla.tasks import spawn, TaskSpace


def run(order=1, static_cond=False,
        meshfile='', visualization=False,
        device='cpu', pa=False):
    '''
    run ex1
    '''

    device = mfem.Device(device)
    device.Print()
   
    print("\nPreparing to launch the main task...\n")

    mesh = mfem.Mesh(meshfile, 1, 1)
    dim = mesh.Dimension()
  
    # Define the main task for parla
    @spawn(placement=cpu)
    async def main_task():

        # First, create the task space
        taskSpace = TaskSpace('SimpleTaskSpace')

        # Create a dictionary for MFEM objects to expand their scope
        # In each task, such methods will be declared as "nonlocal".
        # Our strategy is to create the dictionary of objects and gradually
        # add entries as we process tasks.
        pymfem_obj = {"dim": dim, "order": order, "mesh": mesh}

        # No dependencies here
        @spawn(taskid=taskSpace[0])
        def task1():
 
            # Retreive any nonlocal objects required for this task
            nonlocal pymfem_obj
            dim = pymfem_obj["dim"]
            mesh = pymfem_obj["mesh"]
   
            # 1. Refine the mesh to increase the resolution. In this example we do
            #    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
            #    largest number that gives a final mesh with no more than 50,000
            #    elements.

            ref_levels = int(
                np.floor(
                    np.log(
                        50000. /
                        mesh.GetNE()) /
                    np.log(2.) /
                    dim))
            for x in range(ref_levels):
                mesh.UniformRefinement()

            print("Completed task 1...\n")
      
            # Return the awaitable
            return taskSpace[0]

        # This task depends on task 1
        @spawn(taskid=taskSpace[1], dependencies=[taskSpace[0]])
        def task2():

            # Retreive any nonlocal objects required for this task
            nonlocal pymfem_obj
            dim = pymfem_obj["dim"]
            order = pymfem_obj["order"]
            mesh = pymfem_obj["mesh"]

            # 2. Define a finite element space on the mesh. Here we use vector finite
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
                   str(fespace.GetTrueVSize()), "\n")
        
            print("Completed task 2...\n")

            # Add new element(s) to our dictionary
            pymfem_obj["fespace"] = fespace

            # Return the awaitable
            return taskSpace[1]

        # This task depends on task 2
        @spawn(taskid=taskSpace[2], dependencies=[taskSpace[1]])
        def task3():

            # Retreive any nonlocal objects required for this task
            nonlocal pymfem_obj            
            mesh = pymfem_obj["mesh"]
            fespace = pymfem_obj["fespace"]

            # 3. Determine the list of true (i.e. conforming) essential boundary dofs.
            #    In this example, the boundary conditions are defined by marking all
            #    the boundary attributes from the mesh as essential (Dirichlet) and
            #    converting them to a list of true dofs.
            ess_tdof_list = mfem.intArray()
            if mesh.bdr_attributes.Size() > 0:
                ess_bdr = mfem.intArray([1] * mesh.bdr_attributes.Max())
                ess_bdr = mfem.intArray(mesh.bdr_attributes.Max())
                ess_bdr.Assign(1)
                fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)
         
            # Add new element(s) to our dictionary
            pymfem_obj["ess_tdof_list"] = ess_tdof_list

            print("Completed task 3...\n")
      
            # Return the awaitable
            return taskSpace[2]
    
        # This task depends on task 3
        @spawn(taskid=taskSpace[3], dependencies=[taskSpace[2]])
        def task4():

            # Retreive any nonlocal objects required for this task
            nonlocal pymfem_obj
            fespace = pymfem_obj["fespace"]
   
            # 4. Set up the linear form b(.) which corresponds to the right-hand side of
            #   the FEM linear system, which in this case is (1,phi_i) where phi_i are
            #   the basis functions in the finite element fespace.
            b = mfem.LinearForm(fespace)
            one = mfem.ConstantCoefficient(1.0)
            b.AddDomainIntegrator(mfem.DomainLFIntegrator(one))
            b.Assemble()

            # Add new element(s) to our dictionary
            pymfem_obj["b"] = b
            pymfem_obj["one"] = one # This is our RHS function, i.e., f(x,y) = 1.0

            print("Completed task 4...\n")
      
            # Return the awaitable
            return taskSpace[3]

        # This task depends on task 4
        @spawn(taskid=taskSpace[4], dependencies=[taskSpace[3]])
        def task5():

            # Retreive any nonlocal objects required for this task
            nonlocal pymfem_obj
            fespace = pymfem_obj["fespace"]
         
            # 5. Define the solution vector x as a finite element grid function
            #   corresponding to fespace. Initialize x with initial guess of zero,
            #   which satisfies the boundary conditions.
            x = mfem.GridFunction(fespace)
            x.Assign(0.0)
    
            # Add new element(s) to our dictionary
            pymfem_obj["x"] = x

            print("Completed task 5...\n")
    
            # Return the awaitable
            return taskSpace[4]

        # This task depends on task 5
        @spawn(taskid=taskSpace[5], dependencies=[taskSpace[4]])
        def task6():

            # Retreive any nonlocal objects required for this task
            nonlocal pymfem_obj, pa
            fespace = pymfem_obj["fespace"]
            one = pymfem_obj["one"]

            # 6. Set up the bilinear form a(.,.) on the finite element space
            #   corresponding to the Laplacian operator -Delta, by adding the Diffusion
            #   domain integrator.
            a = mfem.BilinearForm(fespace)
            if pa:
                a.SetAssemblyLevel(mfem.AssemblyLevel_PARTIAL)
            a.AddDomainIntegrator(mfem.DiffusionIntegrator(one))

            # Add new element(s) to our dictionary
            pymfem_obj["a"] = a

            print("Completed task 6...\n")
    
            # Return the awaitable
            return taskSpace[5]

        # This task depends on task 6
        @spawn(taskid=taskSpace[6], dependencies=[taskSpace[5]])
        def task7():

            # Retreive any nonlocal objects required for this task
            nonlocal pymfem_obj, static_cond
            a = pymfem_obj["a"]
            b = pymfem_obj["b"]
            x = pymfem_obj["x"]
            ess_tdof_list = pymfem_obj["ess_tdof_list"]
         
            # 7. Assemble the bilinear form and the corresponding linear system,
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
         
            ## Check that the numpy interface works
            #print("A as a sparse matrix:", A.AsSparseMatrix(),"\n")
            #print("B as a numpy array:", B.GetDataArray(),"\n")
    
            # Add new element(s) to our dictionary
            pymfem_obj["A"] = A
            pymfem_obj["B"] = B
            pymfem_obj["X"] = X

            print("Completed task 7...\n")
    
            # Return the awaitable
            return taskSpace[6]

        # This task depends on task 7
        @spawn(taskid=taskSpace[7], dependencies=[taskSpace[6]])
        def task8():

            # Retreive any nonlocal objects required for this task
            nonlocal pymfem_obj, pa
            fespace = pymfem_obj["fespace"]
            a = pymfem_obj["a"]
            ess_tdof_list = pymfem_obj["ess_tdof_list"]
            A = pymfem_obj["A"]
            B = pymfem_obj["B"]
            X = pymfem_obj["X"]

            # 8. Solve
         
            if pa:
                if mfem.UsesTensorBasis(fespace):
                    M = mfem.OperatorJacobiSmoother(a, ess_tdof_list)
                    mfem.PCG(A, M, B, X, 1, 4000, 1e-12, 0.0)
                else:
                    mfem.CG(A, B, X, 1, 400, 1e-12, 0.0)
            else:
                #AA = mfem.OperatorHandle2SparseMatrix(A)
                AA = A.AsSparseMatrix()
                M = mfem.GSSmoother(AA)
                mfem.PCG(A, M, B, X, 1, 200, 1e-12, 0.0)
    
            print("\nCompleted task 8...\n")
    
            # Return the awaitable
            return taskSpace[7]

        # This task depends on task 8
        @spawn(taskid=taskSpace[8], dependencies=[taskSpace[7]])
        def task9():  
         
            # Retreive any nonlocal objects required for this task
            nonlocal pymfem_obj
            a = pymfem_obj["a"]
            X = pymfem_obj["X"]
            b = pymfem_obj["b"]
            x = pymfem_obj["x"]
            

            # 9. Recover the solution as a finite element grid function.
            a.RecoverFEMSolution(X, b, x)
    
            print("Completed task 9...\n")
    
            # Return the awaitable
            return taskSpace[8]

        # This task depends on task 9
        @spawn(taskid=taskSpace[9], dependencies=[taskSpace[8]])
        def task10():

            # Retreive any nonlocal objects required for this task
            nonlocal pymfem_obj
            mesh = pymfem_obj["mesh"]
            x = pymfem_obj["x"]
      
            # 10. Save the refined mesh and the solution. This output can be viewed later
            #     using GLVis: "glvis -m refined.mesh -g sol.gf".
            mesh.Print('refined.mesh', 8)
            x.Save('sol.gf', 8)
    
            print("Completed task 10...\n")
    
            # Return the awaitable
            return taskSpace[9]

        # This task depends on task 10
        @spawn(taskid=taskSpace[10], dependencies=[taskSpace[9]])
        def task11():  

            nonlocal pymfem_obj, visualization
            mesh = pymfem_obj["mesh"]
            x = pymfem_obj["x"]
              
            # 11. Send the solution by socket to a GLVis server.
            if (visualization):
                sol_sock = mfem.socketstream("localhost", 19916)
                sol_sock.precision(8)
                sol_sock.send_solution(mesh, x)
         
            print("Completed task 11...\n")

            print("Done!\n")

            # Return the awaitable
            return taskSpace[10]
    

# Execute the Parla program with the Parla Context
if __name__ == '__main__':

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
    
    meshfile = expanduser(join(os.path.dirname(__file__), '../../', 'data', args.mesh))
    visualization = args.visualization
    device = args.device
    pa = args.partial_assembly
    
    # Need to create the Parla context here before spawning tasks
    with Parla():

        run(order=order,
            static_cond=static_cond,
            meshfile=meshfile,
            visualization=visualization,
            device=device,
            pa=pa)
  





