# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.2
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _pgridfunc
else:
    import _pgridfunc

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

_swig_new_instance_method = _pgridfunc.SWIG_PyInstanceMethod_New
_swig_new_static_method = _pgridfunc.SWIG_PyStaticMethod_New

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


import weakref

MFEM_VERSION = _pgridfunc.MFEM_VERSION

MFEM_VERSION_STRING = _pgridfunc.MFEM_VERSION_STRING

MFEM_VERSION_TYPE = _pgridfunc.MFEM_VERSION_TYPE

MFEM_VERSION_TYPE_RELEASE = _pgridfunc.MFEM_VERSION_TYPE_RELEASE

MFEM_VERSION_TYPE_DEVELOPMENT = _pgridfunc.MFEM_VERSION_TYPE_DEVELOPMENT

MFEM_VERSION_MAJOR = _pgridfunc.MFEM_VERSION_MAJOR

MFEM_VERSION_MINOR = _pgridfunc.MFEM_VERSION_MINOR

MFEM_VERSION_PATCH = _pgridfunc.MFEM_VERSION_PATCH

import mfem._par.pfespace
import mfem._par.operators
import mfem._par.mem_manager
import mfem._par.vector
import mfem._par.array
import mfem._par.fespace
import mfem._par.coefficient
import mfem._par.globals
import mfem._par.matrix
import mfem._par.symmat
import mfem._par.intrules
import mfem._par.sparsemat
import mfem._par.densemat
import mfem._par.eltrans
import mfem._par.fe
import mfem._par.geom
import mfem._par.fe_base
import mfem._par.fe_fixed_order
import mfem._par.element
import mfem._par.table
import mfem._par.hash
import mfem._par.fe_h1
import mfem._par.fe_nd
import mfem._par.fe_rt
import mfem._par.fe_l2
import mfem._par.fe_nurbs
import mfem._par.fe_pos
import mfem._par.fe_ser
import mfem._par.mesh
import mfem._par.sort_pairs
import mfem._par.ncmesh
import mfem._par.vtk
import mfem._par.vertex
import mfem._par.gridfunc
import mfem._par.bilininteg
import mfem._par.fe_coll
import mfem._par.lininteg
import mfem._par.linearform
import mfem._par.nonlininteg
import mfem._par.std_vectors
import mfem._par.doftrans
import mfem._par.handle
import mfem._par.hypre
import mfem._par.restriction
import mfem._par.pmesh
import mfem._par.pncmesh
import mfem._par.communication
import mfem._par.sets

def GlobalLpNorm(p, loc_norm, comm):
    r"""GlobalLpNorm(double const p, double loc_norm, MPI_Comm comm) -> double"""
    return _pgridfunc.GlobalLpNorm(p, loc_norm, comm)
GlobalLpNorm = _pgridfunc.GlobalLpNorm
class ParGridFunction(mfem._par.gridfunc.GridFunction):
    r"""Proxy of C++ mfem::ParGridFunction class."""

    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def ParFESpace(self):
        r"""ParFESpace(ParGridFunction self) -> ParFiniteElementSpace"""
        return _pgridfunc.ParGridFunction_ParFESpace(self)
    ParFESpace = _swig_new_instance_method(_pgridfunc.ParGridFunction_ParFESpace)

    def Update(self):
        r"""Update(ParGridFunction self)"""
        return _pgridfunc.ParGridFunction_Update(self)
    Update = _swig_new_instance_method(_pgridfunc.ParGridFunction_Update)

    def SetSpace(self, *args):
        r"""
        SetSpace(ParGridFunction self, FiniteElementSpace f)
        SetSpace(ParGridFunction self, ParFiniteElementSpace f)
        """
        return _pgridfunc.ParGridFunction_SetSpace(self, *args)
    SetSpace = _swig_new_instance_method(_pgridfunc.ParGridFunction_SetSpace)

    def MakeRef(self, *args):
        r"""
        MakeRef(ParGridFunction self, Vector base, int offset, int size)
        MakeRef(ParGridFunction self, Vector base, int offset)
        MakeRef(ParGridFunction self, FiniteElementSpace f, double * v)
        MakeRef(ParGridFunction self, ParFiniteElementSpace f, double * v)
        MakeRef(ParGridFunction self, FiniteElementSpace f, Vector v, int v_offset)
        MakeRef(ParGridFunction self, ParFiniteElementSpace f, Vector v, int v_offset)
        """
        return _pgridfunc.ParGridFunction_MakeRef(self, *args)
    MakeRef = _swig_new_instance_method(_pgridfunc.ParGridFunction_MakeRef)

    def Distribute(self, *args):
        r"""
        Distribute(ParGridFunction self, Vector tv)
        Distribute(ParGridFunction self, Vector tv)
        """
        return _pgridfunc.ParGridFunction_Distribute(self, *args)
    Distribute = _swig_new_instance_method(_pgridfunc.ParGridFunction_Distribute)

    def AddDistribute(self, *args):
        r"""
        AddDistribute(ParGridFunction self, double a, Vector tv)
        AddDistribute(ParGridFunction self, double a, Vector tv)
        """
        return _pgridfunc.ParGridFunction_AddDistribute(self, *args)
    AddDistribute = _swig_new_instance_method(_pgridfunc.ParGridFunction_AddDistribute)

    def SetFromTrueDofs(self, tv):
        r"""SetFromTrueDofs(ParGridFunction self, Vector tv)"""
        return _pgridfunc.ParGridFunction_SetFromTrueDofs(self, tv)
    SetFromTrueDofs = _swig_new_instance_method(_pgridfunc.ParGridFunction_SetFromTrueDofs)

    def GetTrueDofs(self, *args):
        r"""
        GetTrueDofs(ParGridFunction self, Vector tv)
        GetTrueDofs(ParGridFunction self) -> HypreParVector
        """
        return _pgridfunc.ParGridFunction_GetTrueDofs(self, *args)
    GetTrueDofs = _swig_new_instance_method(_pgridfunc.ParGridFunction_GetTrueDofs)

    def ParallelAverage(self, *args):
        r"""
        ParallelAverage(ParGridFunction self, Vector tv)
        ParallelAverage(ParGridFunction self, HypreParVector tv)
        ParallelAverage(ParGridFunction self) -> HypreParVector
        """
        return _pgridfunc.ParGridFunction_ParallelAverage(self, *args)
    ParallelAverage = _swig_new_instance_method(_pgridfunc.ParGridFunction_ParallelAverage)

    def ParallelProject(self, *args):
        r"""
        ParallelProject(ParGridFunction self, Vector tv)
        ParallelProject(ParGridFunction self, HypreParVector tv)
        ParallelProject(ParGridFunction self) -> HypreParVector
        """
        return _pgridfunc.ParGridFunction_ParallelProject(self, *args)
    ParallelProject = _swig_new_instance_method(_pgridfunc.ParGridFunction_ParallelProject)

    def ParallelAssemble(self, *args):
        r"""
        ParallelAssemble(ParGridFunction self, Vector tv)
        ParallelAssemble(ParGridFunction self, HypreParVector tv)
        ParallelAssemble(ParGridFunction self) -> HypreParVector
        """
        return _pgridfunc.ParGridFunction_ParallelAssemble(self, *args)
    ParallelAssemble = _swig_new_instance_method(_pgridfunc.ParGridFunction_ParallelAssemble)

    def ExchangeFaceNbrData(self):
        r"""ExchangeFaceNbrData(ParGridFunction self)"""
        return _pgridfunc.ParGridFunction_ExchangeFaceNbrData(self)
    ExchangeFaceNbrData = _swig_new_instance_method(_pgridfunc.ParGridFunction_ExchangeFaceNbrData)

    def FaceNbrData(self, *args):
        r"""
        FaceNbrData(ParGridFunction self) -> Vector
        FaceNbrData(ParGridFunction self) -> Vector
        """
        return _pgridfunc.ParGridFunction_FaceNbrData(self, *args)
    FaceNbrData = _swig_new_instance_method(_pgridfunc.ParGridFunction_FaceNbrData)

    def GetValue(self, *args):
        r"""
        GetValue(ParGridFunction self, int i, IntegrationPoint ip, int vdim=1) -> double
        GetValue(ParGridFunction self, ElementTransformation T) -> double
        GetValue(ParGridFunction self, ElementTransformation T, IntegrationPoint ip, int comp=0, Vector tr=None) -> double
        """
        return _pgridfunc.ParGridFunction_GetValue(self, *args)
    GetValue = _swig_new_instance_method(_pgridfunc.ParGridFunction_GetValue)

    def GetVectorValue(self, *args):
        r"""
        GetVectorValue(ParGridFunction self, int i, IntegrationPoint ip, Vector val)
        GetVectorValue(ParGridFunction self, ElementTransformation T, IntegrationPoint ip, Vector val, Vector tr=None)
        """
        return _pgridfunc.ParGridFunction_GetVectorValue(self, *args)
    GetVectorValue = _swig_new_instance_method(_pgridfunc.ParGridFunction_GetVectorValue)

    def GetDerivative(self, comp, der_comp, der):
        r"""GetDerivative(ParGridFunction self, int comp, int der_comp, ParGridFunction der)"""
        return _pgridfunc.ParGridFunction_GetDerivative(self, comp, der_comp, der)
    GetDerivative = _swig_new_instance_method(_pgridfunc.ParGridFunction_GetDerivative)

    def GetElementDofValues(self, el, dof_vals):
        r"""GetElementDofValues(ParGridFunction self, int el, Vector dof_vals)"""
        return _pgridfunc.ParGridFunction_GetElementDofValues(self, el, dof_vals)
    GetElementDofValues = _swig_new_instance_method(_pgridfunc.ParGridFunction_GetElementDofValues)

    def ProjectCoefficient(self, *args):
        r"""
        ProjectCoefficient(ParGridFunction self, Coefficient coeff)
        ProjectCoefficient(ParGridFunction self, Coefficient coeff, intArray dofs, int vd=0)
        ProjectCoefficient(ParGridFunction self, VectorCoefficient vcoeff)
        ProjectCoefficient(ParGridFunction self, VectorCoefficient vcoeff, intArray dofs)
        ProjectCoefficient(ParGridFunction self, VectorCoefficient vcoeff, int attribute)
        ProjectCoefficient(ParGridFunction self, mfem::Coefficient *[] coeff)
        ProjectCoefficient(ParGridFunction self, Coefficient coeff)
        """
        return _pgridfunc.ParGridFunction_ProjectCoefficient(self, *args)
    ProjectCoefficient = _swig_new_instance_method(_pgridfunc.ParGridFunction_ProjectCoefficient)

    def ProjectDiscCoefficient(self, *args):
        r"""
        ProjectDiscCoefficient(ParGridFunction self, VectorCoefficient coeff, intArray dof_attr)
        ProjectDiscCoefficient(ParGridFunction self, VectorCoefficient coeff)
        ProjectDiscCoefficient(ParGridFunction self, Coefficient coeff, mfem::GridFunction::AvgType type)
        ProjectDiscCoefficient(ParGridFunction self, VectorCoefficient coeff, mfem::GridFunction::AvgType type)
        ProjectDiscCoefficient(ParGridFunction self, VectorCoefficient coeff)
        ProjectDiscCoefficient(ParGridFunction self, Coefficient coeff, mfem::GridFunction::AvgType type)
        ProjectDiscCoefficient(ParGridFunction self, VectorCoefficient vcoeff, mfem::GridFunction::AvgType type)
        """
        return _pgridfunc.ParGridFunction_ProjectDiscCoefficient(self, *args)
    ProjectDiscCoefficient = _swig_new_instance_method(_pgridfunc.ParGridFunction_ProjectDiscCoefficient)

    def ProjectBdrCoefficient(self, *args):
        r"""
        ProjectBdrCoefficient(ParGridFunction self, Coefficient coeff, intArray attr)
        ProjectBdrCoefficient(ParGridFunction self, VectorCoefficient vcoeff, intArray attr)
        ProjectBdrCoefficient(ParGridFunction self, mfem::Coefficient *[] coeff, intArray attr)
        ProjectBdrCoefficient(ParGridFunction self, VectorCoefficient vcoeff, intArray attr)
        ProjectBdrCoefficient(ParGridFunction self, mfem::Coefficient *[] coeff, intArray attr)
        """
        return _pgridfunc.ParGridFunction_ProjectBdrCoefficient(self, *args)
    ProjectBdrCoefficient = _swig_new_instance_method(_pgridfunc.ParGridFunction_ProjectBdrCoefficient)

    def ProjectBdrCoefficientTangent(self, vcoeff, bdr_attr):
        r"""ProjectBdrCoefficientTangent(ParGridFunction self, VectorCoefficient vcoeff, intArray bdr_attr)"""
        return _pgridfunc.ParGridFunction_ProjectBdrCoefficientTangent(self, vcoeff, bdr_attr)
    ProjectBdrCoefficientTangent = _swig_new_instance_method(_pgridfunc.ParGridFunction_ProjectBdrCoefficientTangent)

    def ComputeL1Error(self, *args):
        r"""
        ComputeL1Error(ParGridFunction self, mfem::Coefficient *[] exsol, mfem::IntegrationRule const *[] irs=0) -> double
        ComputeL1Error(ParGridFunction self, Coefficient exsol, mfem::IntegrationRule const *[] irs=0) -> double
        ComputeL1Error(ParGridFunction self, VectorCoefficient exsol, mfem::IntegrationRule const *[] irs=0) -> double
        """
        return _pgridfunc.ParGridFunction_ComputeL1Error(self, *args)
    ComputeL1Error = _swig_new_instance_method(_pgridfunc.ParGridFunction_ComputeL1Error)

    def ComputeL2Error(self, *args):
        r"""
        ComputeL2Error(ParGridFunction self, mfem::Coefficient *[] exsol, mfem::IntegrationRule const *[] irs=0) -> double
        ComputeL2Error(ParGridFunction self, Coefficient exsol, mfem::IntegrationRule const *[] irs=0) -> double
        ComputeL2Error(ParGridFunction self, VectorCoefficient exsol, mfem::IntegrationRule const *[] irs=0, intArray elems=None) -> double
        """
        return _pgridfunc.ParGridFunction_ComputeL2Error(self, *args)
    ComputeL2Error = _swig_new_instance_method(_pgridfunc.ParGridFunction_ComputeL2Error)

    def ComputeGradError(self, exgrad, irs=0):
        r"""ComputeGradError(ParGridFunction self, VectorCoefficient exgrad, mfem::IntegrationRule const *[] irs=0) -> double"""
        return _pgridfunc.ParGridFunction_ComputeGradError(self, exgrad, irs)
    ComputeGradError = _swig_new_instance_method(_pgridfunc.ParGridFunction_ComputeGradError)

    def ComputeCurlError(self, excurl, irs=0):
        r"""ComputeCurlError(ParGridFunction self, VectorCoefficient excurl, mfem::IntegrationRule const *[] irs=0) -> double"""
        return _pgridfunc.ParGridFunction_ComputeCurlError(self, excurl, irs)
    ComputeCurlError = _swig_new_instance_method(_pgridfunc.ParGridFunction_ComputeCurlError)

    def ComputeDivError(self, exdiv, irs=0):
        r"""ComputeDivError(ParGridFunction self, Coefficient exdiv, mfem::IntegrationRule const *[] irs=0) -> double"""
        return _pgridfunc.ParGridFunction_ComputeDivError(self, exdiv, irs)
    ComputeDivError = _swig_new_instance_method(_pgridfunc.ParGridFunction_ComputeDivError)

    def ComputeDGFaceJumpError(self, exsol, ell_coeff, jump_scaling, irs=0):
        r"""ComputeDGFaceJumpError(ParGridFunction self, Coefficient exsol, Coefficient ell_coeff, JumpScaling jump_scaling, mfem::IntegrationRule const *[] irs=0) -> double"""
        return _pgridfunc.ParGridFunction_ComputeDGFaceJumpError(self, exsol, ell_coeff, jump_scaling, irs)
    ComputeDGFaceJumpError = _swig_new_instance_method(_pgridfunc.ParGridFunction_ComputeDGFaceJumpError)

    def ComputeH1Error(self, *args):
        r"""
        ComputeH1Error(ParGridFunction self, Coefficient exsol, VectorCoefficient exgrad, Coefficient ell_coef, double Nu, int norm_type) -> double
        ComputeH1Error(ParGridFunction self, Coefficient exsol, VectorCoefficient exgrad, mfem::IntegrationRule const *[] irs=0) -> double
        """
        return _pgridfunc.ParGridFunction_ComputeH1Error(self, *args)
    ComputeH1Error = _swig_new_instance_method(_pgridfunc.ParGridFunction_ComputeH1Error)

    def ComputeHDivError(self, exsol, exdiv, irs=0):
        r"""ComputeHDivError(ParGridFunction self, VectorCoefficient exsol, Coefficient exdiv, mfem::IntegrationRule const *[] irs=0) -> double"""
        return _pgridfunc.ParGridFunction_ComputeHDivError(self, exsol, exdiv, irs)
    ComputeHDivError = _swig_new_instance_method(_pgridfunc.ParGridFunction_ComputeHDivError)

    def ComputeHCurlError(self, exsol, excurl, irs=0):
        r"""ComputeHCurlError(ParGridFunction self, VectorCoefficient exsol, VectorCoefficient excurl, mfem::IntegrationRule const *[] irs=0) -> double"""
        return _pgridfunc.ParGridFunction_ComputeHCurlError(self, exsol, excurl, irs)
    ComputeHCurlError = _swig_new_instance_method(_pgridfunc.ParGridFunction_ComputeHCurlError)

    def ComputeMaxError(self, *args):
        r"""
        ComputeMaxError(ParGridFunction self, mfem::Coefficient *[] exsol, mfem::IntegrationRule const *[] irs=0) -> double
        ComputeMaxError(ParGridFunction self, Coefficient exsol, mfem::IntegrationRule const *[] irs=0) -> double
        ComputeMaxError(ParGridFunction self, VectorCoefficient exsol, mfem::IntegrationRule const *[] irs=0) -> double
        """
        return _pgridfunc.ParGridFunction_ComputeMaxError(self, *args)
    ComputeMaxError = _swig_new_instance_method(_pgridfunc.ParGridFunction_ComputeMaxError)

    def ComputeLpError(self, *args):
        r"""
        ComputeLpError(ParGridFunction self, double const p, Coefficient exsol, Coefficient weight=None, mfem::IntegrationRule const *[] irs=0) -> double
        ComputeLpError(ParGridFunction self, double const p, VectorCoefficient exsol, Coefficient weight=None, VectorCoefficient v_weight=None, mfem::IntegrationRule const *[] irs=0) -> double
        """
        return _pgridfunc.ParGridFunction_ComputeLpError(self, *args)
    ComputeLpError = _swig_new_instance_method(_pgridfunc.ParGridFunction_ComputeLpError)

    def ComputeFlux(self, blfi, flux, wcoef=True, subdomain=-1):
        r"""ComputeFlux(ParGridFunction self, BilinearFormIntegrator blfi, GridFunction flux, bool wcoef=True, int subdomain=-1)"""
        return _pgridfunc.ParGridFunction_ComputeFlux(self, blfi, flux, wcoef, subdomain)
    ComputeFlux = _swig_new_instance_method(_pgridfunc.ParGridFunction_ComputeFlux)
    __swig_destroy__ = _pgridfunc.delete_ParGridFunction

    def __init__(self, *args):
        r"""
        __init__(ParGridFunction self) -> ParGridFunction
        __init__(ParGridFunction self, ParGridFunction orig) -> ParGridFunction
        __init__(ParGridFunction self, ParFiniteElementSpace pf) -> ParGridFunction
        __init__(ParGridFunction self, ParFiniteElementSpace pf, double * data) -> ParGridFunction
        __init__(ParGridFunction self, ParFiniteElementSpace pf, Vector base, int base_offset=0) -> ParGridFunction
        __init__(ParGridFunction self, ParFiniteElementSpace pf, GridFunction gf) -> ParGridFunction
        __init__(ParGridFunction self, ParFiniteElementSpace pf, HypreParVector tv) -> ParGridFunction
        __init__(ParGridFunction self, ParMesh pmesh, GridFunction gf, int const * partitioning=None) -> ParGridFunction
        __init__(ParGridFunction self, ParMesh pmesh, std::istream & input) -> ParGridFunction
        __init__(ParGridFunction self, ParFiniteElementSpace fes, Vector v, int offset) -> ParGridFunction
        """

        from mfem._par.pmesh import ParMesh
        from mfem._par.pfespace import ParFiniteElementSpace
        from mfem._par.gridfunc import GridFunction
        if (len(args) == 2 and isinstance(args[1], str) and
             isinstance(args[0], ParMesh)):
            g0 = GridFunction(args[0], args[1])
            fec = g0.OwnFEC()
            fes = g0.FESpace()
            pfes = ParFiniteElementSpace(args[0], fec, fes.GetVDim(),
                                              fes.GetOrdering())
            x = ParGridFunction(pfes, g0)
            x.thisown = 0
            pfes.thisown = 0
            g0.thisown = 0
            self.this = x.this
            return 


        _pgridfunc.ParGridFunction_swiginit(self, _pgridfunc.new_ParGridFunction(*args))

    def Assign(self, *args):
        r"""
        Assign(ParGridFunction self, HypreParVector v)
        Assign(ParGridFunction self, ParGridFunction v)
        Assign(ParGridFunction self, double const v)
        Assign(ParGridFunction self, Vector v)
        Assign(ParGridFunction self, PyObject * param)
        """

        from numpy import ndarray, ascontiguousarray, array
        keep_link = False
        if len(args) == 1:
            if isinstance(args[0], ndarray):
                if args[0].dtype != 'float64':
                     raise ValueError('Must be float64 array ' + str(args[0].dtype) +
        			      ' is given')
                elif args[0].ndim != 1:
                    raise ValueError('Ndim must be one') 
                elif args[0].shape[0] != self.Size():
                    raise ValueError('Length does not match')
                else:
                    args = (ascontiguousarray(args[0]),)
            elif isinstance(args[0], tuple):
                args = (array(args[0], dtype = float),)      
            elif isinstance(args[0], list):	      
                args = (array(args[0], dtype = float),)      
            else:
                pass


        val = _pgridfunc.ParGridFunction_Assign(self, *args)

        return self


        return val


    def Save(self, *args):
        r"""
        Save(ParGridFunction self, std::ostream & out)
        Save(ParGridFunction self, char const * fname, int precision=16)
        Save(ParGridFunction self, char const * file, int precision=16)
        """
        return _pgridfunc.ParGridFunction_Save(self, *args)
    Save = _swig_new_instance_method(_pgridfunc.ParGridFunction_Save)

    def SaveGZ(self, file, precision=16):
        r"""SaveGZ(ParGridFunction self, char const * file, int precision=16)"""
        return _pgridfunc.ParGridFunction_SaveGZ(self, file, precision)
    SaveGZ = _swig_new_instance_method(_pgridfunc.ParGridFunction_SaveGZ)

    def SaveAsOne(self, *args):
        r"""
        SaveAsOne(ParGridFunction self, char const * fname, int precision=16)
        SaveAsOne(ParGridFunction self, std::ostream & out=out)
        SaveAsOne(ParGridFunction self, char const * file, int precision=16)
        """
        return _pgridfunc.ParGridFunction_SaveAsOne(self, *args)
    SaveAsOne = _swig_new_instance_method(_pgridfunc.ParGridFunction_SaveAsOne)

    def SaveAsOneGZ(self, file, precision=16):
        r"""SaveAsOneGZ(ParGridFunction self, char const * file, int precision=16)"""
        return _pgridfunc.ParGridFunction_SaveAsOneGZ(self, file, precision)
    SaveAsOneGZ = _swig_new_instance_method(_pgridfunc.ParGridFunction_SaveAsOneGZ)

# Register ParGridFunction in _pgridfunc:
_pgridfunc.ParGridFunction_swigregister(ParGridFunction)


def L2ZZErrorEstimator(flux_integrator, x, smooth_flux_fes, flux_fes, errors, norm_p=2, solver_tol=1e-12, solver_max_it=200):
    r"""L2ZZErrorEstimator(BilinearFormIntegrator flux_integrator, ParGridFunction x, ParFiniteElementSpace smooth_flux_fes, ParFiniteElementSpace flux_fes, Vector errors, int norm_p=2, double solver_tol=1e-12, int solver_max_it=200) -> double"""
    return _pgridfunc.L2ZZErrorEstimator(flux_integrator, x, smooth_flux_fes, flux_fes, errors, norm_p, solver_tol, solver_max_it)
L2ZZErrorEstimator = _pgridfunc.L2ZZErrorEstimator


