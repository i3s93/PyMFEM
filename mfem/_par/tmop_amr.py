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
    from . import _tmop_amr
else:
    import _tmop_amr

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

_swig_new_instance_method = _tmop_amr.SWIG_PyInstanceMethod_New
_swig_new_static_method = _tmop_amr.SWIG_PyStaticMethod_New

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

MFEM_VERSION = _tmop_amr.MFEM_VERSION
MFEM_VERSION_STRING = _tmop_amr.MFEM_VERSION_STRING
MFEM_VERSION_TYPE = _tmop_amr.MFEM_VERSION_TYPE
MFEM_VERSION_TYPE_RELEASE = _tmop_amr.MFEM_VERSION_TYPE_RELEASE
MFEM_VERSION_TYPE_DEVELOPMENT = _tmop_amr.MFEM_VERSION_TYPE_DEVELOPMENT
MFEM_VERSION_MAJOR = _tmop_amr.MFEM_VERSION_MAJOR
MFEM_VERSION_MINOR = _tmop_amr.MFEM_VERSION_MINOR
MFEM_VERSION_PATCH = _tmop_amr.MFEM_VERSION_PATCH
import mfem._par.tmop
import mfem._par.intrules
import mfem._par.array
import mfem._par.mem_manager
import mfem._par.gridfunc
import mfem._par.vector
import mfem._par.coefficient
import mfem._par.globals
import mfem._par.matrix
import mfem._par.operators
import mfem._par.symmat
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
import mfem._par.fespace
import mfem._par.mesh
import mfem._par.sort_pairs
import mfem._par.ncmesh
import mfem._par.vtk
import mfem._par.vertex
import mfem._par.std_vectors
import mfem._par.fe_coll
import mfem._par.lininteg
import mfem._par.doftrans
import mfem._par.handle
import mfem._par.hypre
import mfem._par.restriction
import mfem._par.bilininteg
import mfem._par.linearform
import mfem._par.nonlininteg
import mfem._par.estimators
import mfem._par.bilinearform
import mfem._par.nonlinearform
import mfem._par.pnonlinearform
import mfem._par.blockoperator
import mfem._par.pfespace
import mfem._par.pmesh
import mfem._par.pncmesh
import mfem._par.communication
import mfem._par.sets
import mfem._par.pgridfunc
class TMOPRefinerEstimator(mfem._par.estimators.AnisotropicErrorEstimator):
    r"""Proxy of C++ mfem::TMOPRefinerEstimator class."""

    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, mesh_, nlf_, order_, amrmetric_):
        r"""__init__(TMOPRefinerEstimator self, Mesh mesh_, NonlinearForm nlf_, int order_, int amrmetric_) -> TMOPRefinerEstimator"""
        _tmop_amr.TMOPRefinerEstimator_swiginit(self, _tmop_amr.new_TMOPRefinerEstimator(mesh_, nlf_, order_, amrmetric_))
    __swig_destroy__ = _tmop_amr.delete_TMOPRefinerEstimator

    def GetLocalErrors(self):
        r"""GetLocalErrors(TMOPRefinerEstimator self) -> Vector"""
        return _tmop_amr.TMOPRefinerEstimator_GetLocalErrors(self)
    GetLocalErrors = _swig_new_instance_method(_tmop_amr.TMOPRefinerEstimator_GetLocalErrors)

    def GetAnisotropicFlags(self):
        r"""GetAnisotropicFlags(TMOPRefinerEstimator self) -> intArray"""
        return _tmop_amr.TMOPRefinerEstimator_GetAnisotropicFlags(self)
    GetAnisotropicFlags = _swig_new_instance_method(_tmop_amr.TMOPRefinerEstimator_GetAnisotropicFlags)

    def SetEnergyScalingFactor(self, scale):
        r"""SetEnergyScalingFactor(TMOPRefinerEstimator self, double scale)"""
        return _tmop_amr.TMOPRefinerEstimator_SetEnergyScalingFactor(self, scale)
    SetEnergyScalingFactor = _swig_new_instance_method(_tmop_amr.TMOPRefinerEstimator_SetEnergyScalingFactor)

    def SetSpatialIndicator(self, spat_gf_, spat_gf_critical_=0.5):
        r"""SetSpatialIndicator(TMOPRefinerEstimator self, GridFunction spat_gf_, double spat_gf_critical_=0.5)"""
        return _tmop_amr.TMOPRefinerEstimator_SetSpatialIndicator(self, spat_gf_, spat_gf_critical_)
    SetSpatialIndicator = _swig_new_instance_method(_tmop_amr.TMOPRefinerEstimator_SetSpatialIndicator)

    def SetSpatialIndicatorCritical(self, val_):
        r"""SetSpatialIndicatorCritical(TMOPRefinerEstimator self, double val_)"""
        return _tmop_amr.TMOPRefinerEstimator_SetSpatialIndicatorCritical(self, val_)
    SetSpatialIndicatorCritical = _swig_new_instance_method(_tmop_amr.TMOPRefinerEstimator_SetSpatialIndicatorCritical)

    def Reset(self):
        r"""Reset(TMOPRefinerEstimator self)"""
        return _tmop_amr.TMOPRefinerEstimator_Reset(self)
    Reset = _swig_new_instance_method(_tmop_amr.TMOPRefinerEstimator_Reset)

# Register TMOPRefinerEstimator in _tmop_amr:
_tmop_amr.TMOPRefinerEstimator_swigregister(TMOPRefinerEstimator)

class TMOPDeRefinerEstimator(mfem._par.estimators.ErrorEstimator):
    r"""Proxy of C++ mfem::TMOPDeRefinerEstimator class."""

    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        r"""
        __init__(TMOPDeRefinerEstimator self, Mesh mesh_, NonlinearForm nlf_) -> TMOPDeRefinerEstimator
        __init__(TMOPDeRefinerEstimator self, ParMesh pmesh_, ParNonlinearForm pnlf_) -> TMOPDeRefinerEstimator
        """
        _tmop_amr.TMOPDeRefinerEstimator_swiginit(self, _tmop_amr.new_TMOPDeRefinerEstimator(*args))
    __swig_destroy__ = _tmop_amr.delete_TMOPDeRefinerEstimator

    def GetLocalErrors(self):
        r"""GetLocalErrors(TMOPDeRefinerEstimator self) -> Vector"""
        return _tmop_amr.TMOPDeRefinerEstimator_GetLocalErrors(self)
    GetLocalErrors = _swig_new_instance_method(_tmop_amr.TMOPDeRefinerEstimator_GetLocalErrors)

    def Reset(self):
        r"""Reset(TMOPDeRefinerEstimator self)"""
        return _tmop_amr.TMOPDeRefinerEstimator_Reset(self)
    Reset = _swig_new_instance_method(_tmop_amr.TMOPDeRefinerEstimator_Reset)

# Register TMOPDeRefinerEstimator in _tmop_amr:
_tmop_amr.TMOPDeRefinerEstimator_swigregister(TMOPDeRefinerEstimator)

class TMOPHRSolver(object):
    r"""Proxy of C++ mfem::TMOPHRSolver class."""

    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        r"""
        __init__(TMOPHRSolver self, Mesh mesh_, NonlinearForm nlf_, mfem::TMOPNewtonSolver & tmopns_, GridFunction x_, bool move_bnd_, bool hradaptivity_, int mesh_poly_deg_, int amr_metric_id_, int hr_iter_=5, int h_per_r_iter_=1) -> TMOPHRSolver
        __init__(TMOPHRSolver self, ParMesh pmesh_, ParNonlinearForm pnlf_, mfem::TMOPNewtonSolver & tmopns_, ParGridFunction x_, bool move_bnd_, bool hradaptivity_, int mesh_poly_deg_, int amr_metric_id_, int hr_iter_=5, int h_per_r_iter_=1) -> TMOPHRSolver
        """
        _tmop_amr.TMOPHRSolver_swiginit(self, _tmop_amr.new_TMOPHRSolver(*args))

    def Mult(self):
        r"""Mult(TMOPHRSolver self)"""
        return _tmop_amr.TMOPHRSolver_Mult(self)
    Mult = _swig_new_instance_method(_tmop_amr.TMOPHRSolver_Mult)

    def AddGridFunctionForUpdate(self, *args):
        r"""
        AddGridFunctionForUpdate(TMOPHRSolver self, GridFunction gf)
        AddGridFunctionForUpdate(TMOPHRSolver self, ParGridFunction pgf_)
        """
        return _tmop_amr.TMOPHRSolver_AddGridFunctionForUpdate(self, *args)
    AddGridFunctionForUpdate = _swig_new_instance_method(_tmop_amr.TMOPHRSolver_AddGridFunctionForUpdate)

    def AddFESpaceForUpdate(self, *args):
        r"""
        AddFESpaceForUpdate(TMOPHRSolver self, FiniteElementSpace fes)
        AddFESpaceForUpdate(TMOPHRSolver self, ParFiniteElementSpace pfes_)
        """
        return _tmop_amr.TMOPHRSolver_AddFESpaceForUpdate(self, *args)
    AddFESpaceForUpdate = _swig_new_instance_method(_tmop_amr.TMOPHRSolver_AddFESpaceForUpdate)
    __swig_destroy__ = _tmop_amr.delete_TMOPHRSolver

    def SetHRAdaptivityIterations(self, iter):
        r"""SetHRAdaptivityIterations(TMOPHRSolver self, int iter)"""
        return _tmop_amr.TMOPHRSolver_SetHRAdaptivityIterations(self, iter)
    SetHRAdaptivityIterations = _swig_new_instance_method(_tmop_amr.TMOPHRSolver_SetHRAdaptivityIterations)

    def SetHAdaptivityIterations(self, iter):
        r"""SetHAdaptivityIterations(TMOPHRSolver self, int iter)"""
        return _tmop_amr.TMOPHRSolver_SetHAdaptivityIterations(self, iter)
    SetHAdaptivityIterations = _swig_new_instance_method(_tmop_amr.TMOPHRSolver_SetHAdaptivityIterations)

# Register TMOPHRSolver in _tmop_amr:
_tmop_amr.TMOPHRSolver_swigregister(TMOPHRSolver)



