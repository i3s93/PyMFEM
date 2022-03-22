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
    from . import _fespacehierarchy
else:
    import _fespacehierarchy

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

_swig_new_instance_method = _fespacehierarchy.SWIG_PyInstanceMethod_New
_swig_new_static_method = _fespacehierarchy.SWIG_PyStaticMethod_New

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

import mfem._ser.vector
import mfem._ser.array
import mfem._ser.mem_manager
import mfem._ser.bilinearform
import mfem._ser.globals
import mfem._ser.fespace
import mfem._ser.coefficient
import mfem._ser.matrix
import mfem._ser.operators
import mfem._ser.symmat
import mfem._ser.intrules
import mfem._ser.sparsemat
import mfem._ser.densemat
import mfem._ser.eltrans
import mfem._ser.fe
import mfem._ser.geom
import mfem._ser.fe_base
import mfem._ser.fe_fixed_order
import mfem._ser.element
import mfem._ser.table
import mfem._ser.hash
import mfem._ser.fe_h1
import mfem._ser.fe_nd
import mfem._ser.fe_rt
import mfem._ser.fe_l2
import mfem._ser.fe_nurbs
import mfem._ser.fe_pos
import mfem._ser.fe_ser
import mfem._ser.mesh
import mfem._ser.sort_pairs
import mfem._ser.ncmesh
import mfem._ser.gridfunc
import mfem._ser.bilininteg
import mfem._ser.fe_coll
import mfem._ser.lininteg
import mfem._ser.linearform
import mfem._ser.nonlininteg
import mfem._ser.vertex
import mfem._ser.vtk
import mfem._ser.std_vectors
import mfem._ser.doftrans
import mfem._ser.handle
import mfem._ser.restriction
class FiniteElementSpaceHierarchy(object):
    r"""Proxy of C++ mfem::FiniteElementSpaceHierarchy class."""

    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        r"""
        __init__(FiniteElementSpaceHierarchy self) -> FiniteElementSpaceHierarchy
        __init__(FiniteElementSpaceHierarchy self, Mesh mesh, FiniteElementSpace fespace, bool ownM, bool ownFES) -> FiniteElementSpaceHierarchy
        """
        _fespacehierarchy.FiniteElementSpaceHierarchy_swiginit(self, _fespacehierarchy.new_FiniteElementSpaceHierarchy(*args))
    __swig_destroy__ = _fespacehierarchy.delete_FiniteElementSpaceHierarchy

    def GetNumLevels(self):
        r"""GetNumLevels(FiniteElementSpaceHierarchy self) -> int"""
        return _fespacehierarchy.FiniteElementSpaceHierarchy_GetNumLevels(self)
    GetNumLevels = _swig_new_instance_method(_fespacehierarchy.FiniteElementSpaceHierarchy_GetNumLevels)

    def GetFinestLevelIndex(self):
        r"""GetFinestLevelIndex(FiniteElementSpaceHierarchy self) -> int"""
        return _fespacehierarchy.FiniteElementSpaceHierarchy_GetFinestLevelIndex(self)
    GetFinestLevelIndex = _swig_new_instance_method(_fespacehierarchy.FiniteElementSpaceHierarchy_GetFinestLevelIndex)

    def AddLevel(self, mesh, fespace, prolongation, ownM, ownFES, ownP):
        r"""AddLevel(FiniteElementSpaceHierarchy self, Mesh mesh, FiniteElementSpace fespace, Operator prolongation, bool ownM, bool ownFES, bool ownP)"""
        return _fespacehierarchy.FiniteElementSpaceHierarchy_AddLevel(self, mesh, fespace, prolongation, ownM, ownFES, ownP)
    AddLevel = _swig_new_instance_method(_fespacehierarchy.FiniteElementSpaceHierarchy_AddLevel)

    def AddUniformlyRefinedLevel(self, *args, **kwargs):
        r"""AddUniformlyRefinedLevel(FiniteElementSpaceHierarchy self, int dim=1, int ordering=byVDIM)"""
        return _fespacehierarchy.FiniteElementSpaceHierarchy_AddUniformlyRefinedLevel(self, *args, **kwargs)
    AddUniformlyRefinedLevel = _swig_new_instance_method(_fespacehierarchy.FiniteElementSpaceHierarchy_AddUniformlyRefinedLevel)

    def AddOrderRefinedLevel(self, *args, **kwargs):
        r"""AddOrderRefinedLevel(FiniteElementSpaceHierarchy self, FiniteElementCollection fec, int dim=1, int ordering=byVDIM)"""
        return _fespacehierarchy.FiniteElementSpaceHierarchy_AddOrderRefinedLevel(self, *args, **kwargs)
    AddOrderRefinedLevel = _swig_new_instance_method(_fespacehierarchy.FiniteElementSpaceHierarchy_AddOrderRefinedLevel)

    def GetFESpaceAtLevel(self, *args):
        r"""
        GetFESpaceAtLevel(FiniteElementSpaceHierarchy self, int level) -> FiniteElementSpace
        GetFESpaceAtLevel(FiniteElementSpaceHierarchy self, int level) -> FiniteElementSpace
        """
        return _fespacehierarchy.FiniteElementSpaceHierarchy_GetFESpaceAtLevel(self, *args)
    GetFESpaceAtLevel = _swig_new_instance_method(_fespacehierarchy.FiniteElementSpaceHierarchy_GetFESpaceAtLevel)

    def GetFinestFESpace(self, *args):
        r"""
        GetFinestFESpace(FiniteElementSpaceHierarchy self) -> FiniteElementSpace
        GetFinestFESpace(FiniteElementSpaceHierarchy self) -> FiniteElementSpace
        """
        return _fespacehierarchy.FiniteElementSpaceHierarchy_GetFinestFESpace(self, *args)
    GetFinestFESpace = _swig_new_instance_method(_fespacehierarchy.FiniteElementSpaceHierarchy_GetFinestFESpace)

    def GetProlongationAtLevel(self, level):
        r"""GetProlongationAtLevel(FiniteElementSpaceHierarchy self, int level) -> Operator"""
        return _fespacehierarchy.FiniteElementSpaceHierarchy_GetProlongationAtLevel(self, level)
    GetProlongationAtLevel = _swig_new_instance_method(_fespacehierarchy.FiniteElementSpaceHierarchy_GetProlongationAtLevel)

# Register FiniteElementSpaceHierarchy in _fespacehierarchy:
_fespacehierarchy.FiniteElementSpaceHierarchy_swigregister(FiniteElementSpaceHierarchy)



