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

import mfem._par.vector
import mfem._par.array
import mfem._par.mem_manager
import mfem._par.bilinearform
import mfem._par.globals
import mfem._par.fespace
import mfem._par.coefficient
import mfem._par.matrix
import mfem._par.operators
import mfem._par.intrules
import mfem._par.sparsemat
import mfem._par.densemat
import mfem._par.eltrans
import mfem._par.fe
import mfem._par.geom
import mfem._par.mesh
import mfem._par.sort_pairs
import mfem._par.ncmesh
import mfem._par.vtk
import mfem._par.element
import mfem._par.table
import mfem._par.hash
import mfem._par.vertex
import mfem._par.gridfunc
import mfem._par.bilininteg
import mfem._par.fe_coll
import mfem._par.lininteg
import mfem._par.linearform
import mfem._par.nonlininteg
import mfem._par.handle
import mfem._par.hypre
import mfem._par.restriction
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

class ParFiniteElementSpaceHierarchy(FiniteElementSpaceHierarchy):
    r"""Proxy of C++ mfem::ParFiniteElementSpaceHierarchy class."""

    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, mesh, fespace, ownM, ownFES):
        r"""__init__(ParFiniteElementSpaceHierarchy self, mfem::ParMesh * mesh, mfem::ParFiniteElementSpace * fespace, bool ownM, bool ownFES) -> ParFiniteElementSpaceHierarchy"""
        _fespacehierarchy.ParFiniteElementSpaceHierarchy_swiginit(self, _fespacehierarchy.new_ParFiniteElementSpaceHierarchy(mesh, fespace, ownM, ownFES))

    def AddUniformlyRefinedLevel(self, *args, **kwargs):
        r"""AddUniformlyRefinedLevel(ParFiniteElementSpaceHierarchy self, int dim=1, int ordering=byVDIM)"""
        return _fespacehierarchy.ParFiniteElementSpaceHierarchy_AddUniformlyRefinedLevel(self, *args, **kwargs)
    AddUniformlyRefinedLevel = _swig_new_instance_method(_fespacehierarchy.ParFiniteElementSpaceHierarchy_AddUniformlyRefinedLevel)

    def AddOrderRefinedLevel(self, *args, **kwargs):
        r"""AddOrderRefinedLevel(ParFiniteElementSpaceHierarchy self, FiniteElementCollection fec, int dim=1, int ordering=byVDIM)"""
        return _fespacehierarchy.ParFiniteElementSpaceHierarchy_AddOrderRefinedLevel(self, *args, **kwargs)
    AddOrderRefinedLevel = _swig_new_instance_method(_fespacehierarchy.ParFiniteElementSpaceHierarchy_AddOrderRefinedLevel)

    def GetFESpaceAtLevel(self, *args):
        r"""
        GetFESpaceAtLevel(ParFiniteElementSpaceHierarchy self, int level) -> mfem::ParFiniteElementSpace const
        GetFESpaceAtLevel(ParFiniteElementSpaceHierarchy self, int level) -> mfem::ParFiniteElementSpace &
        """
        return _fespacehierarchy.ParFiniteElementSpaceHierarchy_GetFESpaceAtLevel(self, *args)
    GetFESpaceAtLevel = _swig_new_instance_method(_fespacehierarchy.ParFiniteElementSpaceHierarchy_GetFESpaceAtLevel)

    def GetFinestFESpace(self, *args):
        r"""
        GetFinestFESpace(ParFiniteElementSpaceHierarchy self) -> mfem::ParFiniteElementSpace const
        GetFinestFESpace(ParFiniteElementSpaceHierarchy self) -> mfem::ParFiniteElementSpace &
        """
        return _fespacehierarchy.ParFiniteElementSpaceHierarchy_GetFinestFESpace(self, *args)
    GetFinestFESpace = _swig_new_instance_method(_fespacehierarchy.ParFiniteElementSpaceHierarchy_GetFinestFESpace)
    __swig_destroy__ = _fespacehierarchy.delete_ParFiniteElementSpaceHierarchy

# Register ParFiniteElementSpaceHierarchy in _fespacehierarchy:
_fespacehierarchy.ParFiniteElementSpaceHierarchy_swigregister(ParFiniteElementSpaceHierarchy)



