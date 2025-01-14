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
    from . import _handle
else:
    import _handle

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

_swig_new_instance_method = _handle.SWIG_PyInstanceMethod_New
_swig_new_static_method = _handle.SWIG_PyStaticMethod_New

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

MFEM_VERSION = _handle.MFEM_VERSION

MFEM_VERSION_STRING = _handle.MFEM_VERSION_STRING

MFEM_VERSION_TYPE = _handle.MFEM_VERSION_TYPE

MFEM_VERSION_TYPE_RELEASE = _handle.MFEM_VERSION_TYPE_RELEASE

MFEM_VERSION_TYPE_DEVELOPMENT = _handle.MFEM_VERSION_TYPE_DEVELOPMENT

MFEM_VERSION_MAJOR = _handle.MFEM_VERSION_MAJOR

MFEM_VERSION_MINOR = _handle.MFEM_VERSION_MINOR

MFEM_VERSION_PATCH = _handle.MFEM_VERSION_PATCH

import mfem._ser.operators
import mfem._ser.mem_manager
import mfem._ser.vector
import mfem._ser.array
class OperatorHandle(object):
    r"""Proxy of C++ mfem::OperatorHandle class."""

    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        r"""
        __init__(OperatorHandle self) -> OperatorHandle
        __init__(OperatorHandle self, mfem::Operator::Type tid) -> OperatorHandle
        __init__(OperatorHandle self, OperatorHandle other) -> OperatorHandle
        """
        _handle.OperatorHandle_swiginit(self, _handle.new_OperatorHandle(*args))
    __swig_destroy__ = _handle.delete_OperatorHandle

    def Ptr(self):
        r"""Ptr(OperatorHandle self) -> Operator"""
        return _handle.OperatorHandle_Ptr(self)
    Ptr = _swig_new_instance_method(_handle.OperatorHandle_Ptr)

    def __deref__(self):
        r"""__deref__(OperatorHandle self) -> Operator"""
        return _handle.OperatorHandle___deref__(self)
    __deref__ = _swig_new_instance_method(_handle.OperatorHandle___deref__)

    def __ref__(self, *args):
        r"""
        __ref__(OperatorHandle self) -> Operator
        __ref__(OperatorHandle self) -> Operator
        """
        return _handle.OperatorHandle___ref__(self, *args)
    __ref__ = _swig_new_instance_method(_handle.OperatorHandle___ref__)

    def Type(self):
        r"""Type(OperatorHandle self) -> mfem::Operator::Type"""
        return _handle.OperatorHandle_Type(self)
    Type = _swig_new_instance_method(_handle.OperatorHandle_Type)

    def OwnsOperator(self):
        r"""OwnsOperator(OperatorHandle self) -> bool"""
        return _handle.OperatorHandle_OwnsOperator(self)
    OwnsOperator = _swig_new_instance_method(_handle.OperatorHandle_OwnsOperator)

    def SetOperatorOwner(self, own=True):
        r"""SetOperatorOwner(OperatorHandle self, bool own=True)"""
        return _handle.OperatorHandle_SetOperatorOwner(self, own)
    SetOperatorOwner = _swig_new_instance_method(_handle.OperatorHandle_SetOperatorOwner)

    def Clear(self):
        r"""Clear(OperatorHandle self)"""
        return _handle.OperatorHandle_Clear(self)
    Clear = _swig_new_instance_method(_handle.OperatorHandle_Clear)

    def SetType(self, tid):
        r"""SetType(OperatorHandle self, mfem::Operator::Type tid)"""
        return _handle.OperatorHandle_SetType(self, tid)
    SetType = _swig_new_instance_method(_handle.OperatorHandle_SetType)

    def MakePtAP(self, A, P):
        r"""MakePtAP(OperatorHandle self, OperatorHandle A, OperatorHandle P)"""
        return _handle.OperatorHandle_MakePtAP(self, A, P)
    MakePtAP = _swig_new_instance_method(_handle.OperatorHandle_MakePtAP)

    def MakeRAP(self, Rt, A, P):
        r"""MakeRAP(OperatorHandle self, OperatorHandle Rt, OperatorHandle A, OperatorHandle P)"""
        return _handle.OperatorHandle_MakeRAP(self, Rt, A, P)
    MakeRAP = _swig_new_instance_method(_handle.OperatorHandle_MakeRAP)

    def ConvertFrom(self, A):
        r"""ConvertFrom(OperatorHandle self, OperatorHandle A)"""
        return _handle.OperatorHandle_ConvertFrom(self, A)
    ConvertFrom = _swig_new_instance_method(_handle.OperatorHandle_ConvertFrom)

    def EliminateRowsCols(self, A, ess_dof_list):
        r"""EliminateRowsCols(OperatorHandle self, OperatorHandle A, intArray ess_dof_list)"""
        return _handle.OperatorHandle_EliminateRowsCols(self, A, ess_dof_list)
    EliminateRowsCols = _swig_new_instance_method(_handle.OperatorHandle_EliminateRowsCols)

    def EliminateRows(self, ess_dof_list):
        r"""EliminateRows(OperatorHandle self, intArray ess_dof_list)"""
        return _handle.OperatorHandle_EliminateRows(self, ess_dof_list)
    EliminateRows = _swig_new_instance_method(_handle.OperatorHandle_EliminateRows)

    def EliminateCols(self, ess_dof_list):
        r"""EliminateCols(OperatorHandle self, intArray ess_dof_list)"""
        return _handle.OperatorHandle_EliminateCols(self, ess_dof_list)
    EliminateCols = _swig_new_instance_method(_handle.OperatorHandle_EliminateCols)

    def EliminateBC(self, A_e, ess_dof_list, X, B):
        r"""EliminateBC(OperatorHandle self, OperatorHandle A_e, intArray ess_dof_list, Vector X, Vector B)"""
        return _handle.OperatorHandle_EliminateBC(self, A_e, ess_dof_list, X, B)
    EliminateBC = _swig_new_instance_method(_handle.OperatorHandle_EliminateBC)

    def AsSparseMatrix(self):
        r"""AsSparseMatrix(OperatorHandle self) -> mfem::SparseMatrix *"""
        return _handle.OperatorHandle_AsSparseMatrix(self)
    AsSparseMatrix = _swig_new_instance_method(_handle.OperatorHandle_AsSparseMatrix)

    def IsSparseMatrix(self):
        r"""IsSparseMatrix(OperatorHandle self) -> mfem::SparseMatrix *"""
        return _handle.OperatorHandle_IsSparseMatrix(self)
    IsSparseMatrix = _swig_new_instance_method(_handle.OperatorHandle_IsSparseMatrix)

    def GetSparseMatrix(self, A):
        r"""GetSparseMatrix(OperatorHandle self, mfem::SparseMatrix *& A)"""
        return _handle.OperatorHandle_GetSparseMatrix(self, A)
    GetSparseMatrix = _swig_new_instance_method(_handle.OperatorHandle_GetSparseMatrix)

    def ResetSparseMatrix(self, A, own_A=True):
        r"""ResetSparseMatrix(OperatorHandle self, mfem::SparseMatrix * A, bool own_A=True)"""
        return _handle.OperatorHandle_ResetSparseMatrix(self, A, own_A)
    ResetSparseMatrix = _swig_new_instance_method(_handle.OperatorHandle_ResetSparseMatrix)

    def ConvertFromSparseMatrix(self, A):
        r"""ConvertFromSparseMatrix(OperatorHandle self, mfem::SparseMatrix * A)"""
        return _handle.OperatorHandle_ConvertFromSparseMatrix(self, A)
    ConvertFromSparseMatrix = _swig_new_instance_method(_handle.OperatorHandle_ConvertFromSparseMatrix)

    def InitTVectors(self, Po, Ri, Pi, x, b, X, B):
        r"""InitTVectors(OperatorHandle self, Operator Po, Operator Ri, Operator Pi, Vector x, Vector b, Vector X, Vector B)"""
        return _handle.OperatorHandle_InitTVectors(self, Po, Ri, Pi, x, b, X, B)
    InitTVectors = _swig_new_instance_method(_handle.OperatorHandle_InitTVectors)

    def Height(self):
        r"""Height(OperatorHandle self) -> int"""
        return _handle.OperatorHandle_Height(self)
    Height = _swig_new_instance_method(_handle.OperatorHandle_Height)

    def NumRows(self):
        r"""NumRows(OperatorHandle self) -> int"""
        return _handle.OperatorHandle_NumRows(self)
    NumRows = _swig_new_instance_method(_handle.OperatorHandle_NumRows)

    def Width(self):
        r"""Width(OperatorHandle self) -> int"""
        return _handle.OperatorHandle_Width(self)
    Width = _swig_new_instance_method(_handle.OperatorHandle_Width)

    def NumCols(self):
        r"""NumCols(OperatorHandle self) -> int"""
        return _handle.OperatorHandle_NumCols(self)
    NumCols = _swig_new_instance_method(_handle.OperatorHandle_NumCols)

    def GetMemoryClass(self):
        r"""GetMemoryClass(OperatorHandle self) -> mfem::MemoryClass"""
        return _handle.OperatorHandle_GetMemoryClass(self)
    GetMemoryClass = _swig_new_instance_method(_handle.OperatorHandle_GetMemoryClass)

    def Mult(self, x, y):
        r"""Mult(OperatorHandle self, Vector x, Vector y)"""
        return _handle.OperatorHandle_Mult(self, x, y)
    Mult = _swig_new_instance_method(_handle.OperatorHandle_Mult)

    def MultTranspose(self, x, y):
        r"""MultTranspose(OperatorHandle self, Vector x, Vector y)"""
        return _handle.OperatorHandle_MultTranspose(self, x, y)
    MultTranspose = _swig_new_instance_method(_handle.OperatorHandle_MultTranspose)

    def GetGradient(self, x):
        r"""GetGradient(OperatorHandle self, Vector x) -> Operator"""
        return _handle.OperatorHandle_GetGradient(self, x)
    GetGradient = _swig_new_instance_method(_handle.OperatorHandle_GetGradient)

    def AssembleDiagonal(self, diag):
        r"""AssembleDiagonal(OperatorHandle self, Vector diag)"""
        return _handle.OperatorHandle_AssembleDiagonal(self, diag)
    AssembleDiagonal = _swig_new_instance_method(_handle.OperatorHandle_AssembleDiagonal)

    def GetProlongation(self):
        r"""GetProlongation(OperatorHandle self) -> Operator"""
        return _handle.OperatorHandle_GetProlongation(self)
    GetProlongation = _swig_new_instance_method(_handle.OperatorHandle_GetProlongation)

    def GetRestriction(self):
        r"""GetRestriction(OperatorHandle self) -> Operator"""
        return _handle.OperatorHandle_GetRestriction(self)
    GetRestriction = _swig_new_instance_method(_handle.OperatorHandle_GetRestriction)

    def GetOutputProlongation(self):
        r"""GetOutputProlongation(OperatorHandle self) -> Operator"""
        return _handle.OperatorHandle_GetOutputProlongation(self)
    GetOutputProlongation = _swig_new_instance_method(_handle.OperatorHandle_GetOutputProlongation)

    def GetOutputRestrictionTranspose(self):
        r"""GetOutputRestrictionTranspose(OperatorHandle self) -> Operator"""
        return _handle.OperatorHandle_GetOutputRestrictionTranspose(self)
    GetOutputRestrictionTranspose = _swig_new_instance_method(_handle.OperatorHandle_GetOutputRestrictionTranspose)

    def GetOutputRestriction(self):
        r"""GetOutputRestriction(OperatorHandle self) -> Operator"""
        return _handle.OperatorHandle_GetOutputRestriction(self)
    GetOutputRestriction = _swig_new_instance_method(_handle.OperatorHandle_GetOutputRestriction)

    def FormLinearSystem(self, ess_tdof_list, x, b, A, X, B, copy_interior=0):
        r"""FormLinearSystem(OperatorHandle self, intArray ess_tdof_list, Vector x, Vector b, mfem::Operator *& A, Vector X, Vector B, int copy_interior=0)"""
        return _handle.OperatorHandle_FormLinearSystem(self, ess_tdof_list, x, b, A, X, B, copy_interior)
    FormLinearSystem = _swig_new_instance_method(_handle.OperatorHandle_FormLinearSystem)

    def FormRectangularLinearSystem(self, trial_tdof_list, test_tdof_list, x, b, A, X, B):
        r"""FormRectangularLinearSystem(OperatorHandle self, intArray trial_tdof_list, intArray test_tdof_list, Vector x, Vector b, mfem::Operator *& A, Vector X, Vector B)"""
        return _handle.OperatorHandle_FormRectangularLinearSystem(self, trial_tdof_list, test_tdof_list, x, b, A, X, B)
    FormRectangularLinearSystem = _swig_new_instance_method(_handle.OperatorHandle_FormRectangularLinearSystem)

    def RecoverFEMSolution(self, X, b, x):
        r"""RecoverFEMSolution(OperatorHandle self, Vector X, Vector b, Vector x)"""
        return _handle.OperatorHandle_RecoverFEMSolution(self, X, b, x)
    RecoverFEMSolution = _swig_new_instance_method(_handle.OperatorHandle_RecoverFEMSolution)

    def FormSystemOperator(self, ess_tdof_list, A):
        r"""FormSystemOperator(OperatorHandle self, intArray ess_tdof_list, mfem::Operator *& A)"""
        return _handle.OperatorHandle_FormSystemOperator(self, ess_tdof_list, A)
    FormSystemOperator = _swig_new_instance_method(_handle.OperatorHandle_FormSystemOperator)

    def FormRectangularSystemOperator(self, trial_tdof_list, test_tdof_list, A):
        r"""FormRectangularSystemOperator(OperatorHandle self, intArray trial_tdof_list, intArray test_tdof_list, mfem::Operator *& A)"""
        return _handle.OperatorHandle_FormRectangularSystemOperator(self, trial_tdof_list, test_tdof_list, A)
    FormRectangularSystemOperator = _swig_new_instance_method(_handle.OperatorHandle_FormRectangularSystemOperator)

    def FormDiscreteOperator(self, A):
        r"""FormDiscreteOperator(OperatorHandle self, mfem::Operator *& A)"""
        return _handle.OperatorHandle_FormDiscreteOperator(self, A)
    FormDiscreteOperator = _swig_new_instance_method(_handle.OperatorHandle_FormDiscreteOperator)

    def GetType(self):
        r"""GetType(OperatorHandle self) -> mfem::Operator::Type"""
        return _handle.OperatorHandle_GetType(self)
    GetType = _swig_new_instance_method(_handle.OperatorHandle_GetType)

# Register OperatorHandle in _handle:
_handle.OperatorHandle_swigregister(OperatorHandle)


OperatorPtr=OperatorHandle  



