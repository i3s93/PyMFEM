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
    from . import _constraints
else:
    import _constraints

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

_swig_new_instance_method = _constraints.SWIG_PyInstanceMethod_New
_swig_new_static_method = _constraints.SWIG_PyStaticMethod_New

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
import mfem._ser.fespace
import mfem._ser.coefficient
import mfem._ser.globals
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
import mfem._ser.solvers
class ConstrainedSolver(mfem._ser.solvers.IterativeSolver):
    r"""Proxy of C++ mfem::ConstrainedSolver class."""

    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, A_, B_):
        r"""__init__(ConstrainedSolver self, Operator A_, Operator B_) -> ConstrainedSolver"""
        _constraints.ConstrainedSolver_swiginit(self, _constraints.new_ConstrainedSolver(A_, B_))
    __swig_destroy__ = _constraints.delete_ConstrainedSolver

    def SetOperator(self, op):
        r"""SetOperator(ConstrainedSolver self, Operator op)"""
        return _constraints.ConstrainedSolver_SetOperator(self, op)
    SetOperator = _swig_new_instance_method(_constraints.ConstrainedSolver_SetOperator)

    def SetConstraintRHS(self, r):
        r"""SetConstraintRHS(ConstrainedSolver self, Vector r)"""
        return _constraints.ConstrainedSolver_SetConstraintRHS(self, r)
    SetConstraintRHS = _swig_new_instance_method(_constraints.ConstrainedSolver_SetConstraintRHS)

    def GetMultiplierSolution(self, _lambda):
        r"""GetMultiplierSolution(ConstrainedSolver self, Vector _lambda)"""
        return _constraints.ConstrainedSolver_GetMultiplierSolution(self, _lambda)
    GetMultiplierSolution = _swig_new_instance_method(_constraints.ConstrainedSolver_GetMultiplierSolution)

    def Mult(self, f, x):
        r"""Mult(ConstrainedSolver self, Vector f, Vector x)"""
        return _constraints.ConstrainedSolver_Mult(self, f, x)
    Mult = _swig_new_instance_method(_constraints.ConstrainedSolver_Mult)

    def LagrangeSystemMult(self, f_and_r, x_and_lambda):
        r"""LagrangeSystemMult(ConstrainedSolver self, Vector f_and_r, Vector x_and_lambda)"""
        return _constraints.ConstrainedSolver_LagrangeSystemMult(self, f_and_r, x_and_lambda)
    LagrangeSystemMult = _swig_new_instance_method(_constraints.ConstrainedSolver_LagrangeSystemMult)

# Register ConstrainedSolver in _constraints:
_constraints.ConstrainedSolver_swigregister(ConstrainedSolver)

class Eliminator(object):
    r"""Proxy of C++ mfem::Eliminator class."""

    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, B, lagrange_dofs, primary_tdofs, secondary_tdofs):
        r"""__init__(Eliminator self, SparseMatrix B, intArray lagrange_dofs, intArray primary_tdofs, intArray secondary_tdofs) -> Eliminator"""
        _constraints.Eliminator_swiginit(self, _constraints.new_Eliminator(B, lagrange_dofs, primary_tdofs, secondary_tdofs))

    def LagrangeDofs(self):
        r"""LagrangeDofs(Eliminator self) -> intArray"""
        return _constraints.Eliminator_LagrangeDofs(self)
    LagrangeDofs = _swig_new_instance_method(_constraints.Eliminator_LagrangeDofs)

    def PrimaryDofs(self):
        r"""PrimaryDofs(Eliminator self) -> intArray"""
        return _constraints.Eliminator_PrimaryDofs(self)
    PrimaryDofs = _swig_new_instance_method(_constraints.Eliminator_PrimaryDofs)

    def SecondaryDofs(self):
        r"""SecondaryDofs(Eliminator self) -> intArray"""
        return _constraints.Eliminator_SecondaryDofs(self)
    SecondaryDofs = _swig_new_instance_method(_constraints.Eliminator_SecondaryDofs)

    def Eliminate(self, _in, out):
        r"""Eliminate(Eliminator self, Vector _in, Vector out)"""
        return _constraints.Eliminator_Eliminate(self, _in, out)
    Eliminate = _swig_new_instance_method(_constraints.Eliminator_Eliminate)

    def EliminateTranspose(self, _in, out):
        r"""EliminateTranspose(Eliminator self, Vector _in, Vector out)"""
        return _constraints.Eliminator_EliminateTranspose(self, _in, out)
    EliminateTranspose = _swig_new_instance_method(_constraints.Eliminator_EliminateTranspose)

    def LagrangeSecondary(self, _in, out):
        r"""LagrangeSecondary(Eliminator self, Vector _in, Vector out)"""
        return _constraints.Eliminator_LagrangeSecondary(self, _in, out)
    LagrangeSecondary = _swig_new_instance_method(_constraints.Eliminator_LagrangeSecondary)

    def LagrangeSecondaryTranspose(self, _in, out):
        r"""LagrangeSecondaryTranspose(Eliminator self, Vector _in, Vector out)"""
        return _constraints.Eliminator_LagrangeSecondaryTranspose(self, _in, out)
    LagrangeSecondaryTranspose = _swig_new_instance_method(_constraints.Eliminator_LagrangeSecondaryTranspose)

    def ExplicitAssembly(self, mat):
        r"""ExplicitAssembly(Eliminator self, DenseMatrix mat)"""
        return _constraints.Eliminator_ExplicitAssembly(self, mat)
    ExplicitAssembly = _swig_new_instance_method(_constraints.Eliminator_ExplicitAssembly)
    __swig_destroy__ = _constraints.delete_Eliminator

# Register Eliminator in _constraints:
_constraints.Eliminator_swigregister(Eliminator)

class EliminationProjection(mfem._ser.operators.Operator):
    r"""Proxy of C++ mfem::EliminationProjection class."""

    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, A, eliminators):
        r"""__init__(EliminationProjection self, Operator A, mfem::Array< mfem::Eliminator * > & eliminators) -> EliminationProjection"""
        _constraints.EliminationProjection_swiginit(self, _constraints.new_EliminationProjection(A, eliminators))

    def Mult(self, x, y):
        r"""Mult(EliminationProjection self, Vector x, Vector y)"""
        return _constraints.EliminationProjection_Mult(self, x, y)
    Mult = _swig_new_instance_method(_constraints.EliminationProjection_Mult)

    def MultTranspose(self, x, y):
        r"""MultTranspose(EliminationProjection self, Vector x, Vector y)"""
        return _constraints.EliminationProjection_MultTranspose(self, x, y)
    MultTranspose = _swig_new_instance_method(_constraints.EliminationProjection_MultTranspose)

    def AssembleExact(self):
        r"""AssembleExact(EliminationProjection self) -> SparseMatrix"""
        return _constraints.EliminationProjection_AssembleExact(self)
    AssembleExact = _swig_new_instance_method(_constraints.EliminationProjection_AssembleExact)

    def BuildGTilde(self, g, gtilde):
        r"""BuildGTilde(EliminationProjection self, Vector g, Vector gtilde)"""
        return _constraints.EliminationProjection_BuildGTilde(self, g, gtilde)
    BuildGTilde = _swig_new_instance_method(_constraints.EliminationProjection_BuildGTilde)

    def RecoverMultiplier(self, primalrhs, primalvars, lm):
        r"""RecoverMultiplier(EliminationProjection self, Vector primalrhs, Vector primalvars, Vector lm)"""
        return _constraints.EliminationProjection_RecoverMultiplier(self, primalrhs, primalvars, lm)
    RecoverMultiplier = _swig_new_instance_method(_constraints.EliminationProjection_RecoverMultiplier)
    __swig_destroy__ = _constraints.delete_EliminationProjection

# Register EliminationProjection in _constraints:
_constraints.EliminationProjection_swigregister(EliminationProjection)

class SchurConstrainedSolver(ConstrainedSolver):
    r"""Proxy of C++ mfem::SchurConstrainedSolver class."""

    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, A_, B_, primal_pc_):
        r"""__init__(SchurConstrainedSolver self, Operator A_, Operator B_, Solver primal_pc_) -> SchurConstrainedSolver"""
        _constraints.SchurConstrainedSolver_swiginit(self, _constraints.new_SchurConstrainedSolver(A_, B_, primal_pc_))
    __swig_destroy__ = _constraints.delete_SchurConstrainedSolver

    def LagrangeSystemMult(self, x, y):
        r"""LagrangeSystemMult(SchurConstrainedSolver self, Vector x, Vector y)"""
        return _constraints.SchurConstrainedSolver_LagrangeSystemMult(self, x, y)
    LagrangeSystemMult = _swig_new_instance_method(_constraints.SchurConstrainedSolver_LagrangeSystemMult)

# Register SchurConstrainedSolver in _constraints:
_constraints.SchurConstrainedSolver_swigregister(SchurConstrainedSolver)


def BuildNormalConstraints(fespace, constrained_att, constraint_rowstarts, parallel=False):
    r"""BuildNormalConstraints(FiniteElementSpace fespace, intArray constrained_att, intArray constraint_rowstarts, bool parallel=False) -> SparseMatrix"""
    return _constraints.BuildNormalConstraints(fespace, constrained_att, constraint_rowstarts, parallel)
BuildNormalConstraints = _constraints.BuildNormalConstraints


