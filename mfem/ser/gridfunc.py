# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.8
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.





from sys import version_info
if version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_gridfunc', [dirname(__file__)])
        except ImportError:
            import _gridfunc
            return _gridfunc
        if fp is not None:
            try:
                _mod = imp.load_module('_gridfunc', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _gridfunc = swig_import_helper()
    del swig_import_helper
else:
    import _gridfunc
del version_info
try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.


def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr_nondynamic(self, class_type, name, static=1):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    if (not static):
        return object.__getattr__(self, name)
    else:
        raise AttributeError(name)

def _swig_getattr(self, class_type, name):
    return _swig_getattr_nondynamic(self, class_type, name, 0)


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object:
        pass
    _newclass = 0


try:
    import weakref
    weakref_proxy = weakref.proxy
except Exception:
    weakref_proxy = lambda x: x


import array
import vector
import coefficient
import matrix
import operators
import intrules
import sparsemat
import densemat
import eltrans
import fe
import fespace
import mesh
import ncmesh
import element
import geom
import table
import vertex
import fe_coll
import lininteg
import bilininteg
import linearform
class GridFunction(vector.Vector):
    __swig_setmethods__ = {}
    for _s in [vector.Vector]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, GridFunction, name, value)
    __swig_getmethods__ = {}
    for _s in [vector.Vector]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, GridFunction, name)
    __repr__ = _swig_repr

    def MakeOwner(self, _fec):
        return _gridfunc.GridFunction_MakeOwner(self, _fec)

    def OwnFEC(self):
        return _gridfunc.GridFunction_OwnFEC(self)

    def VectorDim(self):
        return _gridfunc.GridFunction_VectorDim(self)

    def GetValue(self, i, ip, vdim=1):
        return _gridfunc.GridFunction_GetValue(self, i, ip, vdim)

    def GetVectorValue(self, i, ip, val):
        return _gridfunc.GridFunction_GetVectorValue(self, i, ip, val)

    def GetValues(self, *args):
        return _gridfunc.GridFunction_GetValues(self, *args)

    def GetFaceValues(self, i, side, ir, vals, tr, vdim=1):
        return _gridfunc.GridFunction_GetFaceValues(self, i, side, ir, vals, tr, vdim)

    def GetVectorValues(self, *args):
        return _gridfunc.GridFunction_GetVectorValues(self, *args)

    def GetFaceVectorValues(self, i, side, ir, vals, tr):
        return _gridfunc.GridFunction_GetFaceVectorValues(self, i, side, ir, vals, tr)

    def GetValuesFrom(self, arg2):
        return _gridfunc.GridFunction_GetValuesFrom(self, arg2)

    def GetBdrValuesFrom(self, arg2):
        return _gridfunc.GridFunction_GetBdrValuesFrom(self, arg2)

    def GetVectorFieldValues(self, i, ir, vals, tr, comp=0):
        return _gridfunc.GridFunction_GetVectorFieldValues(self, i, ir, vals, tr, comp)

    def ReorderByNodes(self):
        return _gridfunc.GridFunction_ReorderByNodes(self)

    def GetNodalValues(self, *args):
        '''
        GetNodalValues(i)   ->   GetNodalValues(vector, vdim)
        GetNodalValues(i, array<dobule>, vdim)
        '''
        from .vector import Vector
        if len(args) == 1:
            vec = Vector()
            _gridfunc.GridFunction_GetNodalValues(self, vec, args[0])
            vec.thisown = 0
            return vec.GetDataArray()
        else:
            return _gridfunc.GridFunction_GetNodalValues(self, *args)



    def GetVectorFieldNodalValues(self, val, comp):
        return _gridfunc.GridFunction_GetVectorFieldNodalValues(self, val, comp)

    def ProjectVectorFieldOn(self, vec_field, comp=0):
        return _gridfunc.GridFunction_ProjectVectorFieldOn(self, vec_field, comp)

    def GetDerivative(self, comp, der_comp, der):
        return _gridfunc.GridFunction_GetDerivative(self, comp, der_comp, der)

    def GetDivergence(self, tr):
        return _gridfunc.GridFunction_GetDivergence(self, tr)

    def GetCurl(self, tr, curl):
        return _gridfunc.GridFunction_GetCurl(self, tr, curl)

    def GetGradient(self, tr, grad):
        return _gridfunc.GridFunction_GetGradient(self, tr, grad)

    def GetGradients(self, elem, ir, grad):
        return _gridfunc.GridFunction_GetGradients(self, elem, ir, grad)

    def GetVectorGradient(self, tr, grad):
        return _gridfunc.GridFunction_GetVectorGradient(self, tr, grad)

    def GetElementAverages(self, avgs):
        return _gridfunc.GridFunction_GetElementAverages(self, avgs)

    def ImposeBounds(self, *args):
        return _gridfunc.GridFunction_ImposeBounds(self, *args)

    def ProjectGridFunction(self, src):
        return _gridfunc.GridFunction_ProjectGridFunction(self, src)

    def ProjectCoefficient(self, *args):
        return _gridfunc.GridFunction_ProjectCoefficient(self, *args)

    def ProjectDiscCoefficient(self, coeff):
        return _gridfunc.GridFunction_ProjectDiscCoefficient(self, coeff)

    def ProjectBdrCoefficient(self, *args):
        return _gridfunc.GridFunction_ProjectBdrCoefficient(self, *args)

    def ProjectBdrCoefficientNormal(self, vcoeff, bdr_attr):
        return _gridfunc.GridFunction_ProjectBdrCoefficientNormal(self, vcoeff, bdr_attr)

    def ProjectBdrCoefficientTangent(self, vcoeff, bdr_attr):
        return _gridfunc.GridFunction_ProjectBdrCoefficientTangent(self, vcoeff, bdr_attr)

    def ComputeL2Error(self, *args):
        return _gridfunc.GridFunction_ComputeL2Error(self, *args)

    def ComputeH1Error(self, exsol, exgrad, ell_coef, Nu, norm_type):
        return _gridfunc.GridFunction_ComputeH1Error(self, exsol, exgrad, ell_coef, Nu, norm_type)

    def ComputeMaxError(self, *args):
        return _gridfunc.GridFunction_ComputeMaxError(self, *args)

    def ComputeW11Error(self, exsol, exgrad, norm_type, elems=None, irs=0):
        return _gridfunc.GridFunction_ComputeW11Error(self, exsol, exgrad, norm_type, elems, irs)

    def ComputeL1Error(self, *args):
        return _gridfunc.GridFunction_ComputeL1Error(self, *args)

    def ComputeLpError(self, *args):
        return _gridfunc.GridFunction_ComputeLpError(self, *args)

    def ComputeFlux(self, blfi, flux, wcoef=1, subdomain=-1):
        return _gridfunc.GridFunction_ComputeFlux(self, blfi, flux, wcoef, subdomain)

    def Assign(self, *args):
        return _gridfunc.GridFunction_Assign(self, *args)

    def Update(self):
        return _gridfunc.GridFunction_Update(self)

    def FESpace(self, *args):
        return _gridfunc.GridFunction_FESpace(self, *args)

    def SetSpace(self, f):
        return _gridfunc.GridFunction_SetSpace(self, f)

    def MakeRef(self, f, v, v_offset):
        return _gridfunc.GridFunction_MakeRef(self, f, v, v_offset)

    def Save(self, out):
        return _gridfunc.GridFunction_Save(self, out)

    def SaveVTK(self, out, field_name, ref):
        return _gridfunc.GridFunction_SaveVTK(self, out, field_name, ref)

    def SaveSTL(self, out, TimesToRefine=1):
        return _gridfunc.GridFunction_SaveSTL(self, out, TimesToRefine)
    __swig_destroy__ = _gridfunc.delete_GridFunction
    __del__ = lambda self: None

    def __init__(self, *args):
        this = _gridfunc.new_GridFunction(*args)
        try:
            self.this.append(this)
        except Exception:
            self.this = this

    def SaveToFile(self, gf_file, precision):
        return _gridfunc.GridFunction_SaveToFile(self, gf_file, precision)

    def iadd(self, c):
        return _gridfunc.GridFunction_iadd(self, c)

    def isub(self, *args):
        return _gridfunc.GridFunction_isub(self, *args)

    def imul(self, c):
        return _gridfunc.GridFunction_imul(self, c)

    def idiv(self, c):
        return _gridfunc.GridFunction_idiv(self, c)
GridFunction_swigregister = _gridfunc.GridFunction_swigregister
GridFunction_swigregister(GridFunction)


def __lshift__(*args):
    return _gridfunc.__lshift__(*args)
__lshift__ = _gridfunc.__lshift__

def ZZErrorEstimator(blfi, u, flux, error_estimates, aniso_flags=None, with_subdomains=1):
    return _gridfunc.ZZErrorEstimator(blfi, u, flux, error_estimates, aniso_flags, with_subdomains)
ZZErrorEstimator = _gridfunc.ZZErrorEstimator

def ComputeElementLpDistance(p, i, gf1, gf2):
    return _gridfunc.ComputeElementLpDistance(p, i, gf1, gf2)
ComputeElementLpDistance = _gridfunc.ComputeElementLpDistance
class ExtrudeCoefficient(coefficient.Coefficient):
    __swig_setmethods__ = {}
    for _s in [coefficient.Coefficient]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, ExtrudeCoefficient, name, value)
    __swig_getmethods__ = {}
    for _s in [coefficient.Coefficient]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, ExtrudeCoefficient, name)
    __repr__ = _swig_repr

    def __init__(self, m, s, _n):
        this = _gridfunc.new_ExtrudeCoefficient(m, s, _n)
        try:
            self.this.append(this)
        except Exception:
            self.this = this

    def Eval(self, T, ip):
        return _gridfunc.ExtrudeCoefficient_Eval(self, T, ip)
    __swig_destroy__ = _gridfunc.delete_ExtrudeCoefficient
    __del__ = lambda self: None
ExtrudeCoefficient_swigregister = _gridfunc.ExtrudeCoefficient_swigregister
ExtrudeCoefficient_swigregister(ExtrudeCoefficient)


def Extrude1DGridFunction(mesh, mesh2d, sol, ny):
    return _gridfunc.Extrude1DGridFunction(mesh, mesh2d, sol, ny)
Extrude1DGridFunction = _gridfunc.Extrude1DGridFunction

def __iadd__(self, v):
    ret = _gridfunc.GridFunction_iadd(self, v)
    ret.thisown = self.thisown
    self.thisown = 0      
    return ret
def __isub__(self, v):
    ret = _gridfunc.GridFunction_isub(self, v)
    ret.thisown = self.thisown
    self.thisown = 0      
    return ret
def __idiv__(self, v):
    ret = _gridfunc.GridFunction_idiv(self, v)
    ret.thisown = self.thisown
    self.thisown = 0
    return ret
def __imul__(self, v):
    ret = _gridfunc.GridFunction_imul(self, v)
    ret.thisown = self.thisown
    self.thisown = 0
    return ret

GridFunction.__iadd__  = __iadd__
GridFunction.__idiv__  = __idiv__
GridFunction.__isub__  = __isub__
GridFunction.__imul__  = __imul__      

# This file is compatible with both classic and new-style classes.

