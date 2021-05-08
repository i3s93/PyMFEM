# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.1
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _cpointers
else:
    import _cpointers

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

_swig_new_instance_method = _cpointers.SWIG_PyInstanceMethod_New
_swig_new_static_method = _cpointers.SWIG_PyStaticMethod_New

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


class intp(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _cpointers.intp_swiginit(self, _cpointers.new_intp())
    __swig_destroy__ = _cpointers.delete_intp

    def assign(self, value):
        return _cpointers.intp_assign(self, value)
    assign = _swig_new_instance_method(_cpointers.intp_assign)

    def value(self):
        return _cpointers.intp_value(self)
    value = _swig_new_instance_method(_cpointers.intp_value)

    def cast(self):
        return _cpointers.intp_cast(self)
    cast = _swig_new_instance_method(_cpointers.intp_cast)

    @staticmethod
    def frompointer(t):
        return _cpointers.intp_frompointer(t)
    frompointer = _swig_new_static_method(_cpointers.intp_frompointer)

# Register intp in _cpointers:
_cpointers.intp_swigregister(intp)

def intp_frompointer(t):
    return _cpointers.intp_frompointer(t)
intp_frompointer = _cpointers.intp_frompointer

class doublep(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _cpointers.doublep_swiginit(self, _cpointers.new_doublep())
    __swig_destroy__ = _cpointers.delete_doublep

    def assign(self, value):
        return _cpointers.doublep_assign(self, value)
    assign = _swig_new_instance_method(_cpointers.doublep_assign)

    def value(self):
        return _cpointers.doublep_value(self)
    value = _swig_new_instance_method(_cpointers.doublep_value)

    def cast(self):
        return _cpointers.doublep_cast(self)
    cast = _swig_new_instance_method(_cpointers.doublep_cast)

    @staticmethod
    def frompointer(t):
        return _cpointers.doublep_frompointer(t)
    frompointer = _swig_new_static_method(_cpointers.doublep_frompointer)

# Register doublep in _cpointers:
_cpointers.doublep_swigregister(doublep)

def doublep_frompointer(t):
    return _cpointers.doublep_frompointer(t)
doublep_frompointer = _cpointers.doublep_frompointer


