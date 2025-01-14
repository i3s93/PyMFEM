%module(package="mfem._ser", directors="1") nonlininteg
%{
#include "fem/nonlininteg.hpp"
#include "fem/linearform.hpp"  
#include "../common/pycoefficient.hpp"
#include "pyoperator.hpp"
#include "fem/ceed/interface/operator.hpp"
using namespace mfem;  
%}
/*
%init %{
import_array();
%}
*/
/*
%exception {
    try { $action }
    catch (Swig::DirectorException &e) { SWIG_fail; }    
    //catch (...){
    //  SWIG_fail;
    //}
    //    catch (Swig::DirectorMethodException &e) { SWIG_fail; }
    //    catch (std::exception &e) { SWIG_fail; }    
}
*/
%include "exception.i"
%import "vector.i"
%import "operators.i"
%import "fespace.i"
%import "eltrans.i"
%import "../common/exception_director.i"

%feature("director") mfem::NonlinearFormIntegrator;

%include "fem/nonlininteg.hpp"
