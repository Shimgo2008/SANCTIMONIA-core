#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <string>
#include <optional>

namespace nb = nanobind;

namespace sanctimonia::core {

// --- 例外スロー用ヘルパー ---
inline void throw_solver_error(const std::string& class_name, const std::string& msg, 
                        std::optional<int> iterations = std::nullopt, 
                        std::optional<double> error = std::nullopt) {
    nb::gil_scoped_acquire acquire;

    nb::handle ex_mod = nb::module_::import_("sanctimonia.types.exception");
    nb::object ex_cls = ex_mod.attr(class_name.c_str());
    
    nb::object instance;
    if (iterations && error) {
        instance = ex_cls(msg, *iterations, *error);
    } else {
        instance = ex_cls(msg);
    }

    PyErr_SetObject(ex_cls.ptr(), instance.ptr());
    throw nb::python_error();
}

inline void throw_decomposition_error(const std::string& msg) {
    nb::gil_scoped_acquire acquire;

    nb::object ex_mod = nb::module_::import_("sanctimonia.types.exception");
    nb::object ex_cls = nb::borrow<nb::object>(PyObject_GetAttrString(ex_mod.ptr(), "DecompositionError"));
    if (!ex_cls) { PyErr_Clear(); return; }

    nb::object instance = ex_cls(msg);
    PyErr_SetObject(ex_cls.ptr(), instance.ptr());
    throw nb::python_error();
}

}
