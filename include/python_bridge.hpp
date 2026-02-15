#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <string>
#include <optional>

#include "common_types.hpp"
#include "solvers.hpp"
#include "nn_preprocessor.hpp"

namespace nb = nanobind;

namespace sanctimonia::core {

// --- Utils (Exception Helpers) ---

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

// --- Solvers ---

// --- 共通実行ロジック ---
template <typename Solver, typename MatrixType, typename VectorType>
VectorType solve_impl(const MatrixType& A, const VectorType& b, std::optional<VectorType> x0, double tol) {
    Solver solver;
    configure_preconditioner(solver);
    solver.setTolerance(tol);
    
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
       throw_decomposition_error("Decomposition failed (info=" + std::to_string(solver.info()) + ")");
    }

    VectorType x;
    if (x0.has_value()) {
        x = solver.solveWithGuess(b, *x0);
    } else {
        x = solver.solve(b);
    }
    
    if (solver.info() != Eigen::Success) {
        throw_solver_error("ConvergenceError", "Solver failed to converge", 
                           solver.iterations(), solver.error());
    }

    return x;
}

template <typename SolverType, typename MatrixType, typename VectorType>
void register_solver(nb::module_& m, const char* name) {
    m.def(name, [](Eigen::Ref<const MatrixType> A, Eigen::Ref<const VectorType> b, 
                   std::optional<VectorType> x0, double tol) {
        nb::gil_scoped_release release;
        return solve_impl<SolverType, MatrixType, VectorType>(A, b, x0, tol);
    }, nb::arg("A").noconvert(), nb::arg("b").noconvert(), 
       nb::arg("x0").noconvert() = nb::none(), nb::arg("tol") = 1e-6);
}

template <typename SolverType, typename MatrixType, typename VectorType, typename SparseMatrixType>
void register_sparse_solver(nb::module_& m, const char* name) {
    m.def(name, [](const SparseMatrixType& A, Eigen::Ref<const VectorType> b, 
                   std::optional<VectorType> x0, double tol) {
        nb::gil_scoped_release release;
        return solve_impl<SolverType, SparseMatrixType, VectorType>(A, b, x0, tol);
    }, nb::arg("A").noconvert(), nb::arg("b").noconvert(), 
       nb::arg("x0").noconvert() = nb::none(), nb::arg("tol") = 1e-6);
}

// --- NN Preprocessor ---

template <typename T>
void register_nn_preprocessor(nb::module_& m, const char* name) {
    using Class = NNPreprocessor<T>;
    nb::class_<Class>(m, name)
        .def(nb::init<const std::string&>(), nb::arg("model_path"))
        .def("predict", &Class::predict, nb::arg("A").noconvert(), nb::arg("b").noconvert());
}

}
