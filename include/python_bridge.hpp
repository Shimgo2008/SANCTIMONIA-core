#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <string>
#include <optional>
#include <stdexcept>

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
template <typename Scalar>
using DenseArray2D = nb::ndarray<Scalar, nb::ndim<2>, nb::c_contig>;

template <typename Scalar>
using DenseArray1D = nb::ndarray<Scalar, nb::ndim<1>, nb::c_contig>;

template <typename Scalar>
std::optional<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> to_optional_vector(
    const std::optional<DenseArray1D<Scalar>>& x0,
    Eigen::Index expected_size
) {
    if (!x0.has_value()) {
        return std::nullopt;
    }

    const auto x0_size = static_cast<Eigen::Index>(x0->shape(0));
    if (x0_size != expected_size) {
        throw std::invalid_argument("x0 size does not match b size");
    }

    Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> x0_map(x0->data(), expected_size);
    return Eigen::Matrix<Scalar, Eigen::Dynamic, 1>(x0_map);
}

template <typename Scalar, typename SolverClass, typename MatrixType, typename VectorType>
nb::class_<SolverClass, SolverBase> bind_dense_solver_class(nb::module_& m, const char* name) {
    return nb::class_<SolverClass, SolverBase>(m, name)
           .def(nb::init<int, std::string, double, int>(),
             nb::arg("num_threads") = 0,
             nb::arg("device") = "cpu",
               nb::arg("tol") = 1e-6,
               nb::arg("max_iterations") = 0)
        .def("solve", [](const SolverClass& self,
                          const DenseArray2D<Scalar>& A,
                          const DenseArray1D<Scalar>& b,
                          std::optional<DenseArray1D<Scalar>> x0,
                          std::optional<double> tol) {
            nb::gil_scoped_release release;

            const auto rows = static_cast<Eigen::Index>(A.shape(0));
            const auto cols = static_cast<Eigen::Index>(A.shape(1));
            const auto b_size = static_cast<Eigen::Index>(b.shape(0));

            if (rows != cols) {
                throw std::invalid_argument("A must be a square matrix");
            }
            if (rows != b_size) {
                throw std::invalid_argument("A.shape[0] must match b.shape[0]");
            }

            Eigen::Map<const MatrixType> A_map(A.data(), rows, cols);
            Eigen::Map<const VectorType> b_map(b.data(), b_size);

            auto x0_vec = to_optional_vector<Scalar>(x0, b_size);

            try {
                return self.solve(A_map, b_map, x0_vec, tol);
            } catch (const std::runtime_error& e) {
                const std::string msg = e.what();
                if (msg.find("converge") != std::string::npos) {
                    throw_solver_error("ConvergenceError", msg);
                }
                throw_decomposition_error(msg);
                return VectorType{};
            }
        },
        nb::arg("A").noconvert(),
        nb::arg("b").noconvert(),
        nb::arg("x0").noconvert() = nb::none(),
        nb::arg("tol") = nb::none());
}

template <typename SolverClass>
void bind_sparse_solver_method(nb::class_<SolverClass, SolverBase>& cls) {
    cls.def("solve_sparse", [](const SolverClass& self,
                                const SparseMatrixXd& A,
                                const DenseArray1D<double>& b,
                                std::optional<DenseArray1D<double>> x0,
                                std::optional<double> tol) {
        nb::gil_scoped_release release;

        const auto b_size = static_cast<Eigen::Index>(b.shape(0));
        if (A.rows() != b_size) {
            throw std::invalid_argument("A.shape[0] must match b.shape[0]");
        }

        Eigen::Map<const VectorXd> b_map(b.data(), b_size);
        auto x0_vec = to_optional_vector<double>(x0, b_size);

        try {
            return self.solve_sparse(A, b_map, x0_vec, tol);
        } catch (const std::runtime_error& e) {
            const std::string msg = e.what();
            if (msg.find("converge") != std::string::npos) {
                throw_solver_error("ConvergenceError", msg);
            }
            throw_decomposition_error(msg);
            return VectorXd{};
        }
    },
    nb::arg("A").noconvert(),
    nb::arg("b").noconvert(),
    nb::arg("x0").noconvert() = nb::none(),
    nb::arg("tol") = nb::none());
}

inline void register_stateful_solvers(nb::module_& m) {
    nb::class_<SolverBase>(m, "SolverBase")
        .def("set_num_threads", &SolverBase::set_num_threads)
        .def("num_threads", &SolverBase::num_threads)
        .def("set_device", &SolverBase::set_device)
        .def("device", &SolverBase::device)
        .def("set_tol", &SolverBase::set_tol)
        .def("tol", &SolverBase::tol)
        .def("set_max_iterations", &SolverBase::set_max_iterations)
        .def("max_iterations", &SolverBase::max_iterations);

    auto cg = bind_dense_solver_class<double, CGSolver, RowMatrixXd, VectorXd>(m, "CGSolverCore");
    bind_sparse_solver_method(cg);

    auto bicg = bind_dense_solver_class<double, BiCGStabSolver, RowMatrixXd, VectorXd>(m, "BiCGStabSolverCore");
    bind_sparse_solver_method(bicg);

    auto lscg = bind_dense_solver_class<double, LSCGSolver, RowMatrixXd, VectorXd>(m, "LSCGSolverCore");
    bind_sparse_solver_method(lscg);

    auto ilu_cg = nb::class_<ILUCGSolver, SolverBase>(m, "ILUCGSolverCore")
           .def(nb::init<int, std::string, double, int>(),
             nb::arg("num_threads") = 0,
             nb::arg("device") = "cpu",
               nb::arg("tol") = 1e-6,
               nb::arg("max_iterations") = 0)
        .def("solve", [](const ILUCGSolver& self,
                          const DenseArray2D<double>& A,
                          const DenseArray1D<double>& b,
                          std::optional<DenseArray1D<double>> x0,
                          std::optional<double> tol) {
            nb::gil_scoped_release release;

            const auto rows = static_cast<Eigen::Index>(A.shape(0));
            const auto cols = static_cast<Eigen::Index>(A.shape(1));
            const auto b_size = static_cast<Eigen::Index>(b.shape(0));

            if (rows != cols) {
                throw std::invalid_argument("A must be a square matrix");
            }
            if (rows != b_size) {
                throw std::invalid_argument("A.shape[0] must match b.shape[0]");
            }

            Eigen::Map<const RowMatrixXd> A_map(A.data(), rows, cols);
            Eigen::Map<const VectorXd> b_map(b.data(), b_size);
            auto x0_vec = to_optional_vector<double>(x0, b_size);
            return self.solve(A_map, b_map, x0_vec, tol);
        },
        nb::arg("A").noconvert(),
        nb::arg("b").noconvert(),
        nb::arg("x0").noconvert() = nb::none(),
        nb::arg("tol") = nb::none());
    bind_sparse_solver_method(ilu_cg);

    auto ilu_bicg = nb::class_<ILUBiCGStabSolver, SolverBase>(m, "ILUBiCGStabSolverCore")
           .def(nb::init<int, std::string, double, int>(),
             nb::arg("num_threads") = 0,
             nb::arg("device") = "cpu",
               nb::arg("tol") = 1e-6,
               nb::arg("max_iterations") = 0)
        .def("solve", [](const ILUBiCGStabSolver& self,
                          const DenseArray2D<double>& A,
                          const DenseArray1D<double>& b,
                          std::optional<DenseArray1D<double>> x0,
                          std::optional<double> tol) {
            nb::gil_scoped_release release;

            const auto rows = static_cast<Eigen::Index>(A.shape(0));
            const auto cols = static_cast<Eigen::Index>(A.shape(1));
            const auto b_size = static_cast<Eigen::Index>(b.shape(0));

            if (rows != cols) {
                throw std::invalid_argument("A must be a square matrix");
            }
            if (rows != b_size) {
                throw std::invalid_argument("A.shape[0] must match b.shape[0]");
            }

            Eigen::Map<const RowMatrixXd> A_map(A.data(), rows, cols);
            Eigen::Map<const VectorXd> b_map(b.data(), b_size);
            auto x0_vec = to_optional_vector<double>(x0, b_size);
            return self.solve(A_map, b_map, x0_vec, tol);
        },
        nb::arg("A").noconvert(),
        nb::arg("b").noconvert(),
        nb::arg("x0").noconvert() = nb::none(),
        nb::arg("tol") = nb::none());
    bind_sparse_solver_method(ilu_bicg);

    auto ilu_lscg = nb::class_<ILULSCGSolver, SolverBase>(m, "ILULSCGSolverCore")
           .def(nb::init<int, std::string, double, int>(),
             nb::arg("num_threads") = 0,
             nb::arg("device") = "cpu",
               nb::arg("tol") = 1e-6,
               nb::arg("max_iterations") = 0)
        .def("solve", [](const ILULSCGSolver& self,
                          const DenseArray2D<double>& A,
                          const DenseArray1D<double>& b,
                          std::optional<DenseArray1D<double>> x0,
                          std::optional<double> tol) {
            nb::gil_scoped_release release;

            const auto rows = static_cast<Eigen::Index>(A.shape(0));
            const auto cols = static_cast<Eigen::Index>(A.shape(1));
            const auto b_size = static_cast<Eigen::Index>(b.shape(0));

            if (rows != cols) {
                throw std::invalid_argument("A must be a square matrix");
            }
            if (rows != b_size) {
                throw std::invalid_argument("A.shape[0] must match b.shape[0]");
            }

            Eigen::Map<const RowMatrixXd> A_map(A.data(), rows, cols);
            Eigen::Map<const VectorXd> b_map(b.data(), b_size);
            auto x0_vec = to_optional_vector<double>(x0, b_size);
            return self.solve(A_map, b_map, x0_vec, tol);
        },
        nb::arg("A").noconvert(),
        nb::arg("b").noconvert(),
        nb::arg("x0").noconvert() = nb::none(),
        nb::arg("tol") = nb::none());
    bind_sparse_solver_method(ilu_lscg);
}

template <typename Scalar, typename MatrixType, typename VectorType, typename SolveFn>
VectorType solve_dense_mapped(
    const DenseArray2D<Scalar>& A,
    const DenseArray1D<Scalar>& b,
    const std::optional<DenseArray1D<Scalar>>& x0,
    SolveFn&& solve_fn
) {
    const auto rows = static_cast<Eigen::Index>(A.shape(0));
    const auto cols = static_cast<Eigen::Index>(A.shape(1));
    const auto b_size = static_cast<Eigen::Index>(b.shape(0));
    if (rows != cols || rows != b_size) {
        throw std::invalid_argument("A and b shape mismatch");
    }

    Eigen::Map<const MatrixType> A_map(A.data(), rows, cols);
    Eigen::Map<const VectorType> b_map(b.data(), b_size);
    auto x0_vec = to_optional_vector<Scalar>(x0, b_size);
    return solve_fn(A_map, b_map, x0_vec);
}

template <typename Scalar, typename VectorType, typename SparseMatrixType, typename SolveFn>
VectorType solve_sparse_mapped(
    const SparseMatrixType& A,
    const DenseArray1D<Scalar>& b,
    const std::optional<DenseArray1D<Scalar>>& x0,
    SolveFn&& solve_fn
) {
    const auto b_size = static_cast<Eigen::Index>(b.shape(0));
    if (A.rows() != b_size) {
        throw std::invalid_argument("A and b shape mismatch");
    }

    Eigen::Map<const VectorType> b_map(b.data(), b_size);
    auto x0_vec = to_optional_vector<Scalar>(x0, b_size);
    return solve_fn(A, b_map, x0_vec);
}

template <typename SolverClass, typename Scalar, typename MatrixType, typename VectorType>
void register_dense_solver_function_class(nb::module_& m, const char* name) {
    m.def(name,
          [](DenseArray2D<Scalar> A,
             DenseArray1D<Scalar> b,
             std::optional<DenseArray1D<Scalar>> x0,
             double tol) {
              return solve_dense_mapped<Scalar, MatrixType, VectorType>(A, b, x0,
                  [&](const MatrixType& A_map, const VectorType& b_map, std::optional<VectorType> x0_vec) {
                      SolverClass solver(0, "cpu", tol);
                      return solver.solve(A_map, b_map, x0_vec, tol);
                  });
          },
          nb::arg("A").noconvert(), nb::arg("b").noconvert(), nb::arg("x0").noconvert() = nb::none(), nb::arg("tol") = 1e-6);
}

template <typename SolverType, typename Scalar, typename MatrixType, typename VectorType>
void register_dense_solver_function_type(nb::module_& m, const char* name) {
    m.def(name,
          [](DenseArray2D<Scalar> A,
             DenseArray1D<Scalar> b,
             std::optional<DenseArray1D<Scalar>> x0,
             double tol) {
              return solve_dense_mapped<Scalar, MatrixType, VectorType>(A, b, x0,
                  [&](const MatrixType& A_map, const VectorType& b_map, std::optional<VectorType> x0_vec) {
                      return solve_with_state<SolverType, MatrixType, VectorType>(A_map, b_map, x0_vec, SolverConfig{0, "cpu", tol}, tol);
                  });
          },
          nb::arg("A").noconvert(), nb::arg("b").noconvert(), nb::arg("x0").noconvert() = nb::none(), nb::arg("tol") = 1e-6);
}

template <typename SolverClass, typename Scalar, typename VectorType, typename SparseMatrixType>
void register_sparse_solver_function_class(nb::module_& m, const char* name) {
    m.def(name,
          [](const SparseMatrixType& A,
             DenseArray1D<Scalar> b,
             std::optional<DenseArray1D<Scalar>> x0,
             double tol) {
              return solve_sparse_mapped<Scalar, VectorType, SparseMatrixType>(A, b, x0,
                  [&](const SparseMatrixType& A_ref, const VectorType& b_map, std::optional<VectorType> x0_vec) {
                      SolverClass solver(0, "cpu", tol);
                      return solver.solve_sparse(A_ref, b_map, x0_vec, tol);
                  });
          },
          nb::arg("A").noconvert(), nb::arg("b").noconvert(), nb::arg("x0").noconvert() = nb::none(), nb::arg("tol") = 1e-6);
}

template <typename SolverType, typename Scalar, typename VectorType, typename SparseMatrixType>
void register_sparse_solver_function_type(nb::module_& m, const char* name) {
    m.def(name,
          [](const SparseMatrixType& A,
             DenseArray1D<Scalar> b,
             std::optional<DenseArray1D<Scalar>> x0,
             double tol) {
              return solve_sparse_mapped<Scalar, VectorType, SparseMatrixType>(A, b, x0,
                  [&](const SparseMatrixType& A_ref, const VectorType& b_map, std::optional<VectorType> x0_vec) {
                      return solve_with_state<SolverType, SparseMatrixType, VectorType>(A_ref, b_map, x0_vec, SolverConfig{0, "cpu", tol}, tol);
                  });
          },
          nb::arg("A").noconvert(), nb::arg("b").noconvert(), nb::arg("x0").noconvert() = nb::none(), nb::arg("tol") = 1e-6);
}

template <typename Scalar, typename MatrixType, typename VectorType>
void register_lu_functions(nb::module_& m) {
    m.def("solve_full_piv_lu", [](DenseArray2D<Scalar> A, DenseArray1D<Scalar> b) -> VectorType {
        return solve_dense_mapped<Scalar, MatrixType, VectorType>(A, b, std::nullopt,
            [&](const MatrixType& A_map, const VectorType& b_map, std::optional<VectorType>) {
                return A_map.fullPivLu().solve(b_map).eval();
            });
    }, nb::arg("A").noconvert(), nb::arg("b").noconvert());

    m.def("solve_partial_piv_lu", [](DenseArray2D<Scalar> A, DenseArray1D<Scalar> b) -> VectorType {
        return solve_dense_mapped<Scalar, MatrixType, VectorType>(A, b, std::nullopt,
            [&](const MatrixType& A_map, const VectorType& b_map, std::optional<VectorType>) {
                return A_map.partialPivLu().solve(b_map).eval();
            });
    }, nb::arg("A").noconvert(), nb::arg("b").noconvert());
}

// --- NN Preprocessor ---

template <typename T>
void register_nn_preprocessor(nb::module_& m, const char* name) {
    using Class = NNPreprocessor<T>;
    using PredictMethod = typename Class::ComplexMatrix (Class::*)(
        const typename Class::ComplexSparseMatrix&,
        const typename Class::ComplexMatrix&);

    nb::class_<Class>(m, name)
        .def(nb::init<const std::string&, const std::string&>(),
             nb::arg("model_path"),
             nb::arg("device") = "auto")
        .def("predict", static_cast<PredictMethod>(&Class::predict), nb::arg("A").noconvert(), nb::arg("b").noconvert());
}

}
