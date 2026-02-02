#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/complex.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <complex>
#include <iostream>

namespace nb = nanobind;

// Real Types
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorXd = Eigen::VectorXd;
using SparseMatrixXd = Eigen::SparseMatrix<double>;

// Complex Types
using Complex = std::complex<double>;
using RowMatrixXcd = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorXcd = Eigen::VectorXcd;
using SparseMatrixXcd = Eigen::SparseMatrix<Complex>;

// --- 例外スロー用ヘルパー ---
void throw_solver_error(const std::string& class_name, const std::string& msg, 
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

void throw_decomposition_error(const std::string& msg) {
    nb::gil_scoped_acquire acquire;

    nb::object ex_mod = nb::module_::import_("sanctimonia.types.exception");
    nb::object ex_cls = nb::borrow<nb::object>(PyObject_GetAttrString(ex_mod.ptr(), "DecompositionError"));
    if (!ex_cls) { PyErr_Clear(); return; }

    nb::object instance = ex_cls(msg);
    PyErr_SetObject(ex_cls.ptr(), instance.ptr());
    throw nb::python_error();
}

// --- 前処理設定 ---
template <typename Solver>
void configure_preconditioner(Solver& solver) {
    using Scalar = typename Solver::Scalar;
    if constexpr (std::is_same_v<typename Solver::Preconditioner, Eigen::IncompleteLUT<Scalar>>) {
        solver.preconditioner().setFillfactor(10);
        solver.preconditioner().setDroptol(1e-4);
    }
}

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

// --- マクロ定義 ---
// 実数・複素数の両方に対応するためのテンプレートマクロ
#define REGISTER_SOLVER_TEMPLATE(name, SolverType, MatrixType, VectorType) \
    m.def(name, [](Eigen::Ref<const MatrixType> A, Eigen::Ref<const VectorType> b, \
                   std::optional<VectorType> x0, double tol) { \
        nb::gil_scoped_release release; \
        return solve_impl<SolverType, MatrixType, VectorType>(A, b, x0, tol); \
    }, nb::arg("A").noconvert(), nb::arg("b").noconvert(), \
       nb::arg("x0").noconvert() = nb::none(), nb::arg("tol") = 1e-6)

#define REGISTER_SPARSE_SOLVER_TEMPLATE(name, SolverType, MatrixType, VectorType, SparseMatrixType) \
    m.def(name, [](const SparseMatrixType& A, Eigen::Ref<const VectorType> b, \
                   std::optional<VectorType> x0, double tol) { \
        nb::gil_scoped_release release; \
        return solve_impl<SolverType, SparseMatrixType, VectorType>(A, b, x0, tol); \
    }, nb::arg("A").noconvert(), nb::arg("b").noconvert(), \
       nb::arg("x0").noconvert() = nb::none(), nb::arg("tol") = 1e-6)

NB_MODULE(core, m) {
    // --- Real Solvers ---
    REGISTER_SOLVER_TEMPLATE("solve_cg", Eigen::ConjugateGradient<RowMatrixXd>, RowMatrixXd, VectorXd);
    REGISTER_SOLVER_TEMPLATE("solve_bicgstab", Eigen::BiCGSTAB<RowMatrixXd>, RowMatrixXd, VectorXd);
    REGISTER_SOLVER_TEMPLATE("solve_lscg", Eigen::LeastSquaresConjugateGradient<RowMatrixXd>, RowMatrixXd, VectorXd);

    using ILU_CG = Eigen::ConjugateGradient<SparseMatrixXd, Eigen::Lower|Eigen::Upper, Eigen::IncompleteLUT<double>>;
    using ILU_BiCG = Eigen::BiCGSTAB<SparseMatrixXd, Eigen::IncompleteLUT<double>>;
    using ILU_LSCG = Eigen::LeastSquaresConjugateGradient<SparseMatrixXd, Eigen::IncompleteLUT<double>>;

    REGISTER_SPARSE_SOLVER_TEMPLATE("solve_cg_ilu", ILU_CG, RowMatrixXd, VectorXd, SparseMatrixXd);
    REGISTER_SPARSE_SOLVER_TEMPLATE("solve_bicgstab_ilu", ILU_BiCG, RowMatrixXd, VectorXd, SparseMatrixXd);
    REGISTER_SPARSE_SOLVER_TEMPLATE("solve_lscg_ilu", ILU_LSCG, RowMatrixXd, VectorXd, SparseMatrixXd);

    // --- Complex Solvers ---
    // Python側では同じ関数名でオーバーロードとして扱う

    // CG (Hermitian Positive Definite)
    REGISTER_SOLVER_TEMPLATE("solve_cg", Eigen::ConjugateGradient<RowMatrixXcd>, RowMatrixXcd, VectorXcd);
    // BiCGSTAB (Non-Hermitian)
    REGISTER_SOLVER_TEMPLATE("solve_bicgstab", Eigen::BiCGSTAB<RowMatrixXcd>, RowMatrixXcd, VectorXcd);
    // LSCG
    REGISTER_SOLVER_TEMPLATE("solve_lscg", Eigen::LeastSquaresConjugateGradient<RowMatrixXcd>, RowMatrixXcd, VectorXcd);

    using ILU_CG_Complex = Eigen::ConjugateGradient<SparseMatrixXcd, Eigen::Lower|Eigen::Upper, Eigen::IncompleteLUT<Complex>>;
    using ILU_BiCG_Complex = Eigen::BiCGSTAB<SparseMatrixXcd, Eigen::IncompleteLUT<Complex>>;
    using ILU_LSCG_Complex = Eigen::LeastSquaresConjugateGradient<SparseMatrixXcd, Eigen::IncompleteLUT<Complex>>;

    REGISTER_SPARSE_SOLVER_TEMPLATE("solve_cg_ilu", ILU_CG_Complex, RowMatrixXcd, VectorXcd, SparseMatrixXcd);
    REGISTER_SPARSE_SOLVER_TEMPLATE("solve_bicgstab_ilu", ILU_BiCG_Complex, RowMatrixXcd, VectorXcd, SparseMatrixXcd);
    REGISTER_SPARSE_SOLVER_TEMPLATE("solve_lscg_ilu", ILU_LSCG_Complex, RowMatrixXcd, VectorXcd, SparseMatrixXcd);

    // 直接解法 (Real)
    m.def("solve_full_piv_lu", [](Eigen::Ref<const RowMatrixXd> A, Eigen::Ref<const VectorXd> b) -> VectorXd {
        return A.fullPivLu().solve(b).eval();
    }, nb::arg("A").noconvert(), nb::arg("b").noconvert());

    m.def("solve_partial_piv_lu", [](Eigen::Ref<const RowMatrixXd> A, Eigen::Ref<const VectorXd> b) -> VectorXd {
        return A.partialPivLu().solve(b).eval();
    }, nb::arg("A").noconvert(), nb::arg("b").noconvert());

    // 直接解法 (Complex)
    m.def("solve_full_piv_lu", [](Eigen::Ref<const RowMatrixXcd> A, Eigen::Ref<const VectorXcd> b) -> VectorXcd {
        return A.fullPivLu().solve(b).eval();
    }, nb::arg("A").noconvert(), nb::arg("b").noconvert());

    m.def("solve_partial_piv_lu", [](Eigen::Ref<const RowMatrixXcd> A, Eigen::Ref<const VectorXcd> b) -> VectorXcd {
        return A.partialPivLu().solve(b).eval();
    }, nb::arg("A").noconvert(), nb::arg("b").noconvert());

}
