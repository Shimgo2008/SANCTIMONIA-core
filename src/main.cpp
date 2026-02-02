#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/string.h>

#include "common_types.hpp"
#include "solvers.hpp"
#include "nn_preprocessor.hpp"

namespace nb = nanobind;
using namespace sanctimonia::core;

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
