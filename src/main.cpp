#include "python_bridge.hpp"

using namespace sanctimonia::core;

NB_MODULE(core, m) {
    // --- Real Solvers ---
    register_solver<Eigen::ConjugateGradient<RowMatrixXd>, RowMatrixXd, VectorXd>(m, "solve_cg");
    register_solver<Eigen::BiCGSTAB<RowMatrixXd>, RowMatrixXd, VectorXd>(m, "solve_bicgstab");
    register_solver<Eigen::LeastSquaresConjugateGradient<RowMatrixXd>, RowMatrixXd, VectorXd>(m, "solve_lscg");

    using ILU_CG = Eigen::ConjugateGradient<SparseMatrixXd, Eigen::Lower|Eigen::Upper, Eigen::IncompleteLUT<double>>;
    using ILU_BiCG = Eigen::BiCGSTAB<SparseMatrixXd, Eigen::IncompleteLUT<double>>;
    using ILU_LSCG = Eigen::LeastSquaresConjugateGradient<SparseMatrixXd, Eigen::IncompleteLUT<double>>;

    register_sparse_solver<ILU_CG, RowMatrixXd, VectorXd, SparseMatrixXd>(m, "solve_cg_ilu");
    register_sparse_solver<ILU_BiCG, RowMatrixXd, VectorXd, SparseMatrixXd>(m, "solve_bicgstab_ilu");
    register_sparse_solver<ILU_LSCG, RowMatrixXd, VectorXd, SparseMatrixXd>(m, "solve_lscg_ilu");

    // --- Complex Solvers ---
    // CG (Hermitian Positive Definite)
    register_solver<Eigen::ConjugateGradient<RowMatrixXcd>, RowMatrixXcd, VectorXcd>(m, "solve_cg");
    register_solver<Eigen::BiCGSTAB<RowMatrixXcd>, RowMatrixXcd, VectorXcd>(m, "solve_bicgstab");
    register_solver<Eigen::LeastSquaresConjugateGradient<RowMatrixXcd>, RowMatrixXcd, VectorXcd>(m, "solve_lscg");

    using ILU_CG_Complex = Eigen::ConjugateGradient<SparseMatrixXcd, Eigen::Lower|Eigen::Upper, Eigen::IncompleteLUT<Complex>>;
    using ILU_BiCG_Complex = Eigen::BiCGSTAB<SparseMatrixXcd, Eigen::IncompleteLUT<Complex>>;
    using ILU_LSCG_Complex = Eigen::LeastSquaresConjugateGradient<SparseMatrixXcd, Eigen::IncompleteLUT<Complex>>;

    // Vector RHS
    register_sparse_solver<ILU_CG_Complex, RowMatrixXcd, VectorXcd, SparseMatrixXcd>(m, "solve_cg_ilu");
    register_sparse_solver<ILU_BiCG_Complex, RowMatrixXcd, VectorXcd, SparseMatrixXcd>(m, "solve_bicgstab_ilu");
    register_sparse_solver<ILU_LSCG_Complex, RowMatrixXcd, VectorXcd, SparseMatrixXcd>(m, "solve_lscg_ilu");

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

    // --- NN Preprocessor ---
    register_nn_preprocessor<double>(m, "NNPreprocessor");
}
