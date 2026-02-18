#include "python_bridge.hpp"

using namespace sanctimonia::core;

NB_MODULE(core, m) {
    register_stateful_solvers(m);

    register_dense_solver_function_class<CGSolver, double, RowMatrixXd, VectorXd>(m, "solve_cg");
    register_dense_solver_function_class<BiCGStabSolver, double, RowMatrixXd, VectorXd>(m, "solve_bicgstab");
    register_dense_solver_function_class<LSCGSolver, double, RowMatrixXd, VectorXd>(m, "solve_lscg");

    register_sparse_solver_function_class<ILUCGSolver, double, VectorXd, SparseMatrixXd>(m, "solve_cg_ilu");
    register_sparse_solver_function_class<ILUBiCGStabSolver, double, VectorXd, SparseMatrixXd>(m, "solve_bicgstab_ilu");
    register_sparse_solver_function_class<ILULSCGSolver, double, VectorXd, SparseMatrixXd>(m, "solve_lscg_ilu");

    using ComplexCG = Eigen::ConjugateGradient<RowMatrixXcd>;
    using ComplexBiCG = Eigen::BiCGSTAB<RowMatrixXcd>;
    using ComplexLSCG = Eigen::LeastSquaresConjugateGradient<RowMatrixXcd>;

    register_dense_solver_function_type<ComplexCG, Complex, RowMatrixXcd, VectorXcd>(m, "solve_cg");
    register_dense_solver_function_type<ComplexBiCG, Complex, RowMatrixXcd, VectorXcd>(m, "solve_bicgstab");
    register_dense_solver_function_type<ComplexLSCG, Complex, RowMatrixXcd, VectorXcd>(m, "solve_lscg");

    using ComplexILUCG = Eigen::ConjugateGradient<SparseMatrixXcd, Eigen::Lower | Eigen::Upper, Eigen::IncompleteLUT<Complex>>;
    using ComplexILUBiCG = Eigen::BiCGSTAB<SparseMatrixXcd, Eigen::IncompleteLUT<Complex>>;
    using ComplexILULSCG = Eigen::LeastSquaresConjugateGradient<SparseMatrixXcd, Eigen::IncompleteLUT<Complex>>;

    register_sparse_solver_function_type<ComplexILUCG, Complex, VectorXcd, SparseMatrixXcd>(m, "solve_cg_ilu");
    register_sparse_solver_function_type<ComplexILUBiCG, Complex, VectorXcd, SparseMatrixXcd>(m, "solve_bicgstab_ilu");
    register_sparse_solver_function_type<ComplexILULSCG, Complex, VectorXcd, SparseMatrixXcd>(m, "solve_lscg_ilu");

    register_lu_functions<double, RowMatrixXd, VectorXd>(m);
    register_lu_functions<Complex, RowMatrixXcd, VectorXcd>(m);

    // --- NN Preprocessor ---
    register_nn_preprocessor<double>(m, "NNPreprocessor");
}
