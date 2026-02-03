#pragma once
#include "common_types.hpp"
#include "utils.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <Eigen/IterativeLinearSolvers>

namespace sanctimonia::core {

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

}
