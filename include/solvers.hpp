#pragma once
#include "common_types.hpp"
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
}
