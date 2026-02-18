#pragma once
#include "common_types.hpp"
#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace sanctimonia::core {

struct SolverConfig {
    int num_threads = 0;
    std::string device = "cpu";
    double tol = 1e-6;
    int max_iterations = 0;
};

// --- 前処理設定 ---
template <typename Solver>
void configure_preconditioner(Solver& solver) {
    using Scalar = typename Solver::Scalar;
    if constexpr (std::is_same_v<typename Solver::Preconditioner, Eigen::IncompleteLUT<Scalar>>) {
        solver.preconditioner().setFillfactor(10);
        solver.preconditioner().setDroptol(1e-4);
    }
}

inline void configure_threads(int num_threads) {
    if (num_threads > 0) {
        Eigen::setNbThreads(num_threads);
    }
}

template <typename SolverType, typename MatrixType, typename VectorType>
VectorType solve_with_state(
    const MatrixType& A,
    const VectorType& b,
    std::optional<VectorType> x0,
    const SolverConfig& config,
    std::optional<double> tol_override = std::nullopt
) {
    configure_threads(config.num_threads);

    SolverType solver;
    configure_preconditioner(solver);
    solver.setTolerance(tol_override.value_or(config.tol));
    if (config.max_iterations > 0) {
        solver.setMaxIterations(config.max_iterations);
    }

    solver.compute(A);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Decomposition failed");
    }

    VectorType x;
    if (x0.has_value()) {
        x = solver.solveWithGuess(b, *x0);
    } else {
        x = solver.solve(b);
    }

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Solver failed to converge");
    }

    return x;
}

class SolverBase {
public:
    explicit SolverBase(SolverConfig config = {}) : config_(std::move(config)) {}
    SolverBase(int num_threads, std::string device, double tol)
        : config_({num_threads, std::move(device), tol, 0}) {}
    SolverBase(int num_threads, std::string device, double tol, int max_iterations)
        : config_({num_threads, std::move(device), tol, max_iterations}) {}
    virtual ~SolverBase() = default;

    void set_num_threads(int num_threads) { config_.num_threads = num_threads; }
    int num_threads() const { return config_.num_threads; }

    void set_device(const std::string& device) { config_.device = device; }
    const std::string& device() const { return config_.device; }

    void set_tol(double tol) { config_.tol = tol; }
    double tol() const { return config_.tol; }

    void set_max_iterations(int max_iterations) { config_.max_iterations = max_iterations; }
    int max_iterations() const { return config_.max_iterations; }

    virtual VectorXd solve(
        const RowMatrixXd& A,
        const VectorXd& b,
        std::optional<VectorXd> x0 = std::nullopt,
        std::optional<double> tol_override = std::nullopt
    ) const = 0;

protected:
    const SolverConfig& config() const { return config_; }

private:
    SolverConfig config_;
};

template <typename SolverType>
class DenseIterativeSolver : public SolverBase {
public:
    using SolverBase::SolverBase;

    VectorXd solve(
        const RowMatrixXd& A,
        const VectorXd& b,
        std::optional<VectorXd> x0 = std::nullopt,
        std::optional<double> tol_override = std::nullopt
    ) const override {
        return solve_with_state<SolverType, RowMatrixXd, VectorXd>(A, b, x0, config(), tol_override);
    }
};


template <typename SolverType>
class SparseCapableIterativeSolver : public SolverBase {
public:
    using SolverBase::SolverBase;

    VectorXd solve_sparse(
        const SparseMatrixXd& A,
        const VectorXd& b,
        std::optional<VectorXd> x0 = std::nullopt,
        std::optional<double> tol_override = std::nullopt
    ) const {
        return solve_with_state<SolverType, SparseMatrixXd, VectorXd>(A, b, x0, config(), tol_override);
    }

    VectorXd solve(
        const RowMatrixXd& A,
        const VectorXd& b,
        std::optional<VectorXd> x0 = std::nullopt,
        std::optional<double> tol_override = std::nullopt
    ) const override {
        SparseMatrixXd sparse_A = A.sparseView();
        return solve_with_state<SolverType, SparseMatrixXd, VectorXd>(sparse_A, b, x0, config(), tol_override);
    }
};


using CGSolver = DenseIterativeSolver<Eigen::ConjugateGradient<RowMatrixXd>>;
using BiCGStabSolver = DenseIterativeSolver<Eigen::BiCGSTAB<RowMatrixXd>>;
using LSCGSolver = DenseIterativeSolver<Eigen::LeastSquaresConjugateGradient<RowMatrixXd>>;


using ILUCGSolver = SparseCapableIterativeSolver<Eigen::ConjugateGradient<SparseMatrixXd, Eigen::Lower | Eigen::Upper, Eigen::IncompleteLUT<double>>>;
using ILUBiCGStabSolver = SparseCapableIterativeSolver<Eigen::BiCGSTAB<SparseMatrixXd, Eigen::IncompleteLUT<double>>>;
using ILULSCGSolver = SparseCapableIterativeSolver<Eigen::LeastSquaresConjugateGradient<SparseMatrixXd, Eigen::IncompleteLUT<double>>>;

}
