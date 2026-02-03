#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <complex>

namespace sanctimonia::core {

// Real Types
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorXd = Eigen::VectorXd;
using SparseMatrixXd = Eigen::SparseMatrix<double>;

// Complex Types
using Complex = std::complex<double>;
using RowMatrixXcd = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorXcd = Eigen::VectorXcd;
using SparseMatrixXcd = Eigen::SparseMatrix<Complex>;

}
