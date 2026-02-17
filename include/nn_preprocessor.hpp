#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <complex>
#include <string>

namespace sanctimonia::core {
    
template<typename T>
class NNPreprocessor {
    public:
    using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    // 複素数型の定義
    using ComplexT = std::complex<T>;
    using ComplexVector = Eigen::Matrix<ComplexT, Eigen::Dynamic, 1>;
    using ComplexMatrix = Eigen::Matrix<ComplexT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using ComplexSparseMatrix = Eigen::SparseMatrix<ComplexT>;

    NNPreprocessor(const std::string& model_path);
    ~NNPreprocessor();

    // A と b を受け取り、初期解 x0 を複素行列で返す
    ComplexMatrix predict(const ComplexSparseMatrix& A, const ComplexMatrix& b);

private:
    struct Impl;
    Impl* pImpl;
};

}