#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <complex>
#include <string>
#include <stdexcept>

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

    NNPreprocessor(const std::string& model_path, const std::string& device = "auto");
    ~NNPreprocessor();

    // A と b を受け取り、初期解 x0 を複素行列で返す
    ComplexMatrix predict(const ComplexSparseMatrix& A, const ComplexMatrix& b);

    template<typename OutputType>
    OutputType predict(const ComplexSparseMatrix& A, const ComplexMatrix& b) {
        return convert_output<OutputType>(predict(A, b));
    }

private:
    template<typename OutputType>
    static OutputType convert_output(const ComplexMatrix& x0) {
        using OutputScalar = typename OutputType::Scalar;

        if constexpr (OutputType::ColsAtCompileTime == 1) {
            if (x0.cols() != 1) {
                throw std::invalid_argument("NNPreprocessor::predict<OutputType>: output has multiple systems; vector output requires cols == 1");
            }
            return x0.col(0).template cast<OutputScalar>();
        }

        return x0.template cast<OutputScalar>();
    }

    struct Impl;
    Impl* pImpl;
};

}