#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include "nn_preprocessor.hpp"

using namespace sanctimonia::core;

// 簡易クロノグラフ
struct Timer {
    std::chrono::high_resolution_clock::time_point start;
    Timer() : start(std::chrono::high_resolution_clock::now()) {}
    double elapsed() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

void run_benchmark(const std::string& label, 
                   const Eigen::SparseMatrix<double>& A, 
                   const Eigen::VectorXd& b, 
                   const Eigen::VectorXd& x0,
                   bool use_guess) {
    
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> solver;
    
    Timer t_compute;
    solver.compute(A);
    double d_compute = t_compute.elapsed();

    Timer t_solve;
    Eigen::VectorXd x;
    if (use_guess) {
        x = solver.solveWithGuess(b, x0);
    } else {
        x = solver.solve(b);
    }
    double d_solve = t_solve.elapsed();

    std::cout << std::left << std::setw(20) << label 
              << " | Iters: " << std::setw(5) << solver.iterations()
              << " | Error: " << std::setw(10) << solver.error()
              << " | Solve: " << std::fixed << std::setprecision(4) << d_solve << " ms"
              << std::endl;
}

int main() {
    // --- 準備: 物理空間を模した少し大きな行列にするのがベター ---
    int N = 1000;
    Eigen::SparseMatrix<double> A(N, N);
    // ここにポアソン方程式などのスパース行列を詰める
    for(int i=0; i<N; ++i) {
        A.insert(i,i) = 2.01;
        if(i>0) A.insert(i, i-1) = -1.0;
        if(i<N-1) A.insert(i, i+1) = -1.0;
    }
    A.makeCompressed();
    Eigen::VectorXd b = Eigen::VectorXd::Random(N);

    std::cout << "--- Sanctimonia Solver Benchmark ---" << std::endl;

    // 1. Normal Case (Guess = 0)
    run_benchmark("Normal CG", A, b, Eigen::VectorXd::Zero(N), false);

    // 2. NN Preprocessed Case
    try {
        NNPreprocessor<double> preprocessor("model_path.onnx");
        
        Timer t_nn;
        Eigen::VectorXd x0 = preprocessor.predict(b);
        double d_nn = t_nn.elapsed();
        
        std::cout << std::left << std::setw(20) << "NN Inference" 
                  << " | " << std::string(22, ' ') 
                  << " | Time : " << d_nn << " ms (Overhead)" << std::endl;

        run_benchmark("NN + CG", A, b, x0, true);
    } catch (const std::exception& e) {
        std::cerr << "NN Error: " << e.what() << std::endl;
    }

    return 0;
}