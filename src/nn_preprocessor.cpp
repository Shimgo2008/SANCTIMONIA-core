#include "device_selector.hpp"
#include "nn_preprocessor.hpp"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <chrono>

namespace sanctimonia::core {

template<typename T>
struct NNPreprocessor<T>::Impl {
    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo mem_info;

    // ヘルパー関数: SessionOptions を組み立ててから返す
    static Ort::SessionOptions CreateOptions(const std::string& device) {
        Ort::SessionOptions options;

        apply_device_strategy(options, device);
        return options;
    }

    Impl(const std::string& model_path, const std::string& device) 
        : env(ORT_LOGGING_LEVEL_ERROR, "SanctimoniaGNN"),

          session(env, model_path.c_str(), CreateOptions(device)),
          mem_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) 
    {
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_outputs = session.GetOutputCount();
        output_name_strings.reserve(num_outputs);
        output_names.reserve(num_outputs);

        for (size_t i = 0; i < num_outputs; ++i) {
            auto name_ptr = session.GetOutputNameAllocated(i, allocator);
            output_name_strings.push_back(name_ptr.get());
            output_names.push_back(output_name_strings.back().c_str());
        }
    }

    std::vector<const char*> input_names = {"b", "edge_index", "edge_attr"};
    std::vector<std::string> output_name_strings;
    std::vector<const char*> output_names;
};

template<typename T>
typename NNPreprocessor<T>::ComplexMatrix 
NNPreprocessor<T>::predict(const ComplexSparseMatrix& A, const ComplexMatrix& b) {
    auto start = std::chrono::high_resolution_clock::now();

    const int64_t num_nodes = b.rows();
    const int64_t num_systems = b.cols(); // num_systems
    const int64_t num_edges = A.nonZeros();

    // 1. b [N, S] (complex) -> b_tensor [N, S, 2] (real)
    // ONNXモデルがfloat32を期待しているため、T=doubleでもfloatにキャストする
    std::vector<float> b_flat(num_nodes * num_systems * 2);
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_systems; ++j) {
            size_t idx = (i * num_systems + j) * 2;
            b_flat[idx + 0] = static_cast<float>(b(i, j).real());
            b_flat[idx + 1] = static_cast<float>(b(i, j).imag());
        }
    }
    std::vector<int64_t> b_shape = {num_nodes, num_systems, 2};

    // 2. A (Sparse) -> edge_index [2, E] & edge_attr [E, 2]
    std::vector<int64_t> edge_index_flat(num_edges * 2);
    std::vector<float> edge_attr_flat(num_edges * 2);
    
    int64_t e_idx = 0;
    // EigenのSparseMatrixはデフォルトでColMajorなので、走査順に注意
    for (int k = 0; k < A.outerSize(); ++k) {
        for (typename ComplexSparseMatrix::InnerIterator it(A, k); it; ++it) {
            // edge_index: PyTorch形式 [2, E] なので、[row1, row2... , col1, col2...] の順
            // もしくはエクスポート設定に合わせてインターリーブ ([row1, col1, row2, col2...]) に調整
            edge_index_flat[e_idx] = it.row();
            edge_index_flat[num_edges + e_idx] = it.col();
            
            edge_attr_flat[e_idx * 2 + 0] = static_cast<float>(it.value().real());
            edge_attr_flat[e_idx * 2 + 1] = static_cast<float>(it.value().imag());
            e_idx++;
        }
    }
    std::vector<int64_t> ei_shape = {2, num_edges};
    std::vector<int64_t> ea_shape = {num_edges, 2};

    // 3. Ort::Value の作成
    auto b_tensor = Ort::Value::CreateTensor<float>(pImpl->mem_info, b_flat.data(), b_flat.size(), b_shape.data(), b_shape.size());
    auto ei_tensor = Ort::Value::CreateTensor<int64_t>(pImpl->mem_info, edge_index_flat.data(), edge_index_flat.size(), ei_shape.data(), ei_shape.size());
    auto ea_tensor = Ort::Value::CreateTensor<float>(pImpl->mem_info, edge_attr_flat.data(), edge_attr_flat.size(), ea_shape.data(), ea_shape.size());

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(b_tensor));
    inputs.push_back(std::move(ei_tensor));
    inputs.push_back(std::move(ea_tensor));

    // 4. 推論実行
    auto outputs = pImpl->session.Run(
        Ort::RunOptions{nullptr}, 
        pImpl->input_names.data(), inputs.data(), inputs.size(),
        pImpl->output_names.data(), 1
    );

    // 5. 出力 [N, S, 2] -> ComplexMatrix [N, S]
    float* out_ptr = outputs.front().template GetTensorMutableData<float>();
    ComplexMatrix x0(num_nodes, num_systems);
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_systems; ++j) {
            size_t idx = (i * num_systems + j) * 2;
            x0(i, j) = ComplexT(static_cast<T>(out_ptr[idx + 0]), static_cast<T>(out_ptr[idx + 1]));
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Predict latency: " << std::chrono::duration<double>(end - start).count() << "s" << std::endl;

    return x0;
}

template<typename T>
NNPreprocessor<T>::NNPreprocessor(const std::string& model_path, const std::string& device) {
    pImpl = new Impl(model_path, device);
};

template<typename T>
NNPreprocessor<T>::~NNPreprocessor() {
    delete pImpl;
};

// Explicit instantiation
template class NNPreprocessor<double>;
template class NNPreprocessor<float>;

}