#pragma once
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

namespace sanctimonia {

inline void apply_device_strategy(Ort::SessionOptions& options) {
    // 利用可能なExecution Providerをすべて取得
    auto providers = Ort::GetAvailableProviders();
    
    // CUDAProviderがあるかチェック
    auto it = std::find(providers.begin(), providers.end(), "CUDAExecutionProvider");

    if (it != providers.end()) {
        try {
            // CUDAの設定（device_id = 0）
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = 0;
            
            // 重要: arena_extend_strategy を設定するとメモリ効率が良くなることがあります
            cuda_options.arena_extend_strategy = 0; 

            options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "[Sanctimonia] Found CUDA. Execution Provider: CUDA" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Sanctimonia] Failed to load CUDA: " << e.what() << ". Falling back to CPU." << std::endl;
        }
    } else {
        std::cout << "[Sanctimonia] CUDA not found. Execution Provider: CPU" << std::endl;
    }
}

}