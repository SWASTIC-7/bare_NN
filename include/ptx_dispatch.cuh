#pragma once

#include <cuda.h>

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "cuda_utils.cuh"

namespace bare_nn {

class PtxKernelLibrary {
public:
    explicit PtxKernelLibrary(std::string ptx_root = "ptx/")
        : ptx_root_(std::move(ptx_root)), initialized_(false) {}

    ~PtxKernelLibrary() {
        for (auto& kv : modules_) {
            cuModuleUnload(kv.second);
        }
    }

    CUfunction getFunction(const char* ptx_file, const char* kernel_name) {
        ensureInitialized();
        CUmodule module = loadModule(ptx_file);
        CUfunction fn;
        CU_CHECK(cuModuleGetFunction(&fn, module, kernel_name));
        return fn;
    }

    void launch1D(
        const char* ptx_file,
        const char* kernel_name,
        int n,
        int block_size,
        void** args,
        CUstream stream = nullptr,
        unsigned int shared_mem_bytes = 0) {
        if (n <= 0) {
            return;
        }
        CUfunction fn = getFunction(ptx_file, kernel_name);
        const int grid = calcGridSize(n, block_size);
        CU_CHECK(cuLaunchKernel(
            fn,
            grid,
            1,
            1,
            block_size,
            1,
            1,
            shared_mem_bytes,
            stream,
            args,
            nullptr));
    }

    void launch2D(
        const char* ptx_file,
        const char* kernel_name,
        dim3 grid,
        dim3 block,
        void** args,
        CUstream stream = nullptr,
        unsigned int shared_mem_bytes = 0) {
        CUfunction fn = getFunction(ptx_file, kernel_name);
        CU_CHECK(cuLaunchKernel(
            fn,
            grid.x,
            grid.y,
            grid.z,
            block.x,
            block.y,
            block.z,
            shared_mem_bytes,
            stream,
            args,
            nullptr));
    }

private:
    void ensureInitialized() {
        if (initialized_) {
            return;
        }
        CU_CHECK(cuInit(0));
        initialized_ = true;
    }

    CUmodule loadModule(const char* ptx_file) {
        auto it = modules_.find(ptx_file);
        if (it != modules_.end()) {
            return it->second;
        }

        const std::string full_path = ptx_root_ + ptx_file;
        CUmodule module;

        std::ifstream fin(full_path);
        if (!fin) {
            fprintf(stderr, "Failed to open PTX file: %s\n", full_path.c_str());
            exit(EXIT_FAILURE);
        }

        std::string ptx_source((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());

        std::vector<char> error_log(8192, 0);
        std::vector<char> info_log(8192, 0);
        unsigned int error_log_size = static_cast<unsigned int>(error_log.size());
        unsigned int info_log_size = static_cast<unsigned int>(info_log.size());

        CUjit_option options[] = {
            CU_JIT_ERROR_LOG_BUFFER,
            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_INFO_LOG_BUFFER,
            CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_TARGET_FROM_CUCONTEXT
        };

        void* option_values[] = {
            error_log.data(),
            reinterpret_cast<void*>(static_cast<uintptr_t>(error_log_size)),
            info_log.data(),
            reinterpret_cast<void*>(static_cast<uintptr_t>(info_log_size)),
            nullptr
        };

        CUresult load_res = cuModuleLoadDataEx(
            &module,
            ptx_source.c_str(),
            sizeof(options) / sizeof(options[0]),
            options,
            option_values);

        if (load_res != CUDA_SUCCESS) {
            const char* msg = nullptr;
            cuGetErrorString(load_res, &msg);
            fprintf(stderr, "CUDA Driver Error loading PTX %s: %s\n", full_path.c_str(), msg ? msg : "unknown");
            if (!error_log.empty() && error_log[0] != '\0') {
                fprintf(stderr, "PTX JIT error log:\n%s\n", error_log.data());
            }
            if (!info_log.empty() && info_log[0] != '\0') {
                fprintf(stderr, "PTX JIT info log:\n%s\n", info_log.data());
            }
            exit(EXIT_FAILURE);
        }

        modules_[ptx_file] = module;
        return module;
    }

    std::string ptx_root_;
    std::unordered_map<std::string, CUmodule> modules_;
    bool initialized_;
};

}  // namespace bare_nn
