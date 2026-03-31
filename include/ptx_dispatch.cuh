#pragma once

#include <cuda.h>

#include <string>
#include <unordered_map>

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
        CU_CHECK(cuModuleLoad(&module, full_path.c_str()));
        modules_[ptx_file] = module;
        return module;
    }

    std::string ptx_root_;
    std::unordered_map<std::string, CUmodule> modules_;
    bool initialized_;
};

}  // namespace bare_nn
