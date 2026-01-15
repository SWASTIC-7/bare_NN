# bare_NN

A minimal neural network framework written in CUDA PTX for educational purposes and low-level GPU programming exploration.

## Overview

**bare_NN** is a bare-metal approach to understanding neural network primitives at the GPU assembly level. This project provides:

- PTX code written for basic operations of neural network
- Corresponding cuda c written to call and operate and synchronize

## Features

### Core Functionality
- [x] vector operations
- [ ] Matrix multiplications
- [x] Activation
- [x] Reduction operation
- [ ] Loss
- [ ] Gradient
- [ ] Forward pass
- [ ] Backward Pass

> For PTX conscise tabular doc refer [PTX.md](./PTX.md)

### Code structure

- **ptx** this folder will contain the handwritten ptx
- **src** this folder will contain linking code to make NN
- **reference** this folder will contain the cuda kernels for nn components for comparison
- **tests** this folder will contain test cuda codes to call ptx written kernels individually
- **cmake** CMake helper modules for building
- **include** common headers and utilities

## Building

### Prerequisites

- CUDA Toolkit 11.0+ (tested with 12.x)
- CMake 3.18+
- C++17 compatible compiler
- Ninja (recommended) or Make

### Quick Build

```bash
# Configure and build (using presets)
cmake --preset default
cmake --build --preset default

# Or manually
mkdir build && cd build
cmake ..
cmake --build .
```

### Build Presets

| Preset | Description |
|--------|-------------|
| `default` | Release build with Ninja |
| `debug` | Debug build with symbols |
| `release` | Optimized release build |
| `msvc` | Visual Studio 2022 |
| `ptx-gen` | Build + generate PTX files |

```bash
# Debug build
cmake --preset debug
cmake --build --preset debug

# Visual Studio (Windows)
cmake --preset msvc
cmake --build --preset msvc-release
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_TESTS` | ON | Build test executables |
| `BUILD_EXAMPLES` | ON | Build example programs |
| `GENERATE_PTX` | OFF | Generate PTX from .cu files |
| `ENABLE_FAST_MATH` | ON | Enable fast math optimizations |
| `CMAKE_CUDA_ARCHITECTURES` | 75;86;89 | Target GPU architectures |

```bash
# Custom build example
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86 -DBUILD_TESTS=ON
cmake --build build --config Release
```

### Running Tests

```bash
# Run all tests
cd build
ctest --output-on-failure

# Or use the custom target
cmake --build build --target run_tests

# Run specific test
./build/tests/test_vector_op
```

### Project Structure After Build

```
bare_NN/
├── build/
│   ├── main.exe              # Main executable
│   ├── ptx/                  # Copied PTX files
│   └── tests/
│       ├── test_vector_op.exe
│       ├── test_matmul.exe
│       └── ptx/              # PTX files for tests
├── src/
├── tests/
└── ptx/
```


## Resources

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [NVIDIA Nsight Tools](https://developer.nvidia.com/nsight-systems)

---

**Happy GPU programming!**
