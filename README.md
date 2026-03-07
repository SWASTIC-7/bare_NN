# bare_NN

A minimal neural network framework written in CUDA PTX for educational purposes and low-level GPU programming exploration.

## Overview

**bare_NN** is a bare-metal approach to understanding neural network primitives at the GPU assembly level. This project provides:

- PTX code written for basic operations of neural network
- Corresponding cuda c written to call and operate and synchronize

## Features

### Core Functionality
- [x] vector operations
- [x] Matrix multiplications
- [x] Activation
- [x] Reduction operation
- [ ] Matrix multiplication advanced
- [ ] Loss
- [x] Gradient
- [ ] Forward pass
- [ ] Backward Pass

> For PTX conscise tabular doc refer [PTX.md](./PTX.md)

### Code structure

- **ptx** this folder will contain the handwritten ptx
- **src** this folder will contain linking code to make NN
- **reference** this folder will contain the vibe coded cuda kernels for nn components for comparison
- **tests** this folder will contain test cuda codes to call ptx written kernels individually
- **include** common headers and utilities

## Building

### Prerequisites

- CUDA Toolkit 11.0+ (tested with 12.x)
- C++17 compatible compiler
- GNU Make

### Quick Build

```bash
# Build everything (release mode)
make

# Build with debug symbols
make BUILD_TYPE=debug

# Show all available targets
make help
```

### Make Targets

| Target | Description |
|--------|-------------|
| `all` | Build everything (default) |
| `main` | Build main executable |
| `cnn` | Build CNN PTX executable |
| `tests` | Build all tests |
| `run` | Run main executable |
| `run_cnn` | Run CNN executable |
| `run_tests` | Run all tests |
| `ptx` | Generate PTX from CUDA source |
| `clean` | Remove build directory |
| `distclean` | Remove all generated files |

### Build Types

| Type | Description |
|------|-------------|
| `release` | Optimized build with fast math (default) |
| `debug` | Debug build with symbols (-G -g) |

```bash
# Release build (default)
make

# Debug build
make BUILD_TYPE=debug

# Build only tests
make tests
```

### Running Tests

```bash
# Build and run all tests
make run_tests

# Run specific test
./build/tests/test_vector_op
./build/tests/test_matmul
./build/tests/test_reduction_op
```

### Project Structure After Build

```
bare_NN/
├── build/
│   ├── main              # Main executable
│   ├── cnn               # CNN executable
│   ├── ptx/              # Copied PTX files
│   └── tests/
│       ├── test_vector_op
│       ├── test_matmul
│       ├── test_reduction_op
│       └── ptx/          # PTX files for tests
├── src/
├── tests/
└── ptx/
```


## Resources

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [NVIDIA Nsight Tools](https://developer.nvidia.com/nsight-systems)
- [Attention is all you need](https://arxiv.org/abs/1706.03762)

---

**Happy GPU programming!**
