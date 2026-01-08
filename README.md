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


## Resources

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [NVIDIA Nsight Tools](https://developer.nvidia.com/nsight-systems)

---

**Happy GPU programming!**
