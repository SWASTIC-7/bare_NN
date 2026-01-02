# bare_NN

A minimal neural network framework written in CUDA PTX for educational purposes and low-level GPU programming exploration.

## Overview

**bare_NN** is a bare-metal approach to understanding neural network primitives at the GPU assembly level. This project provides:

- PTX code written for basic operations of neural network
- Corresponding cuda c written to call and operate and synchronize

## Features

### Core Functionality
- [] vector operations
- [] Matrix multiplications
- [] Activation
- [] Loss
- [] Gradient
- [] Forward pass
- [] Backward Pass

### Code structure

- **ptx** this folder will contain the handwritten ptx
- **src** this folder will contain linking code for calling ptx modules
- **tests** this folder will contain the cuda kernels for nn components for comparison


## Resources

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [NVIDIA Nsight Tools](https://developer.nvidia.com/nsight-systems)

---

**Happy GPU programming!** 🚀
