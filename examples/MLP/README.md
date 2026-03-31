# MLP Example

This folder contains a simple functional MLP inference example built with the wrapper library API.

## What It Does

The example in [examples/MLP/main.cu](examples/MLP/main.cu) runs a 2-layer MLP forward pass on GPU:

- Input vector size: 4
- Hidden layer size: 8
- Output layer size: 3
- Activation: ReLU
- Output normalization: Softmax

Execution flow:

1. Hidden pre-activation = FC(input, W1)
2. Hidden activation = ReLU(hidden pre-activation)
3. Logits = FC(hidden activation, W2)
4. Probabilities = Softmax(logits)

All operations are dispatched through PTX-backed wrapper functions from the library.

## Build

Run from repository root:

```bash
nvcc -std=c++17 -Iinclude -o build/mlp_example examples/MLP/main.cu src/activation_fn.cu src/forward_pass.cu src/losses.cu src/backward_pass.cu -lcuda -lcudart
```

## Run

Run from repository root so relative PTX paths resolve correctly:

```bash
./build/mlp_example
```

On Windows PowerShell:

```powershell
.\build\mlp_example.exe
```

## Expected Output

The program prints:

- input
- hidden_pre_relu
- hidden_post_relu
- logits
- softmax

The softmax values should be non-negative and sum close to 1.

## Notes

- This example is inference-only.
- Weights are randomly initialized with fixed seeds for deterministic runs.
- The wrapper softmax kernel currently supports vectors up to 1024 elements in one launch.
