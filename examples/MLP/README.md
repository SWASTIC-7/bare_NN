# MLP + Training Example

This folder contains one executable with two implementations on the same synthetic data:

1. Wrapper API path (PTX-backed wrappers).
2. Native CUDA kernels path (`examples/MLP/native_kernels.cu`).

## What It Does

### 1) Linear Regression Training

The example first creates synthetic data on host:

`y = 2.5 * x + 1.2 + noise`

Then it trains scalar parameters `(w, b)` on GPU using MSE with gradient descent.
You should see the MSE decrease over epochs and `(w, b)` converge near `(2.5, 1.2)`.

### 2) MLP Forward Pass

After training, the same program runs a 2-layer MLP forward pass for both paths:

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

The wrapper path uses PTX-backed wrapper functions. The native path uses direct CUDA kernels for the same workload.

## Build

Run from repository root:

```bash
nvcc -std=c++17 -Iinclude -o build/mlp_example examples/MLP/main.cu examples/MLP/native_kernels.cu src/activation_fn.cu src/forward_pass.cu src/losses.cu src/backward_pass.cu src/charts/charts_api.cpp -lcuda -lcudart
```

Or use Makefile targets:

```bash
make mlp
make mlp_no_charts
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

Makefile run targets:

```bash
make run_mlp
make run_mlp_no_charts
```

Feature flag used by the MLP example:

- `-DBARE_NN_ENABLE_MLP_CHARTS=1` enables chart export
- `-DBARE_NN_ENABLE_MLP_CHARTS=0` disables chart export

## Expected Output

The program prints training progress and then MLP vectors. Shape example:

```text
=== Linear Regression Training (GPU) ===
[linreg] epoch=1 mse=...
[linreg] epoch=100 mse=...
...
[linreg] learned: w=... b=...
[linreg] target : w=2.500000 b=1.200000

=== Runtime Summary ===
wrapper training   : ... ms
native training    : ... ms
wrapper inference  : ... ms
native inference   : ... ms

=== MLP Forward Demo (PTX wrappers) ===
input: [ ... ]
hidden_pre_relu: [ ... ]
hidden_post_relu: [ ... ]
logits: [ ... ]
softmax: [ ... ]

=== Chart Export ===
[charts] wrote SVGs to examples/MLP/charts/
```

The MLP section includes:

- input
- hidden_pre_relu
- hidden_post_relu
- logits
- softmax

The softmax values should be non-negative and sum close to 1.

## Generated MLP Charts

After run (charts enabled), the program writes comparison SVG charts:

- `examples/MLP/charts/training_loss_comparison.svg` (loss on each epoch: wrapper vs native)
- `examples/MLP/charts/runtime_comparison.svg` (training and inference time comparison)
- `examples/MLP/charts/final_param_error.svg` (final learned parameter error)
- `examples/MLP/charts/softmax_comparison.svg` (wrapper vs native output distribution)

## Notes

- Weights are randomly initialized with fixed seeds for deterministic runs.
- The wrapper softmax kernel currently supports vectors up to 1024 elements in one launch.
