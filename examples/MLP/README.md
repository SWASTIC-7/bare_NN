# MLP + Training Example

This folder contains one executable with two demonstrations:

1. Linear regression training on synthetic data.
2. A 2-layer MLP forward pass using PTX-backed wrapper calls.

## What It Does

### 1) Linear Regression Training

The example first creates synthetic data on host:

`y = 2.5 * x + 1.2 + noise`

Then it trains scalar parameters `(w, b)` on GPU using MSE with gradient descent.
You should see the MSE decrease over epochs and `(w, b)` converge near `(2.5, 1.2)`.

### 2) MLP Forward Pass

After training, the same program runs a 2-layer MLP forward pass:

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

The MLP operations are dispatched through PTX-backed wrapper functions from the library.

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

The program prints training progress and then MLP vectors. Shape example:

```text
=== Linear Regression Training (GPU) ===
[linreg] epoch=1 mse=...
[linreg] epoch=100 mse=...
...
[linreg] learned: w=... b=...
[linreg] target : w=2.500000 b=1.200000

=== MLP Forward Demo (PTX wrappers) ===
input: [ ... ]
hidden_pre_relu: [ ... ]
hidden_post_relu: [ ... ]
logits: [ ... ]
softmax: [ ... ]
```

The MLP section includes:

- input
- hidden_pre_relu
- hidden_post_relu
- logits
- softmax

The softmax values should be non-negative and sum close to 1.

## Notes

- Weights are randomly initialized with fixed seeds for deterministic runs.
- The wrapper softmax kernel currently supports vectors up to 1024 elements in one launch.
