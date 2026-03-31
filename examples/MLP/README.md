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
nvcc -std=c++17 -Iinclude -o build/mlp_example examples/MLP/main.cu src/activation_fn.cu src/forward_pass.cu src/losses.cu src/backward_pass.cu src/charts/charts_api.cpp -lcuda -lcudart
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

After run, the program writes these SVG charts:

- `examples/MLP/charts/linreg_mse_line.svg` (line chart for training loss)
- `examples/MLP/charts/linreg_grad_bar.svg` (bar chart for `|dw|` checkpoints)
- `examples/MLP/charts/mlp_softmax_pie.svg` (pie chart of class probabilities)
- `examples/MLP/charts/mlp_hidden_stacked.svg` (stacked bar chart for hidden pre-ReLU positive/negative magnitudes)
- `examples/MLP/charts/theme_showcase.svg` (reference dashboard panel in same theme)

## Notes

- Weights are randomly initialized with fixed seeds for deterministic runs.
- The wrapper softmax kernel currently supports vectors up to 1024 elements in one launch.
