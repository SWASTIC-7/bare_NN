# Bare Neural Network - Implementation Blueprint

A minimal neural network framework from scratch using CUDA/PTX, inspired by micrograd and nanoGPT.

## Phase 1: Foundation - Tensor Operations (CUDA Kernels)

### 1.1 Basic Element-wise Operations 
- [x] Vector addition (a + b)
- [x] Vector subtraction (a - b)
- [x] Vector multiplication (a * b, Hadamard product)
- [x] Vector division (a / b)
- [x] Scalar operations (a + scalar, a * scalar)

### 1.2 Activation Functions
- [ ] ReLU: `f(x) = max(0, x)`
- [ ] Sigmoid: `f(x) = 1 / (1 + exp(-x))`
- [ ] Tanh: `f(x) = tanh(x)`
- [ ] Leaky ReLU: `f(x) = x if x > 0 else alpha * x`
- [ ] GELU (for transformers)

### 1.3 Activation Gradients (Backward Pass)
- [ ] ReLU gradient: `f'(x) = 1 if x > 0 else 0`
- [ ] Sigmoid gradient: `f'(x) = sigmoid(x) * (1 - sigmoid(x))`
- [ ] Tanh gradient: `f'(x) = 1 - tanh²(x)`
- [ ] Leaky ReLU gradient
- [ ] GELU gradient

### 1.4 Reduction Operations
- [ ] Sum: `sum(x)`
- [ ] Mean: `mean(x)`
- [ ] Max: `max(x)`
- [ ] Min: `min(x)`
- [ ] Variance: `var(x)`
- [ ] Standard deviation: `std(x)`

### 1.5 Matrix Operations
- [ ] Matrix-Vector multiplication: `y = A * x`
- [ ] Matrix-Matrix multiplication: `C = A * B`
- [ ] Batched matrix multiplication
- [ ] Transpose
- [ ] Matrix-Vector with bias: `y = A * x + b`

### 1.6 Advanced Operations
- [ ] Broadcasting support
- [ ] Concatenation
- [ ] Slicing/Indexing
- [ ] Reshape operations
- [ ] Softmax: `softmax(x)_i = exp(x_i) / sum(exp(x))`
- [ ] Log-softmax (numerically stable)

---

## Phase 2: Autograd Engine (Like micrograd)

### 2.1 Tensor Class with Gradient Tracking
```cpp
class Tensor {
    float* data;           // Device pointer
    float* grad;           // Gradient (same shape as data)
    int* shape;            // Dimensions
    int ndim;              // Number of dimensions
    bool requires_grad;    // Track gradients?
    
    // Computation graph
    std::vector<Tensor*> children;
    std::function<void()> backward_fn;
};
```

### 2.2 Forward Operations (Building Compute Graph)
- [ ] Add operation with gradient tracking
- [ ] Multiply operation with gradient tracking
- [ ] MatMul operation with gradient tracking
- [ ] Activation operations with gradient tracking
- [ ] Loss operations with gradient tracking

### 2.3 Backward Pass (Automatic Differentiation)
- [ ] Topological sort for computation graph
- [ ] Backward pass implementation
- [ ] Chain rule application
- [ ] Gradient accumulation

### 2.4 Gradient Operations
- [ ] Zero gradients
- [ ] Gradient clipping
- [ ] Gradient checking (numerical vs analytical)

---

## Phase 3: Neural Network Layers

### 3.1 Core Layer Abstraction
```cpp
class Layer {
    virtual Tensor* forward(Tensor* input) = 0;
    virtual void backward(Tensor* grad_output) = 0;
    virtual std::vector<Tensor*> parameters() = 0;
};
```

### 3.2 Linear (Dense) Layer
- [ ] Forward: `y = Wx + b`
- [ ] Backward: Compute gradients for W, b, and input
- [ ] Xavier/He initialization
- [ ] Parameter storage and retrieval

### 3.3 Activation Layers
- [ ] ReLU layer
- [ ] Sigmoid layer
- [ ] Tanh layer
- [ ] Softmax layer

### 3.4 Normalization Layers
- [ ] Batch Normalization
  - Forward: normalize, scale, shift
  - Backward: gradient computation
  - Running statistics (mean, variance)
- [ ] Layer Normalization (for transformers)
- [ ] RMSNorm (modern alternative)

### 3.5 Regularization Layers
- [ ] Dropout
  - Training mode: random zeroing
  - Inference mode: scale adjustment
  - Gradient computation

### 3.6 Embedding Layer (for NLP)
- [ ] Lookup table implementation
- [ ] Gradient w.r.t. embeddings
- [ ] Vocabulary size and embedding dimension

---

## Phase 4: Loss Functions

### 4.1 Regression Losses
- [ ] Mean Squared Error (MSE): `L = mean((y_pred - y_true)²)`
- [ ] Mean Absolute Error (MAE): `L = mean(|y_pred - y_true|)`

### 4.2 Classification Losses
- [ ] Cross-Entropy Loss: `L = -sum(y_true * log(y_pred))`
- [ ] Binary Cross-Entropy
- [ ] Sparse Cross-Entropy (for integer labels)

### 4.3 Advanced Losses
- [ ] Hinge loss (SVM)
- [ ] Focal loss
- [ ] Contrastive loss

---

## Phase 5: Optimizers

### 5.1 Basic Optimizers
- [ ] **SGD (Stochastic Gradient Descent)**
  ```
  w = w - learning_rate * grad_w
  ```

- [ ] **SGD with Momentum**
  ```
  v = momentum * v - learning_rate * grad_w
  w = w + v
  ```

### 5.2 Adaptive Learning Rate Optimizers
- [ ] **AdaGrad**
  ```
  cache = cache + grad_w²
  w = w - learning_rate * grad_w / (sqrt(cache) + eps)
  ```

- [ ] **RMSprop**
  ```
  cache = decay * cache + (1 - decay) * grad_w²
  w = w - learning_rate * grad_w / (sqrt(cache) + eps)
  ```

- [ ] **Adam (Adaptive Moment Estimation)**
  ```
  m = beta1 * m + (1 - beta1) * grad_w          // First moment
  v = beta2 * v + (1 - beta2) * grad_w²         // Second moment
  m_hat = m / (1 - beta1^t)                     // Bias correction
  v_hat = v / (1 - beta2^t)
  w = w - learning_rate * m_hat / (sqrt(v_hat) + eps)
  ```

- [ ] **AdamW (Adam with weight decay)**

### 5.3 Learning Rate Scheduling
- [ ] Constant learning rate
- [ ] Step decay
- [ ] Exponential decay
- [ ] Cosine annealing
- [ ] Warmup + decay schedule

---

## Phase 6: Network Architecture

### 6.1 Sequential Model Container
```cpp
class Sequential {
    std::vector<Layer*> layers;
    
    Tensor* forward(Tensor* input);
    void backward(Tensor* loss);
    std::vector<Tensor*> parameters();
};
```

### 6.2 Multi-Layer Perceptron (MLP)
- [ ] Input layer
- [ ] Hidden layers with activations
- [ ] Output layer
- [ ] Forward pass through all layers
- [ ] Backward pass through all layers

### 6.3 Example: Simple MLP for MNIST
```
Input(784) -> Linear(128) -> ReLU -> 
Linear(64) -> ReLU -> 
Linear(10) -> Softmax
```

---

## Phase 7: Training Infrastructure

### 7.1 Data Handling
- [ ] Dataset abstraction
- [ ] Batch creation
- [ ] Data shuffling
- [ ] Train/validation split
- [ ] Data augmentation (optional)

### 7.2 Training Loop
```cpp
for epoch in epochs:
    for batch in dataloader:
        // Forward pass
        predictions = model.forward(batch.x)
        
        // Compute loss
        loss = loss_function(predictions, batch.y)
        
        // Backward pass
        model.zero_grad()
        loss.backward()
        
        // Update weights
        optimizer.step()
        
    // Validation
    val_loss = validate(model, val_data)
    print(f"Epoch {epoch}: train_loss={loss}, val_loss={val_loss}")
```

### 7.3 Evaluation Metrics
- [ ] Accuracy (classification)
- [ ] Precision, Recall, F1-score
- [ ] Confusion matrix
- [ ] MSE, RMSE (regression)

### 7.4 Model Persistence
- [ ] Save model weights
- [ ] Load model weights
- [ ] Checkpoint saving during training

---

## Phase 8: Simple Projects (Like micrograd examples)

### 8.1 Binary Classification
- [ ] Dataset: Simple 2D points (moon/circles)
- [ ] Model: 2-layer MLP
- [ ] Loss: Binary cross-entropy
- [ ] Visualization of decision boundary

### 8.2 MNIST Digit Classification
- [ ] Dataset: 28x28 grayscale images
- [ ] Model: 3-layer MLP
- [ ] Loss: Cross-entropy
- [ ] Metrics: Accuracy

### 8.3 Simple Regression
- [ ] Dataset: Synthetic function (sin, polynomial)
- [ ] Model: MLP
- [ ] Loss: MSE
- [ ] Visualization of fit

---

## Phase 9: Attention & Transformer (Like nanoGPT)

### 9.1 Attention Mechanism Components
- [ ] **Scaled Dot-Product Attention**
  ```
  Q, K, V = input transformations
  scores = (Q @ K^T) / sqrt(d_k)
  attention = softmax(scores)
  output = attention @ V
  ```

### 9.2 Multi-Head Attention
- [ ] Multiple parallel attention heads
- [ ] Concatenation and linear projection
- [ ] Forward and backward passes

### 9.3 Positional Encoding
- [ ] Sinusoidal positional encoding
- [ ] Learned positional embeddings
- [ ] Addition to input embeddings

### 9.4 Transformer Block
- [ ] Multi-head self-attention
- [ ] Layer normalization
- [ ] Feed-forward network (MLP)
- [ ] Residual connections
- [ ] Dropout

### 9.5 Full Transformer Architecture
```
Input Embeddings + Positional Encoding
  ↓
Transformer Block × N layers
  ↓
Final Layer Norm
  ↓
Output Projection
```

---

## Phase 10: Language Model (nanoGPT-style)

### 10.1 Tokenization
- [ ] Character-level tokenizer
- [ ] Byte-pair encoding (BPE) tokenizer
- [ ] Vocabulary building

### 10.2 Dataset Preparation
- [ ] Text corpus loading
- [ ] Tokenization
- [ ] Creating training sequences
- [ ] Batch generation with masking

### 10.3 GPT Model
- [ ] Token embeddings
- [ ] Positional embeddings
- [ ] Transformer decoder blocks
- [ ] Causal (autoregressive) masking
- [ ] Output projection to vocabulary

### 10.4 Training
- [ ] Cross-entropy loss on next token prediction
- [ ] Teacher forcing
- [ ] Gradient accumulation
- [ ] Learning rate warmup + cosine decay

### 10.5 Text Generation
- [ ] Greedy decoding
- [ ] Top-k sampling
- [ ] Top-p (nucleus) sampling
- [ ] Temperature scaling

---

## Phase 11: Optimization & Performance

### 11.1 Memory Optimization
- [ ] In-place operations
- [ ] Gradient checkpointing
- [ ] Mixed precision training (FP16/FP32)

### 11.2 Compute Optimization
- [ ] Kernel fusion
- [ ] Shared memory usage
- [ ] Warp-level optimizations
- [ ] Tensor cores (if available)

### 11.3 Profiling
- [ ] CUDA event timing
- [ ] Nsight profiling
- [ ] Memory usage tracking
- [ ] Bottleneck identification

---

## Phase 12: Testing & Validation

### 12.1 Unit Tests
- [ ] Kernel correctness tests
- [ ] Gradient checking
- [ ] Shape broadcasting tests
- [ ] Edge cases (zero, negative values)

### 12.2 Integration Tests
- [ ] End-to-end training
- [ ] Overfitting on small dataset (sanity check)
- [ ] Comparison with PyTorch/reference implementation

### 12.3 Numerical Stability
- [ ] Log-sum-exp trick for softmax
- [ ] Gradient clipping
- [ ] Epsilon in denominators
- [ ] Overflow/underflow handling

---

## Implementation Order (Recommended)

### Week 1-2: Foundation
1. Complete Phase 1 (Tensor operations)
2. Test all kernels thoroughly
3. Benchmark against cuBLAS

### Week 3: Autograd
4. Implement Phase 2 (Autograd engine)
5. Start with scalar autograd (like micrograd)
6. Extend to tensors
7. Test gradient correctness

### Week 4: Basic NN
8. Implement Phase 3 (Layers)
9. Implement Phase 4 (Loss functions)
10. Implement Phase 5 (SGD optimizer)

### Week 5: First Model
11. Implement Phase 6 (Sequential model)
12. Implement Phase 7 (Training loop)
13. Train simple MLP on toy dataset

### Week 6-7: MNIST
14. Complete Phase 8.2 (MNIST)
15. Achieve >95% accuracy
16. Optimize performance

### Week 8-10: Transformer
17. Implement Phase 9 (Attention)
18. Build transformer block by block
19. Test each component

### Week 11-12: Language Model
20. Implement Phase 10 (GPT)
21. Train on small text corpus
22. Generate coherent text

---

## Key Milestones

- ✓ **Milestone 1**: Vector operations working
- **Milestone 2**: Autograd passes gradient checks
- **Milestone 3**: Train XOR problem (classic NN test)
- **Milestone 4**: >95% accuracy on MNIST
- **Milestone 5**: Attention mechanism working
- **Milestone 6**: Small transformer can memorize sequences
- **Milestone 7**: GPT generates coherent text

---

## Testing Strategy

For each component:
1. **Unit test**: Test in isolation
2. **Gradient check**: Compare numerical vs analytical gradients
3. **Shape test**: Verify all tensor shapes
4. **Reference test**: Compare with PyTorch/NumPy
5. **Edge cases**: Zero, infinity, NaN handling

---

## Resources & References

### Tutorials
- Andrej Karpathy's micrograd: Autograd engine
- Andrej Karpathy's nanoGPT: Minimal GPT implementation
- "Neural Networks: Zero to Hero" YouTube series

### Papers
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)
- "Batch Normalization" (Ioffe & Szegedy, 2015)

### Books
- "Deep Learning" by Goodfellow, Bengio, Courville
- "Dive into Deep Learning" (d2l.ai)

---

## File Structure

```
bare_NN/
├── src/
│   ├── tensor.cu           # Tensor class with autograd
│   ├── ops.cu              # Basic operations
│   ├── layers.cu           # NN layers
│   ├── optimizer.cu        # Optimizers
│   ├── loss.cu             # Loss functions
│   ├── model.cu            # Model containers
│   └── main.cu             # Training examples
├── ptx/
│   ├── vector_op.ptx       # Basic kernels
│   ├── matmul.ptx          # Matrix operations
│   ├── activation.ptx      # Activation functions
│   └── attention.ptx       # Attention kernels
├── tests/
│   ├── test_ops.cu
│   ├── test_autograd.cu
│   ├── test_layers.cu
│   └── test_model.cu
└── examples/
    ├── mnist.cu
    ├── text_generation.cu
    └── toy_problems.cu
```

---

## Notes

- Start simple, test thoroughly at each step
- Gradient checking is crucial - don't skip it
- Visualize intermediate outputs when debugging
- Compare with PyTorch to verify correctness
- Profile early to avoid performance pitfalls
- Document weird CUDA behaviors you encounter
