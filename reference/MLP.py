"""
Minimalistic Two Hidden Layer MLP for Handwritten Digit Recognition (MNIST)
"""

import numpy as np

# ============== Activation Functions ==============
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# ============== Loss Function ==============
def cross_entropy_loss(predictions, targets):
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -np.mean(np.sum(targets * np.log(predictions), axis=1))

# ============== MLP Class ==============
class MLP:
    def __init__(self, input_size=784, hidden1_size=128, hidden2_size=64, output_size=10):
        """
        Two hidden layer MLP for digit recognition
        Architecture: 784 -> 128 -> 64 -> 10
        """
        # Xavier initialization
        self.W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden1_size))
        
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2.0 / hidden1_size)
        self.b2 = np.zeros((1, hidden2_size))
        
        self.W3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2.0 / hidden2_size)
        self.b3 = np.zeros((1, output_size))
    
    def forward(self, X):
        """Forward pass through the network"""
        # Hidden layer 1
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        
        # Hidden layer 2
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = relu(self.z2)
        
        # Output layer
        self.z3 = self.a2 @ self.W3 + self.b3
        self.output = softmax(self.z3)
        
        return self.output
    
    def backward(self, X, y, learning_rate=0.01):
        """Backward pass - compute gradients and update weights"""
        batch_size = X.shape[0]
        
        # Output layer gradient (softmax + cross-entropy)
        dz3 = self.output - y
        dW3 = (self.a2.T @ dz3) / batch_size
        db3 = np.mean(dz3, axis=0, keepdims=True)
        
        # Hidden layer 2 gradient
        da2 = dz3 @ self.W3.T
        dz2 = da2 * relu_derivative(self.z2)
        dW2 = (self.a1.T @ dz2) / batch_size
        db2 = np.mean(dz2, axis=0, keepdims=True)
        
        # Hidden layer 1 gradient
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_derivative(self.z1)
        dW1 = (X.T @ dz1) / batch_size
        db1 = np.mean(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X_train, y_train, epochs=10, batch_size=32, learning_rate=0.1):
        """Train the network"""
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            total_loss = 0
            n_batches = n_samples // batch_size
            
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # Forward and backward pass
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)
                total_loss += cross_entropy_loss(output, y_batch)
            
            avg_loss = total_loss / n_batches
            accuracy = self.evaluate(X_train, y_train)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")
    
    def predict(self, X):
        """Predict class labels"""
        return np.argmax(self.forward(X), axis=1)
    
    def evaluate(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        labels = np.argmax(y, axis=1)
        return np.mean(predictions == labels)

# ============== Data Loading ==============
def load_mnist_from_ubyte(images_path, labels_path):
    """Load MNIST from ubyte files"""
    # Load images
    with open(images_path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        n_images = int.from_bytes(f.read(4), 'big')
        n_rows = int.from_bytes(f.read(4), 'big')
        n_cols = int.from_bytes(f.read(4), 'big')
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(n_images, n_rows * n_cols)
    
    # Load labels
    with open(labels_path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        n_labels = int.from_bytes(f.read(4), 'big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    return images, labels

def one_hot_encode(labels, num_classes=10):
    """Convert labels to one-hot encoding"""
    one_hot = np.zeros((labels.shape[0], num_classes))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot

# ============== Main ==============
if __name__ == "__main__":
    import os
    
    # Paths to MNIST data
    data_dir = os.path.join(os.path.dirname(__file__), "CUDA-CNN", "data")
    
    print("Loading MNIST dataset...")
    X_train, y_train = load_mnist_from_ubyte(
        os.path.join(data_dir, "train-images.idx3-ubyte"),
        os.path.join(data_dir, "train-labels.idx1-ubyte")
    )
    X_test, y_test = load_mnist_from_ubyte(
        os.path.join(data_dir, "t10k-images.idx3-ubyte"),
        os.path.join(data_dir, "t10k-labels.idx1-ubyte")
    )
    
    # Normalize pixel values to [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    # One-hot encode labels
    y_train_onehot = one_hot_encode(y_train)
    y_test_onehot = one_hot_encode(y_test)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Input size: {X_train.shape[1]}")
    
    # Create and train MLP
    print("\nCreating MLP: 784 -> 128 -> 64 -> 10")
    mlp = MLP(input_size=784, hidden1_size=128, hidden2_size=64, output_size=10)
    
    print("\nTraining...")
    mlp.train(X_train, y_train_onehot, epochs=20, batch_size=64, learning_rate=0.1)
    
    # Evaluate on test set
    test_accuracy = mlp.evaluate(X_test, y_test_onehot)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
