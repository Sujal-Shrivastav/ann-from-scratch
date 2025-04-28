import numpy as np
from utils.activations import sigmoid, relu, softmax, sigmoid_derivative, relu_derivative

class ANN:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        # Initialize sizes of each layer
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weights1 = np.random.randn(input_size, hidden_size1) * 0.1  # Input to hidden layer 1
        self.bias1 = np.zeros((1, hidden_size1))  # Bias for hidden layer 1
        
        self.weights2 = np.random.randn(hidden_size1, hidden_size2) * 0.1  # Hidden layer 1 to hidden layer 2
        self.bias2 = np.zeros((1, hidden_size2))  # Bias for hidden layer 2
        
        self.weights3 = np.random.randn(hidden_size2, output_size) * 0.1  # Hidden layer 2 to output layer
        self.bias3 = np.zeros((1, output_size))  # Bias for output layer

    def forward(self, X):
        # Forward pass
        self.Z1 = np.dot(X, self.weights1) + self.bias1
        self.A1 = relu(self.Z1)  # Activation of hidden layer 1
        
        self.Z2 = np.dot(self.A1, self.weights2) + self.bias2
        self.A2 = relu(self.Z2)  # Activation of hidden layer 2
        
        self.Z3 = np.dot(self.A2, self.weights3) + self.bias3
        output = softmax(self.Z3)  # Output layer activation (softmax for classification)
        
        return output

    def backward(self, X, y, output, learning_rate):
        # Backward pass (backpropagation)
        
        # Output layer error
        output_error = output - y
        dZ3 = output_error  # No activation derivative needed for softmax in this case
        
        # Gradient for weights and biases of output layer
        dW3 = np.dot(self.A2.T, dZ3)
        dB3 = np.sum(dZ3, axis=0, keepdims=True)
        
        # Hidden layer 2 error
        dA2 = np.dot(dZ3, self.weights3.T)
        dZ2 = dA2 * relu_derivative(self.Z2)  # ReLU derivative for hidden layer 2
        
        # Gradient for weights and biases of hidden layer 2
        dW2 = np.dot(self.A1.T, dZ2)
        dB2 = np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden layer 1 error
        dA1 = np.dot(dZ2, self.weights2.T)
        dZ1 = dA1 * relu_derivative(self.Z1)  # ReLU derivative for hidden layer 1
        
        # Gradient for weights and biases of hidden layer 1
        dW1 = np.dot(X.T, dZ1)
        dB1 = np.sum(dZ1, axis=0, keepdims=True)
        
        # Update weights and biases using gradient descent
        self.weights3 -= learning_rate * dW3
        self.bias3 -= learning_rate * dB3
        self.weights2 -= learning_rate * dW2
        self.bias2 -= learning_rate * dB2
        self.weights1 -= learning_rate * dW1
        self.bias1 -= learning_rate * dB1
    
    def save_weights(self, filename):
        # Save weights and biases
        np.savez(filename, weights1=self.weights1, bias1=self.bias1, weights2=self.weights2, bias2=self.bias2, weights3=self.weights3, bias3=self.bias3)
