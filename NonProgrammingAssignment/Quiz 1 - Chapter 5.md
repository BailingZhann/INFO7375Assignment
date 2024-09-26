# Quiz 1 – Chapter 5 “Forward and Backpropagation“
## Non-programming Assignment:
### Describe in detail forward- and backpropagation method for a neural network with one hidden layer including the expressions how to calculate the derivatives and update the parameters for a deep neural network.

#### Forward Propagation

In a neural network with one hidden layer, we calculate the output of the network based on the input, weights, biases, and activation functions.

Linear transformation (hidden layer):

Z1 = W1 * X + b1

Where:

- W1 is the weight matrix for the hidden layer (shape: number of neurons in hidden layer x number of input features).
- X is the input data matrix (shape: number of input features x number of examples).
- b1 is the bias vector for the hidden layer (shape: number of neurons in hidden layer x 1).

Activation (hidden layer) (assuming a sigmoid activation function):

A1 = 1 / (1 + exp(-Z1))

- A1 is the activation output of the hidden layer (shape: number of neurons in hidden layer x number of examples).

Linear transformation (output layer):

Z2 = W2 * A1 + b2

- W2 is the weight matrix for the output layer (shape: number of output neurons x number of neurons in hidden layer).
- b2 is the bias vector for the output layer (shape: number of output neurons x 1).

Activation (output layer) (using softmax for multi-class classification):

A2 = exp(Z2) / sum(exp(Z2))  (for each class)

- A2 is the final output (shape: number of output neurons x number of examples).

#### Backpropagation

**1. Calculate Error:**
Compute the loss function (e.g., mean squared error) between the predicted output (y_hat) and the true label (y):

Loss = L(y, y_hat)

**2. Backward Pass (Output Layer):**
Calculate Output Layer Error: Compute the derivative of the loss function with respect to the output layer pre-activation:

δ_o = (y_hat - y) * f'(z_o)

Calculate Gradients for Output Layer Weights and Biases:

∇W_o = a_h^T * δ_o

∇b_o = δ_o

**3. Backward Pass (Hidden Layer):**
Calculate Hidden Layer Error: Compute the error signal for the hidden layer by backpropagating the error from the output layer, weighted by the output layer weights:

δ_h = W_o^T * δ_o * f'(z_h)

Calculate Gradients for Hidden Layer Weights and Biases:

∇W_h = x^T * δ_h

∇b_h = δ_h

#### Parameter Update:

Use a gradient descent optimizer (like SGD, Adam) to update the weights and biases based on the calculated gradients:
- W_h = W_h - learning_rate * ∇W_h
- b_h = b_h - learning_rate * ∇b_h
- W_o = W_o - learning_rate * ∇W_o
- b_o = b_o - learning_rate * ∇b_o




