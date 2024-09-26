# HW to Chapter 4 & 5 “Neural Network with one hidden layer”
## Non-programming Assignment:

### 1. What is Hadamard matrix product?
   
The Hadamard matrix product, also known as the element-wise product or Schur product, is an operation where corresponding elements of two matrices of the same dimensions are multiplied together. Unlike matrix multiplication, it does not involve dot products of rows and columns.

For two matrices A and B of the same dimensions:

A ∘ B = 
| a₁₁  a₁₂ |   ∘   | b₁₁  b₁₂ | 
| a₂₁  a₂₂ |       | b₂₁  b₂₂ |

= 
| a₁₁ ⋅ b₁₁  a₁₂ ⋅ b₁₂ |
| a₂₁ ⋅ b₂₁  a₂₂ ⋅ b₂₂ |


### 2. Describe matrix multiplication?

Matrix multiplication is a binary operation where two matrices, A and B, are multiplied, producing a third matrix. The number of columns in matrix A must be equal to the number of rows in matrix B. The element in row i, column j of the resulting matrix C is the dot product of the i-th row of A and the j-th column of B.

For matrices A (of size m × n) and B (of size n × p):

C_ij = Σ (k=1 to n) [ A_ik * B_kj ]

Matrix multiplication is not commutative, meaning:

A * B ≠ B * A

### 3. What is transpose matrix and vector?
The transpose of a matrix is an operation where the rows of a matrix are turned into columns and vice versa. For a matrix A, the transpose is denoted as A^T.

For matrix A:

A =
| a  b |
| c  d |

A^T =
| a  c |
| b  d |

For a vector, which is a special case of a matrix, transposition converts a column vector into a row vector and vice versa.

For example:

v =
| v1 |
| v2 |
| v3 |

v^T = [ v1  v2  v3 ]


### 4. Describe the training set batch

In machine learning, a batch is a subset of the training dataset used during each iteration of the training process. Instead of updating model weights after every single data point (stochastic gradient descent) or after the entire dataset (batch gradient descent), the weights are updated after processing a batch of samples. This approach is called mini-batch gradient descent.

The advantages of using batches include:
- More efficient use of computational resources.
- Smoother convergence compared to stochastic gradient descent.
- Ability to parallelize over multiple processing units (like GPUs).


### 5. Describe the entropy-based loss (cost or error)function and explain why it is used for training neural networks.
Entropy-based loss functions, specifically cross-entropy loss, are commonly used in classification problems. Cross-entropy measures the difference between two probability distributions, the true label distribution y and the predicted probability distribution ŷ.

For binary classification, the cross-entropy loss is:

L = - [ y * log(ŷ) + (1 - y) * log(1 - ŷ) ]

For multi-class classification, the loss function generalizes to:

L = - ∑(i=1 to C) [ y_i * log(ŷ_i) ]

where C is the number of classes.

Cross-entropy is preferred because:
- It penalizes confident but wrong predictions more than squared error, making it a good fit for classification tasks.
- It provides better convergence in neural network training compared to other loss functions, such as Mean Squared Error (MSE).

### 6. Describe neural network supervised training process.
In supervised learning, the neural network is trained using a labeled dataset. The process involves:
- **Initialization:** Weights of the network are initialized randomly or using specific techniques like Xavier or He initialization.
- **Forward Pass:** The input data is passed through the network layer by layer, and at each layer, computations are performed (like matrix multiplications and activation functions). The final output is generated at the output layer.
- **Loss Calculation:** The predicted output is compared with the actual target using a loss function (e.g., cross-entropy for classification). The loss quantifies how far off the predictions are from the true labels.
- **Backward Pass (Backpropagation):** Gradients of the loss with respect to each weight are computed using the chain rule. These gradients are then used to update the weights of the network.
- **Weight Update:** Weights are adjusted using an optimization algorithm like gradient descent or Adam to minimize the loss.
- **Repeat:** The process is repeated over several epochs (complete passes through the dataset), and over time, the network learns to minimize the loss and improve its predictions.

### 7. Describe in detail forward propagation and backpropagation.
#### Forward Propagation:
**1. Input Layer:** The input data is fed into the network.

**2. Hidden Layers:** The data passes through multiple hidden layers. At each hidden layer, the input is transformed by the neurons. For each neuron, the input is multiplied by weights, summed, and passed through an activation function (e.g., ReLU, sigmoid, tanh).

The operation at each neuron is:
z = W ⋅ x + b
where:
- W are the weights,
- x is the input,
- b is the bias.

The result is passed through an activation function to introduce non-linearity: a = σ(z)

**3. Output Layer:** The result from the last hidden layer is passed to the output layer, producing the final prediction.

#### Backpropagation:
**1. Loss Calculation:** First, the loss (error) between the predicted output and the actual output is computed using a loss function.

**2. Gradient Calculation:** The key idea is to compute the gradient of the loss with respect to each weight in the network. This is done using the chain rule of calculus, which propagates the error backward through the network.

The weight updates follow the gradient descent rule:
w_new = w_old - η ⋅ (∂L / ∂w)
where:
- η is the learning rate,
- (∂L / ∂w) is the gradient of the loss with respect to the weights.

**3. Update Weights:** Each weight is adjusted according to the computed gradients. The goal is to reduce the error by moving the weights in the direction of the negative gradient.

**4. Repeat:** The process is repeated for all layers in the network, adjusting the weights iteratively until the loss is minimized.

#### Summary:
- **Forward propagation** generates the prediction, passing the data from the input layer through hidden layers to the output layer.
- **Backpropagation** calculates how much each weight in the network contributed to the error and updates the weights to reduce this error. This iterative process is repeated until the model improves its predictions.
