# HW to Chapters 6 “Deep Neural Networks” and 7 “Activation Functions”
## Non-programming Assignment
### 1. Why are multilayer (deep) neural networks needed?
Multilayer neural networks, or deep neural networks (DNNs), are needed because they allow for the learning of complex hierarchical representations of data. Shallow networks can only capture limited, simple patterns, while deep networks, through multiple hidden layers, can model more abstract features and relationships. For example, in image processing, lower layers may detect edges, while higher layers identify more sophisticated patterns like shapes or objects.
### 2. What is the structure of weight matrix (how many rows and columns)?
The structure of a weight matrix in a neural network depends on the number of neurons in the current layer and the previous layer. If there are n neurons in the current layer and m neurons in the previous layer, the weight matrix will have n rows and m columns. Each entry in the matrix corresponds to the weight associated with the connection between a neuron in the previous layer and a neuron in the current layer.
### 3. Describe the gradient descent method.
Gradient descent is an optimization algorithm used to minimize the loss function of a neural network. It works by iteratively adjusting the weights in the direction opposite to the gradient of the loss function with respect to the weights. The basic steps of gradient descent include:

Compute the gradient (partial derivatives) of the loss function concerning each weight.
Update the weights by subtracting the gradient multiplied by a learning rate.
Repeat until the loss function converges or reaches an acceptable level of accuracy.
### 4. Describe in detail forward propagation and backpropagation for deep neural networks.
- **Forward Propagation:** In forward propagation, the input data is passed through the network layer by layer. Each neuron computes a weighted sum of its inputs, applies an activation function, and sends the output to the next layer. This process continues until the output layer produces the final prediction. The weights are not adjusted during forward propagation; they are only used to calculate the outputs.

- **Backpropagation:** Backpropagation is the process of updating the weights of the network after forward propagation to minimize the loss. It involves computing the gradient of the loss function with respect to each weight by applying the chain rule of calculus. The error is propagated backward from the output layer to the input layer, and the weights are adjusted based on the computed gradients. The goal is to minimize the loss by reducing the error at each layer.
### 5. Describe linear, ReLU, sigmoid,  tanh, and softmax activation functions and explain for what purposes and where they are typically used.
- **Linear Activation Function:**
  The output is a linear function of the input, meaning the output is directly proportional to the input. Mathematically, it can be represented as:
  
  f(x) = x
  
  This function is typically used in the output layer of regression tasks where continuous values are predicted. It is rarely used in hidden layers since it lacks the non-linearity needed for learning complex patterns.

- **ReLU (Rectified Linear Unit):**
  The ReLU function outputs zero if the input is negative, and it outputs the input itself if positive. In mathematical terms:
  
  f(x) = max(0, x)
  
  ReLU is widely used in hidden layers of deep neural networks because it introduces non-linearity and helps mitigate the vanishing gradient problem. It is effective in speeding up the learning process in neural networks.

- **Sigmoid Function:**
  The sigmoid function outputs values between 0 and 1, making it useful for binary classification tasks. The function is defined as:
  
  f(x) = 1 / (1 + e^(-x))
  
  It is often used in the output layer when the result needs to represent a probability. However, in deeper networks, the sigmoid can suffer from the vanishing gradient problem, slowing down the learning process.

- **Tanh (Hyperbolic Tangent):**
  The tanh function is similar to the sigmoid but outputs values between -1 and 1. It can be written as:
  
  f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
  
  This function is commonly used in hidden layers, particularly in classification tasks, as it can center the data around zero, which often improves convergence. However, like the sigmoid, it can also lead to vanishing gradients in very deep networks.

- **Softmax Function:**
  The softmax function converts a vector of real numbers into a probability distribution. It ensures that the output probabilities sum to 1. The formula is:
  
  f(x_i) = e^(x_i) / sum(e^(x_j)) for all j
  
  It is typically used in the output layer for multi-class classification tasks, where each class is assigned a probability. The softmax function allows for mutually exclusive classification, such as in image recognition tasks with multiple categories.