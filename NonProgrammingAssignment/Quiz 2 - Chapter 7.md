# Quiz 2 – Chapter 7 
## Non-programming Assignment:
## Describe the following activation functions and their usage: linear, ReLU, sigmoid, tanh, and softmax.
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
