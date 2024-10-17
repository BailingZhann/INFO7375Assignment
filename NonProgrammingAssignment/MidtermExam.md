# Exam Questions:
## 1. Describe the artificial neuron model.

- **Description:**- An artificial neuron, inspired by biological neurons, is the basic unit in a neural network. It takes multiple inputs, multiplies each input by a corresponding weight, sums them up, and applies an activation function to produce an output.
- **Example:** In a single-layer perceptron, inputs x1, x2, ..., xn are weighted by w1, w2, ..., wn, and the neuron computes z = sum(wi * xi) + b, where b is a bias term, and applies an activation function like sigmoid to generate the output.
- **Usage:** Neurons form the layers of a neural network to process information.

## 2. What is the logistic regression problem?

- **Description:** Logistic regression is a binary classification algorithm that models the probability of an outcome belonging to a class (0 or 1) by using the logistic function (sigmoid) to ensure the output lies between 0 and 1.
- **Example:** Given input features x1, x2, ..., logistic regression predicts the probability that y = 1 with P(y = 1 | x) = 1 / (1 + exp(-(w * x + b))).
- **Usage:** Used for classification problems, like spam detection.

## 3. Describe multilayer (deep) neural network.

- **Description:** A multilayer neural network (or deep neural network) consists of an input layer, multiple hidden layers, and an output layer. Each layer transforms the input by applying weights, biases, and activation functions to capture complex patterns.
- **Example:** A deep network with three layers might take images as input and pass them through layers to identify objects.
- **Usage:** Used for complex tasks like image recognition, language processing.

## 4. Describe major activation functions: linear, ReLU, sigmoid, tanh, and softmax, and explain their usage.
- **Linear:** Output is a weighted sum of inputs. No non-linearity, so it's not useful for deep networks.
- **ReLU (Rectified Linear Unit):** f(x) = max(0, x), introduces non-linearity and is efficient for deep networks.
- **Sigmoid:** f(x) = 1 / (1 + exp(-x)), squashes output between 0 and 1, often used in binary classification.
- **Tanh:** f(x) = 2 / (1 + exp(-2x)) - 1, ranges from -1 to 1, useful in dealing with negative inputs.
- **Softmax:** Converts logits into a probability distribution over multiple classes. Often used in the output layer of classification networks.

## 5. What is supervised learning?

- **Description:** In supervised learning, the model is trained on labeled data, where each input has a corresponding output, and the goal is to learn a mapping from inputs to outputs.
- **Example:** In image classification, labeled images are provided, and the model learns to predict the correct class label for new images.
- **Usage:** Used in tasks like regression and classification.

## 6. Describe loss/cost function.

- **Description:** The loss function measures the difference between the predicted output and the actual output. The cost function aggregates the loss over the entire training dataset.
- **Example:** Mean Squared Error (MSE) for regression: MSE = (1 / n) * sum((yi - y_hat_i)^2), where yi is the actual value and y_hat_i is the predicted value.
- **Usage:** Minimizing the loss function helps the model improve its predictions.

## 7. Describe forward and backward propagation for a multilayer (deep) neural network.

- **Forward Propagation:** Passes input through the layers, applying weights and activation functions, to produce output predictions.
- **Backward Propagation:** Calculates gradients of the loss with respect to weights using the chain rule and updates weights to minimize the loss.
- **Usage:** Both steps are key to training a neural network via gradient descent.

## 8. What are parameters and hyperparameters in neural networks and what is the conceptual difference between them.

- **Parameters:** Model weights and biases, learned during training.
- **Hyperparameters:** Settings such as learning rate, number of layers, batch size, set before training and tuned manually.
- **Difference:** Parameters are adjusted by the training process, while hyperparameters are chosen by the developer.

## 9. How to set the initial values for the neural network training

- **Description:** Proper initialization of weights is crucial to avoid issues like vanishing or exploding gradients.
- **Example:** Xavier/Glorot initialization sets weights based on the number of input and output units to balance the variance.
- **Usage:** Ensures faster and more reliable training convergence.

## 10. Why are mini-batches used instead of complete batches in training of neural networks.
- **Description:** Mini-batches (smaller subsets of the training data) are used to update weights rather than using the full dataset at once.
- **Example:** A mini-batch size of 32 means that gradient descent updates the model after processing 32 samples.
- **Usage:** Mini-batches speed up training and reduce memory usage while also adding a regularizing effect through noise.
