# Non-programming Assignment
## 1. What is normalization and why is it needed?

Normalization is the process of scaling the input data to ensure that all features have the same range, typically between 0 and 1 or -1 and 1. This is crucial for machine learning models, especially those using gradient-based optimization methods like neural networks, for the following reasons:

- **Improved Convergence Speed:** Normalization helps the optimization algorithm converge faster by ensuring that the gradient updates are not disproportionately large for some features and small for others.
- **Avoiding Large Gradient Updates:** Without normalization, features with large values can dominate the learning process, leading to unstable gradients and slower learning.
- **Better Model Performance:** Models trained on normalized data tend to perform better as they are not biased toward certain features with large magnitudes, leading to better generalization.

## 2. What are vanishing and exploding gradients?

Vanishing and exploding gradients are problems that occur during the training of deep neural networks due to the nature of backpropagation.

- **Vanishing Gradients:** This happens when the gradients of the loss function with respect to earlier layers become very small. As a result, the weights of these layers receive extremely small updates, which makes the model slow to learn or even stop learning entirely. This is commonly seen in deep networks with sigmoid or tanh activation functions, where gradients tend to shrink as they propagate backward through many layers.

- **Exploding Gradients:** The opposite problem occurs when gradients become excessively large, causing huge updates to the model weights. This leads to unstable training, where the model's parameters oscillate or diverge, making it hard for the model to converge.

Both of these issues are more likely to occur in very deep networks and can be mitigated through techniques such as better weight initialization (e.g., Xavier or He initialization), using activation functions like ReLU, or applying gradient clipping.

## 3. What Adam algorithm and why is it needed?

Adam (Adaptive Moment Estimation) is an optimization algorithm that combines the advantages of two popular methods: AdaGrad (which adapts the learning rate for each parameter based on the first moment, i.e., mean of the gradients) and RMSProp (which adapts the learning rate using the second moment, i.e., variance of the gradients).

The Adam algorithm is needed because it:

- Adapts learning rates for each parameter, making it suitable for problems with sparse gradients or different learning rates for different parameters.
- Maintains momentum by using exponentially weighted averages of past gradients (first moment) and squared gradients (second moment) to smooth out updates and improve convergence.
- Avoids the vanishing learning rate problem seen in other adaptive methods like AdaGrad, where the learning rate might shrink too much over time.

Adam is highly popular because it generally requires less hyperparameter tuning, is computationally efficient, and works well in practice for many deep learning tasks.

## 4. How to choose hyperparameters?

Some general guidelines include:

- **Learning Rate:** One of the most important hyperparameters. Too high can cause divergence; too low can result in slow convergence. You can start with a relatively large value (e.g., 0.01) and decrease it using techniques like learning rate schedules or decay.

- **Batch Size:** Smaller batch sizes generally introduce more noise into the gradient estimates, which can help escape local minima but may slow down convergence. Larger batch sizes lead to smoother updates but require more memory. Common values are 32, 64, or 128.

- **Number of Epochs:** This defines how many times the model sees the entire dataset. If your model starts overfitting (performance on validation data degrades while performance on training data improves), early stopping can be applied to stop training when the validation loss increases.

- **Regularization Parameters:** Techniques like L2 regularization or dropout can help prevent overfitting. Hyperparameters for dropout probability or regularization strength (λ) need to be carefully tuned.

- **Optimization Algorithm:** Algorithms like Adam, SGD with momentum, and RMSProp all have hyperparameters like momentum or beta values that can be chosen based on the specific task. Adam generally works well for most applications without much tuning.

Typically, hyperparameters are chosen through grid search, random search, or more advanced methods like Bayesian optimization, depending on the complexity of the model and the resources available.