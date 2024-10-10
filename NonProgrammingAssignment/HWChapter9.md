# HW to Chapter 9 “Fitting, Bias, Regularization, and Dropout”
## Non-programming Assignment:
**1. What are underfitting and overfitting?**
- **Underfitting:** Occurs when the model is too simple and fails to capture the underlying patterns in the data. It results in high bias and poor performance on both training and test sets.
- **Overfitting:** Occurs when the model is too complex and learns not only the patterns but also the noise in the training data. It performs well on the training set but poorly on the test set, leading to high variance.

**2. What nay cause an dearly stopping of the gradient descent optimization process?**
  
  Early stopping is a technique used to prevent overfitting by halting training when the validation loss stops improving. Causes include:
  - Validation loss plateauing or starting to increase.
  - Insufficient learning rate adjustments.
  - Hitting a local minimum in the cost function.

**3. Describe the recognition bias vs variance and their relationship.**
- **Bias:** Error introduced by simplifying assumptions in the model. High bias leads to underfitting.
- **Variance:** Error from sensitivity to small fluctuations in the training set. High variance leads to overfitting.
- **Bias-variance tradeoff:** Reducing bias typically increases variance and vice versa. The goal is to balance bias and variance to minimize total error.

**4. Describe regularization as a method and the reasons for it.**
- **Regularization:** A technique to prevent overfitting by adding a penalty term to the loss function. The most common methods are:
  - **L1 Regularization (Lasso):** Adds a penalty proportional to the sum of the absolute values of the weights.
  - **L2 Regularization (Ridge):** Adds a penalty proportional to the sum of the squared weights.
- **Reasons:** Regularization discourages overly complex models by shrinking the weights, improving generalization to unseen data.

**5. Describe dropout as a method and the reasons for it.**
- **Dropout:** A regularization technique where, during each training iteration, a random subset of neurons is "dropped" (set to zero) in the network. This forces the network to learn redundant representations and prevents over-reliance on specific neurons.
- **Reasons:** Dropout helps prevent overfitting and improves the model's ability to generalize by reducing co-adaptation of neurons.
  