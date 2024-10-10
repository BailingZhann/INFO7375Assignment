# HW to Chapter 8 “Initialization and Training Sets”
## Non-programming Assignment
**1. To which values initialize parameters (W, b) in a neural networks and why?**
- **Weights (W):** Typically initialized randomly from a small distribution. If initialized to zeros, all neurons might behave identically, leading to poor learning. Random initialization helps break symmetry and allows different neurons to learn different features. However, the scale of the initialization matters to prevent exploding or vanishing gradients.
- **Biases (b):** Often initialized to zeros. Since biases shift the activation function output, initializing them to zero ensures the starting point for learning is balanced.

**2. Describe the problem of exploding and vanishing gradients?**
- **Exploding gradients:** In deep networks, gradients during backpropagation can grow exponentially as they propagate backward through layers. This can cause very large weight updates, making the model unstable.

- **Vanishing gradients:** Conversely, gradients can shrink to near zero, especially with sigmoid or tanh activation functions. This can prevent the weights from updating, stalling the learning process.

**3. What is Xavier initialization?**

- **Xavier (or Glorot) initialization:** A method for initializing weights in neural networks, particularly effective with sigmoid and tanh activation functions. It aims to keep the variance of the gradients across layers roughly the same, avoiding both exploding and vanishing gradients. Weights are drawn from a distribution with a variance of 1/n, where n is the number of input neurons in a layer.

**4. Describe training, validation, and testing data sets and explain their role and why all they are needed.**
- **Training set:** Used to train the model, adjusting weights based on error minimization.
- **Validation set:** Used to tune hyperparameters (like learning rate, batch size) and prevent overfitting. It acts as an independent check during training.
- **Testing set:** Used after the model is trained to evaluate its performance on unseen data. It provides an estimate of the model’s real-world performance. All these sets are needed to ensure the model generalizes well beyond the training data.

**5. What is training epoch?**

An epoch refers to one complete pass through the entire training dataset. Multiple epochs are often required for the model to sufficiently learn from the data.

**6. How to distribute training, validation, and testing sets?**

Typically, the data is split as follows:
- **Training:** 60–80%
- **Validation:** 10–20%
- **Testing:** 10–20% 

The exact split depends on the dataset size, with more data allowing for larger validation/testing sets.

**7. What is data augmentation and why may it needed?**

Data augmentation involves creating variations of existing data (e.g., rotating, flipping, or cropping images) to artificially expand the dataset. It helps prevent overfitting, improves generalization, and compensates for limited data.
