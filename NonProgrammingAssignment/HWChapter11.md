# HW to Chapter 11 “Learning Rates Decay and Hyperparameters”
## Non-programming Assignment
### 1. What is learning rate decay and why is it needed?

Learning rate decay refers to the technique of gradually decreasing the learning rate as training progresses. The learning rate controls how much the model's weights are updated with each iteration during training. In the early stages of training, a higher learning rate can help the model converge faster by making larger weight updates. However, as the model approaches the optimal solution, large updates can cause oscillation around the minimum, preventing the model from converging.

Why is it needed?

- **Stabilizes Convergence:** As the model approaches the optimal solution, reducing the learning rate helps to make smaller, more precise updates, allowing for better convergence to a local or global minimum.
- **Prevents Overshooting:** A high learning rate throughout training can cause the optimizer to overshoot the minimum, leading to oscillations. Decay prevents this by reducing the step size.
- **Improves Generalization:** By using a larger learning rate initially and gradually decreasing it, the model can find better generalizations, avoiding overfitting or getting stuck in suboptimal minima.

Common learning rate decay strategies include exponential decay, step decay, and 1/t decay, where the learning rate decreases according to a specific schedule or rule.

### 2. What are saddle and plateau problems?

- **Saddle Point Problem:** A saddle point is a point in the loss landscape where the gradient is zero but the point is neither a local minimum nor a local maximum. In higher-dimensional spaces, the loss surface can have flat regions that curve upward in some directions and downward in others. When the optimizer reaches a saddle point, the gradients are very small, causing slow or stagnant progress, making it difficult for gradient-based methods to escape.

- **Plateau Problem:** A plateau is a flat region in the loss surface where the gradients are very small or zero over a wide area. When the optimizer reaches a plateau, learning becomes extremely slow because gradient updates are minimal, even though the global minimum may lie beyond the plateau. Plateaus can slow down convergence significantly.

Both saddle points and plateaus result in small gradient magnitudes, causing the optimizer to make very slow updates. Techniques such as momentum, adaptive optimizers (e.g., Adam), and learning rate schedules help tackle these issues by maintaining momentum or adjusting learning rates based on the gradients' history.

### 3. Why should we avoid grid approach in hyperparameter choice?

The grid search approach involves specifying a set of values for each hyperparameter and evaluating every combination of these values in a grid format. While simple, this method has several drawbacks:

- **Computationally Expensive:** Grid search grows exponentially with the number of hyperparameters. For each new hyperparameter, the search space expands significantly, leading to an impractically large number of evaluations, especially when working with high-dimensional spaces.

- **Inefficient:** In many cases, not all combinations of hyperparameters are equally important, and some may contribute more to the model's performance than others. Grid search treats all hyperparameters equally and spends unnecessary time evaluating unimportant combinations.

- **Curse of Dimensionality:** For models with many hyperparameters, grid search becomes inefficient because it fails to explore promising regions of the search space. Instead, it evaluates values evenly distributed across the entire space, even though the optimal solution may lie in a narrow region.

- **Random Search and Bayesian Optimization:** Alternative methods like random search and Bayesian optimization tend to perform better in practice. Random search samples values from a probability distribution, allowing it to explore a wider variety of hyperparameter combinations with fewer evaluations. Bayesian optimization uses past evaluations to model the objective function and select hyperparameter combinations more intelligently.

### 4. What is mini batch and how is it used?

Mini-batch is a compromise between two extremes in training deep learning models: stochastic gradient descent (SGD) (where the gradient is calculated for each data point) and batch gradient descent (where the gradient is calculated over the entire dataset).

- **Mini-batch Gradient Descent:** In this method, the dataset is split into smaller groups called mini-batches, and the model's weights are updated based on the average gradient of each mini-batch. Each mini-batch typically contains a small subset of the training data, like 32, 64, or 128 samples.

How is it used?

- **Divide the Dataset:** The dataset is divided into multiple mini-batches.
- **Compute Gradients:** For each mini-batch, the gradient of the loss function is computed with respect to the model’s parameters.
- **Update Weights:** The weights are updated based on the average gradient across the mini-batch, instead of using individual samples (as in SGD) or the entire dataset (as in batch gradient descent).
- **Repeat for Each Epoch:** The process is repeated for all mini-batches and over multiple epochs until the model converges.

Advantages of Mini-batch Gradient Descent:

- **Computational Efficiency:** Mini-batches allow more efficient use of memory and faster computation compared to full batch gradient descent.
- **Generalization:** Training with mini-batches introduces noise into the gradient estimates, which can help escape local minima and lead to better generalization.
- **Convergence Speed:** Mini-batches strike a balance between the noisy updates of SGD and the slow convergence of batch gradient descent, speeding up convergence.