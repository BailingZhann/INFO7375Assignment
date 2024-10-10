# Quiz 3 - Chapter 9 - “Bias vs Variance“
## 1. Describe the “bias vs variance” problem.
- **Bias:** 
Bias refers to the error introduced by approximating a real-world problem, which may be highly complex, using a simplified model.
High bias typically occurs when a model is too simple and underfits the data, meaning it does not capture the underlying patterns well. For example, using a linear model to represent a non-linear relationship.
High bias leads to systematic errors in the model, causing poor performance both on the training data and new data.
- **Variance:**
Variance refers to the error introduced by the model's sensitivity to the small fluctuations or noise in the training data.
High variance typically occurs when a model is too complex and overfits the data, meaning it captures not only the underlying patterns but also the noise or random variations in the training set.
High variance leads to large fluctuations in model predictions on different datasets, causing good performance on the training data but poor generalization to new data.
- **Bias-Variance Trade-off:**
The key challenge in machine learning is finding the right balance between bias and variance.

High bias leads to underfitting, where the model is too simple to represent the data accurately.

High variance leads to overfitting, where the model is too complex and fails to generalize well to unseen data.
The goal is to minimize the total error, which consists of:
- **Bias error:** the error from incorrect assumptions in the model.
- **Variance error:** the error from the model's sensitivity to small variations in the training data.
- **Irreducible error:** the inherent noise in the data that cannot be reduced by any model.