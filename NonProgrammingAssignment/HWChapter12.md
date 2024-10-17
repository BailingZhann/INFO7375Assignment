# HW to Chapter 12 “Softmax”
## Non-programming Assignment
### 1. What is the reason for softmax?

Softmax is used to transform raw output scores (logits) from a neural network into probabilities, making it suitable for multi-class classification problems. The main reasons for using softmax are:

- **Convert scores into probabilities:** Softmax ensures that all output values sum to 1, making them interpretable as probabilities.
- **Handle multi-class classification:** In tasks with multiple classes (e.g., classifying an image into one of many categories), softmax allows us to model the probability distribution over all possible classes.

### 2. What is softmax and how does it works?

- **Softmax Function:** Softmax is a mathematical function that converts a vector of raw scores (logits) into probabilities. It squashes the output values between 0 and 1 while maintaining the relative magnitude of each value.

The softmax formula for the i-th class is:

Softmax(z_i) = exp(z_i) / sum(exp(z_j)) for all j

Where:

- "z_i" is the raw score (logit) for the i-th class.
- The denominator is the sum of the exponentials of all logits.

How it works:

- **Exponentiation:** Each logit is transformed by taking the exponential, which ensures all values are positive and makes larger logits even larger.
- **Normalization:** The exponentiated values are then divided by the sum of all exponentiated values, converting them into probabilities that sum to 1.

**Example:** Suppose a model outputs raw scores for three classes as [2.0, 1.0, 0.1]. Applying softmax would convert these scores into probabilities:

- First, calculate exponentials for each score: exp(2.0) = 7.39, exp(1.0) = 2.72, and exp(0.1) = 1.11.
- Then, sum the exponentials: 7.39 + 2.72 + 1.11 = 11.22.
- Finally, divide each exponential by the sum:
  - 7.39 / 11.22 = 0.66,
  - 2.72 / 11.22 = 0.24,
  - 1.11 / 11.22 = 0.10.

In this case, the first class has the highest probability (66%), making it the most likely class.