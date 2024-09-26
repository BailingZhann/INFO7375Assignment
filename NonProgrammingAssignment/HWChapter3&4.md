# HW to Chapters 2 “The Perceptron”
## Non-programming Assignment
### 1. Describe the Perceptron and how it works,

The Perceptron is one of the simplest types of artificial neural networks, developed for supervised learning of binary classifiers. It was introduced by Frank Rosenblatt in 1958 and is the foundation for more complex neural networks.

The Perceptron works by taking a weighted sum of the input features, passing it through an activation function (usually a step function or sign function), and producing a binary output (either 0 or 1).

**- Input:** The input to a Perceptron is a vector of features, x = [x1, x2, ..., xn].

**- Weights:** Each input has an associated weight w = [w1, w2, ..., wn].

**- Bias:** A bias term is added to shift the decision boundary.

**- Activation Function:** The Perceptron uses a step function that produces a 1 if the weighted sum is above a threshold (typically 0), otherwise, it outputs 0.

The mathematical equation of the Perceptron is:

y = 1 if (w ⋅ x + b) > 0

y = 0 otherwise

Where:

- w is the weight vector.
- x is the input vector.
- b is the bias term.
- y is the output (either 0 or 1).


### 2. What is forward and backpropagation propagation for the Perceptron?

**Forward Propagation:** Forward propagation in the Perceptron involves feeding input data into the network and passing it through the weighted sum and activation function to produce an output. It’s a simple linear combination followed by a threshold function that produces either 0 or 1.

**Backpropagation (Perceptron Training Rule):** Backpropagation, typically used in multilayer networks, involves adjusting weights to minimize the error in neural networks. However, in the original Perceptron (single-layer), learning happens via the Perceptron Learning Rule, which updates the weights as follows:

w_i = w_i + η (y_true - y_pred) x_i

Where:

- w_i is the weight associated with input x_i.
- y_true is the actual output.
- y_pred is the predicted output.
- η is the learning rate, a small constant that controls how much the weights should be adjusted.

The Perceptron updates its weights after each misclassified sample until all training samples are correctly classified or the maximum number of iterations is reached.

### 3. What is the history of the Perceptron?

The Perceptron was invented by Frank Rosenblatt in 1958 at the Cornell Aeronautical Laboratory. It was one of the earliest models for supervised learning of binary classifiers. Rosenblatt’s Perceptron gained widespread attention for its ability to learn from data and classify inputs.

However, the original Perceptron model faced a major limitation: it could only classify linearly separable data. This limitation was famously highlighted by Marvin Minsky and Seymour Papert in 1969 in their book "Perceptrons", where they showed that the Perceptron couldn’t solve problems like XOR (exclusive or) because XOR is not linearly separable.

This criticism significantly slowed research in neural networks until the development of multilayer Perceptrons and the backpropagation algorithm in the 1980s.

### 4. What is Supervised Training?

Supervised Training refers to the process of training a machine learning model using labeled data. In the context of the Perceptron, it means providing the Perceptron with input data along with the corresponding correct output (label). During training:

- The Perceptron makes predictions based on the input.
- The prediction is compared to the actual label (true output), and an error is calculated.
- The model adjusts its weights to reduce this error using a learning rule (like the Perceptron Learning Rule).
  
This process repeats iteratively until the model can accurately predict the labels for the training data or until the stopping criterion is met.

### 5. Why is Perceptron referred to as a binary linear classifier?

The Perceptron is referred to as a binary linear classifier for the following reasons:

**Binary Classifier:** The Perceptron is designed to solve binary classification problems, meaning it classifies data into one of two categories (e.g., 0 or 1).

**Linear Classifier:** The decision boundary of the Perceptron is a straight line (or hyperplane in higher dimensions). It separates the feature space into two regions, one for each class. A Perceptron can only solve problems where the two classes are linearly separable, meaning they can be divided by a straight line or hyperplane.

### 6. What are the disadvantages of binary linear classification?

The main disadvantages of binary linear classification include:

**- Limited to Linearly Separable Problems:** The Perceptron can only classify data that is linearly separable. It cannot solve problems where the classes cannot be separated by a straight line (e.g., XOR problem). Non-linearly separable problems require more complex models (e.g., multilayer neural networks).

**- No Probability Output:** The Perceptron produces hard decisions (0 or 1) without providing a probability of how confident it is about its classification, which limits its usefulness in some applications.

**- Sensitive to Outliers:** The decision boundary of a Perceptron can be sensitive to outliers in the data. A single misclassified point can affect the weight updates significantly and move the decision boundary too far.

**- Limited Expressiveness:** The Perceptron can only model simple linear relationships between inputs and outputs. More complex data structures, like XOR or multiclass classification, cannot be modeled by a single-layer Perceptron.

**- No Hidden Layers:** The Perceptron has no hidden layers, which limits its ability to capture complex relationships in data. Modern neural networks overcome this with multiple hidden layers and non-linear activation functions, which the Perceptron lacks.
