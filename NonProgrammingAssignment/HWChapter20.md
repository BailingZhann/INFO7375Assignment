# HW to Chapter 20 “LM, LSTM, and GRU”

### 1. How does a language model (LM) work?
A language model (LM) is designed to predict the probability of a sequence of words. It works by learning patterns and structures in language data, using this knowledge to estimate the likelihood of a word or sequence of words given its context.

**Key Steps:**
- It calculates the conditional probability of a word based on previous words in a sequence.
- For example, in a sentence "The cat is on the ___," the LM predicts the next word (e.g., "mat") based on the context.

**Mathematical Representation:**
The probability of a sentence `S` = (w₁, w₂, ..., wₙ) is modeled as:

P(S) = P(w₁) * P(w₂ | w₁) * P(w₃ | w₁, w₂) * ... * P(wₙ | w₁, ..., wₙ₋₁)

### 2. How does word prediction work?
Word prediction is a task where the model predicts the next word in a sequence based on the previous context. 

**Steps:**
1. **Input Encoding:** The input sequence of words is converted into embeddings (numerical representations of words).
2. **Contextual Understanding:** Using a language model like RNN, LSTM, GRU, or Transformer, the context of the input is processed.
3. **Output Prediction:** The model calculates the probabilities for the next word using a softmax layer, and the word with the highest probability is selected.

For example:
- Input: "I love to play"
- Prediction: "football" (based on the model's learned patterns in the dataset).

### 3. How to train a language model (LM)?
Training a language model involves the following steps:
1. **Dataset Preparation:** Collect a large corpus of text data relevant to the domain.
2. **Tokenization:** Split the text into tokens (words, subwords, or characters).
3. **Model Architecture:** Use a suitable architecture (e.g., RNN, LSTM, GRU, or Transformer).
4. **Objective Function:** Use a loss function like Cross-Entropy to minimize the difference between predicted and actual words.
5. **Optimization:** Train the model using algorithms like stochastic gradient descent (SGD) or Adam.
6. **Evaluation:** Evaluate the model's performance using metrics like perplexity.

### 4. Describe the problem and the nature of vanishing and exploding gradients
**Vanishing Gradients:**
- In deep networks or RNNs, gradients shrink as they are propagated backward through the network during backpropagation.
- This results in extremely small updates to weights, causing the network to stop learning effectively.

**Exploding Gradients:**
- Opposite to vanishing gradients, exploding gradients occur when gradients grow uncontrollably large during backpropagation.
- This leads to unstable weight updates and may cause the model to diverge.

**Nature of the Problem:**
- Both issues arise due to repeated multiplication of gradients with weights in deep architectures.
- They are more prevalent in RNNs because of their sequential nature and the unrolling of time steps during training.

**Mitigation Techniques:**
- Use gradient clipping to handle exploding gradients.
- Use architectures like LSTM or GRU to mitigate vanishing gradients.

### 5. What is LSTM and the main idea behind it?
**Long Short-Term Memory (LSTM):**
- LSTM is a specialized type of RNN designed to address the vanishing gradient problem by introducing a memory cell that can retain information over long sequences.

**Main Idea:**
- LSTMs use gates (input, forget, and output gates) to control the flow of information, allowing them to learn long-term dependencies.

**Key Components:**
1. **Forget Gate:** Decides which information to discard from the cell state.
2. **Input Gate:** Determines which new information to add to the cell state.
3. **Output Gate:** Controls what information from the cell state is output.

**Advantages:**
- Effectively captures long-range dependencies.
- Mitigates the vanishing gradient problem.

### 6. What is GRU?
**Gated Recurrent Unit (GRU):**
- GRU is a simplified version of LSTM that uses fewer gates while retaining similar performance.

**Key Components:**
1. **Update Gate:** Combines the forget and input gates into a single mechanism.
2. **Reset Gate:** Controls how much of the past information to forget.

**Advantages of GRU:**
- Simpler architecture than LSTM (fewer parameters).
- Faster training and inference times.
- Suitable for tasks with less complex dependencies.
