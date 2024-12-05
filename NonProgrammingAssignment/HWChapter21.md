# HW to Chapter 21 “NLP and Machine Translation”
## Non-programming Assignment

### 1. Describe Word Embedding
Word embedding refers to the process of representing words in a continuous vector space where words with similar meanings are mapped to nearby points. It captures semantic relationships between words by encoding them as dense, low-dimensional vectors as opposed to sparse one-hot encodings. Examples of word embedding methods include:

- **Word2Vec**: Utilizes Skip-Gram and Continuous Bag of Words (CBOW) models to learn word embeddings.
- **GloVe**: Focuses on capturing word relationships based on word co-occurrence statistics in a corpus.

Word embeddings improve performance in natural language processing tasks by preserving syntactic and semantic properties in the vector representation.

---

### 2. What is the Measure of Word Similarity?
Word similarity measures the degree of relatedness between two words. It is often calculated using vector operations on word embeddings. Common measures include:

- **Cosine Similarity**:
  Similarity between two word vectors \(w_1\) and \(w_2\) is calculated as:  

  Similarity(w₁, w₂) = (w₁ ⋅ w₂) / (|w₁| |w₂|)  

  where \(w₁\) and \(w₂\) are word vectors.

- **Euclidean Distance**: Measures the straight-line distance between vectors but is less common in word similarity tasks.
- **Dot Product**: Used in some models like Word2Vec to measure similarity directly.

---

### 3. Describe the Neural Language Model
A Neural Language Model (NLM) is a type of language model that uses neural networks to predict the likelihood of a sequence of words. Key components include:

1. **Input Layer**: Encodes words into embeddings.
2. **Hidden Layer(s)**: Processes the sequence context using architectures like:
   - Recurrent Neural Networks (RNNs)
   - Long Short-Term Memory (LSTM) networks
   - Transformers
3. **Output Layer**: Predicts probabilities of the next word in the sequence.

**Objective**: Maximize the probability of a word given its context:  

  P(wₜ | wₜ₋₁, wₜ₋₂, ..., w₁)

---

### 4. What is Bias in Word Embedding and How to Do Debiasing?
**Bias in Word Embedding** refers to the unintended encoding of societal biases (e.g., gender, race) in the vector representations. For example, vectors for "man" and "woman" may align with stereotypical professions like "engineer" and "nurse."

### **Debiasing Techniques**:
1. **Hard Debiasing**:
   - Identify the bias direction (e.g., gender direction).
   - Neutralize word vectors along the bias axis and equalize pairs of words (e.g., "doctor" and "nurse").
   
2. **Soft Debiasing**:
   - Adjust embeddings to reduce bias without completely removing the bias direction, allowing flexibility for specific tasks.

3. **Contextual Debiasing**:
   - Use contextualized embeddings (e.g., BERT) to reduce bias through dynamic representations.

---

### 5. How Does Modern Machine Translation Work Using the Language Model?
Modern machine translation relies on **transformer-based neural models** and techniques like:

1. **Encoder-Decoder Architecture**:
   - The **encoder** processes the input sentence into a sequence of contextualized embeddings.
   - The **decoder** generates the translated sentence based on the encoder output.

2. **Attention Mechanisms**:
   - Focus on relevant parts of the input sequence for translation.

3. **Pretrained Models**:
   - Models like BERT, GPT, and OpenAI's GPT series are used for transfer learning, improving translation quality.

4. **Transformer Models**:
   - State-of-the-art architectures like **Transformer** (Vaswani et al., 2017) enable parallel processing and better handling of long-range dependencies.

---

### 6. What is Beam Search?
Beam Search is a heuristic search algorithm used in sequence generation tasks (e.g., translation, text summarization). It explores multiple possible sequences and selects the most probable one based on a predefined beam width.

### **Algorithm**:
1. Start with the initial state (e.g., `<start>` token).
2. Expand possible next states and compute scores (e.g., log probabilities).
3. Retain the top \(k\) sequences (where \(k\) is the beam width).
4. Repeat until the sequence ends or a stopping criterion is met.

Beam Search balances exploration and exploitation, improving the quality of generated sequences compared to greedy search.

---

### 7. What is the BLEU Score?
The **BLEU (Bilingual Evaluation Understudy)** score is a metric for evaluating the quality of machine-translated text compared to a reference translation. It measures:

1. **Precision** of n-grams in the candidate translation relative to the reference.
2. **Brevity Penalty** to penalize overly short translations.

### **Formula**:
The BLEU score is calculated as:  

BLEU = BP ⋅ exp(∑ₙ₌₁ⁿ wₙ log Pₙ)  

- \(Pₙ\): Precision of n-grams.  
- \(wₙ\): Weight for n-grams (commonly uniform).  
- \(BP\): Brevity Penalty:  

  BP =  
  - 1, if \(c > r\)  
  - exp(1 - r/c), if \(c \leq r\)  

  where \(c\) is the candidate length and \(r\) is the reference length.

BLEU score ranges from 0 to 1, with higher scores indicating better translation quality.
