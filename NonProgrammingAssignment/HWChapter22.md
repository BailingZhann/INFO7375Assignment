# Homework: Chapter 22 – Attention, Transformers, LLM, GenAI, GPT, Diffusion
## Non-programming Assignment

### 1. Describe the Attention Problem
The **attention problem** addresses the challenge of focusing on relevant parts of the input sequence while processing long or complex sequences. Traditional models like RNNs struggle with long-range dependencies, leading to performance degradation. Attention mechanisms provide a way to dynamically assign importance to input features based on their relevance to the current task.

---

### 2. What is the Attention Model?
The **attention model** is a neural mechanism that enhances sequence-to-sequence tasks by assigning varying weights to input elements. It computes an attention score for each input, normalizes these scores, and uses them to create a weighted sum of the inputs.

### **Attention Scoring Mechanism**:
The score for a query `q` and a key `k` is calculated as:  
score(q, k) = (q ⋅ k) / √(d_k)

Where:
- \(q\): Query vector.
- \(k\): Key vector.
- \(d_k\): Dimensionality of the key vectors (used for scaling).

The scores are passed through a softmax function to obtain attention weights.

---

### 3. Describe the Attention Model for Speech Recognition
In speech recognition, the attention model aligns audio features with text outputs.

### Process:
1. **Input**: Audio features (e.g., spectrogram frames).
2. **Scoring**: Attention scores between decoder states and encoder outputs are computed.
3. **Weighted Summation**: Combine the audio features using attention weights.
4. **Output**: Text is generated sequentially using the context vector and decoder state.

This approach handles variable-length input and aligns text with non-linear audio features.

---

### 4. How Does Trigger Word Detection Work?
Trigger word detection identifies specific words (e.g., "Alexa") in audio streams.

### Steps:
1. **Audio Features**: Convert audio into spectrograms.
2. **Model**: Use a neural network (e.g., CNN or RNN) to classify audio segments.
3. **Sliding Window**: Continuously process overlapping segments of audio to detect the trigger word.

### Example Table:
| Feature   | Description                     |
|-----------|---------------------------------|
| Trigger   | "Hey Google"                   |
| Model     | Recurrent Neural Network (RNN) |
| Output    | Binary decision (trigger/no trigger) |

---

### 5. What is the Idea of Transformers?
Transformers are models that solve sequence-to-sequence tasks using **self-attention** and **parallel processing**. Key ideas:
- Replace recurrence with attention mechanisms.
- Use **positional encodings** to preserve sequence order.
- Scale efficiently to large datasets.

---

### 6. What is Transformer Architecture?
### Components of Transformer Architecture:
1. **Encoder**:
   - Multi-head self-attention.
   - Feed-forward layers.
   - Layer normalization.

2. **Decoder**:
   - Multi-head self-attention.
   - Cross-attention (uses encoder output).
   - Feed-forward layers.

### Formula for Scaled Dot-Product Attention:

The formula for attention is calculated as:  
Attention(Q, K, V) = softmax((QK^T) / √(d_k))V

Where:
- \(Q\): Query matrix.
- \(K\): Key matrix.
- \(V\): Value matrix.
- \(d_k\): Dimensionality of the key vectors.

---

### 7. What is the LLM?
**Large Language Models (LLMs)** are transformer-based models trained on massive datasets to perform language-related tasks.

### Examples Table:
| Model   | Purpose                  |
|---------|--------------------------|
| GPT     | Text generation          |
| BERT    | Text understanding       |
| ChatGPT | Conversational AI        |

---

### 8. What is Generative AI?
**Generative AI** refers to models that generate new, coherent data, such as:
- Text (e.g., GPT).
- Images (e.g., Stable Diffusion).
- Audio (e.g., WaveNet).

---

### 9. What Are the Core Functionalities of Generative AI?
1. **Content Creation**: Generate text, images, and videos.
2. **Data Augmentation**: Enhance training datasets.
3. **Automation**: Reduce human effort in repetitive tasks.
4. **Personalization**: Provide user-specific outputs.

---

### 10. What is GPT and How Does It Work?
**GPT (Generative Pre-trained Transformer)** is a transformer-based model that generates human-like text.

### Key Steps:
1. **Pre-training**: Predict the next word in a sequence using large datasets.
2. **Fine-tuning**: Adapt to specific tasks with labeled data.
3. **Self-Attention**: Captures long-range dependencies in text.

---

### 11. What is the Concept of the Diffusion Network?
**Diffusion Networks** generate data by reversing the process of adding noise.

### Process:
1. **Forward Diffusion**: Gradually add noise to input data until it resembles random noise.
2. **Reverse Diffusion**: Learn to remove noise step-by-step to reconstruct data.

### Applications Table:
| Field         | Use Case                 |
|---------------|--------------------------|
| Image Gen     | DALL·E, Stable Diffusion |
| Video Gen     | Synthesizing animations  |

### Formula for Reverse Diffusion:

The formula for reverse diffusion is:  
p_θ(x_{t-1} | x_t) = 𝒩(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))

Where:
- `x_t`: Noisy data at step `t`.  
- `μ_θ`: Predicted mean.  
- `Σ_θ`: Predicted variance. 
