### Non-Programming Assignment

1. **What is face verification and how does it work?**

   Face verification is a biometric authentication technique that confirms whether two images belong to the same person. It works by comparing a new image (test image) with a known image (reference image) of the same individual. The system extracts facial features from both images, computes a similarity score, and determines if they match based on a predefined threshold. If the similarity score exceeds the threshold, the system verifies the identity; otherwise, it rejects it.

2. **Describe the difference between face verification and face recognition.**

   - **Face Verification**: This is a one-to-one comparison where the system checks if two images represent the same individual. It is primarily used for authentication tasks, such as unlocking a device or verifying identity.
   
   - **Face Recognition**: This is a one-to-many comparison where the system identifies a person by comparing an image to a database of known faces. Face recognition is commonly used for identification purposes, such as tagging people in photos or finding suspects in security footage.

3. **How do you measure similarity of images?**

   Image similarity is typically measured using distance metrics on feature representations extracted from the images. Common methods include:
   
   - **Euclidean Distance**: Measures the straight-line distance between feature vectors of two images.
   - **Cosine Similarity**: Calculates the cosine of the angle between two feature vectors, indicating similarity regardless of vector magnitude.
   - **L2 Norm**: Computes the squared Euclidean distance, commonly used in deep learning to measure differences between feature vectors.
   
   Deep learning models for face recognition often use embeddings (numeric representations of images), and similarity is measured by calculating the distance between these embeddings.

4. **Describe Siamese networks.**

   Siamese networks are neural networks that consist of two identical subnetworks sharing the same weights and architecture. They are designed to learn similarity between two inputs, typically by mapping them to a common embedding space. During training, Siamese networks receive pairs of inputs and learn to output similar embeddings for similar pairs and dissimilar embeddings for different pairs. They are widely used in applications like face verification, image similarity, and signature verification.

5. **What is triplet loss and why is it needed?**

   **Triplet Loss** is a loss function used to train models to distinguish between similar and dissimilar images. It requires three inputs:
   
   - **Anchor**: The reference image.
   - **Positive**: An image of the same class as the anchor.
   - **Negative**: An image of a different class.
   
   Triplet loss minimizes the distance between the anchor and positive images while maximizing the distance between the anchor and negative images. This helps the model learn a more discriminative feature space, where similar images are closer together and dissimilar images are farther apart. Triplet loss is commonly used in face recognition to improve the robustness of embeddings.

6. **What is neural style transfer (NST) and how does it work?**

   Neural Style Transfer (NST) is a technique that applies the artistic style of one image (style image) to another image (content image) by combining their features. NST works by optimizing a generated image to minimize two loss functions:
   
   - **Content Loss**: Ensures the generated image retains the content of the content image.
   - **Style Loss**: Ensures the generated image reflects the style of the style image.
   
   NST uses a pre-trained convolutional neural network (e.g., VGG) to extract content and style features from the images, iteratively updating the generated image until it achieves a visually pleasing blend of content and style.

7. **Describe style cost function.**

   The **style cost function** quantifies how well the style of the generated image matches the style of the style image. It is computed by comparing the **Gram matrices** of the feature maps of the generated and style images at various layers of a convolutional neural network. The Gram matrix captures the correlations between feature maps, which represent textures and patterns. The style cost function minimizes the difference between these Gram matrices, allowing the generated image to mimic the textures, colors, and patterns of the style image.
