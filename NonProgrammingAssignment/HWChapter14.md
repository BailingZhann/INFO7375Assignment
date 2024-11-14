### Non-Programming Assignment

1. **What is pooling layer and how it works?**

   A pooling layer is a type of layer in a convolutional neural network (CNN) that reduces the spatial dimensions (width and height) of the input feature maps. Pooling helps to decrease the computational complexity, reduce memory usage, and prevent overfitting. The most common pooling operations are **Max Pooling** and **Average Pooling**:
   
   - **Max Pooling** selects the maximum value from each region of the feature map.
   - **Average Pooling** computes the average value from each region of the feature map.
   
   Pooling works by dividing the input feature map into small regions (e.g., 2x2) and applying the pooling operation to each region. This reduces the size of the output feature map while retaining important features.

2. **What are three major types of layers in the convolutional neural network?**

   The three major types of layers in a convolutional neural network are:
   
   - **Convolutional Layer**: This layer applies a set of learnable filters (kernels) to the input image to extract spatial features. The convolutional layer helps in detecting patterns such as edges, textures, and more complex shapes as we move deeper into the network.
   
   - **Pooling Layer**: The pooling layer reduces the spatial dimensions of the feature maps, which decreases the number of parameters and computations in the network. It also helps to make the representation more robust by focusing on the most prominent features.
   
   - **Fully Connected Layer**: This layer connects every neuron in one layer to every neuron in the next layer. It is typically used at the end of a CNN for classification tasks, where it combines all extracted features to make a final prediction.

3. **What is the architecture of a convolutional network?**

   The architecture of a convolutional neural network typically consists of a series of **convolutional layers**, **pooling layers**, and **fully connected layers** arranged in a specific sequence. Here’s a common architecture outline:

   - **Input Layer**: Takes in the raw image data, usually as a 2D or 3D array (height, width, color channels).
   
   - **Convolutional and Pooling Layers**: Alternating convolutional layers and pooling layers are used to progressively extract and downsample features from the input image. Each convolutional layer detects patterns, and the pooling layer reduces spatial dimensions, which helps control overfitting and reduces computational load.
   
   - **Flatten Layer**: Converts the 2D feature maps into a 1D vector to be fed into the fully connected layers.
   
   - **Fully Connected Layers**: After flattening, the feature map is passed through one or more fully connected layers to perform high-level reasoning. This part of the network functions similarly to a traditional feedforward neural network.
   
   - **Output Layer**: The final fully connected layer, often using a softmax activation function, produces the probability distribution over the target classes for classification tasks.
   
   In summary, a typical CNN architecture might look like: `Input → [Convolution + Pooling] × N → Flatten → Fully Connected → Output`.
