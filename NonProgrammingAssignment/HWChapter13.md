### Non-Programming Assignment

1. **What is convolution operation and how does it work?**

   Convolution is a mathematical operation applied to two functions, typically used to extract features from images in neural networks. In the context of image processing, the convolution operation slides a smaller matrix, called a filter or kernel, over the original image matrix to compute a weighted sum of the values within the filter’s field. Each sum then populates a corresponding position in the output matrix, effectively highlighting specific patterns or features, like edges, in the image.

2. **Why do we need convolutional layers in neural networks?**

   Convolutional layers are essential in neural networks, particularly Convolutional Neural Networks (CNNs), because they automatically and efficiently learn spatial hierarchies and patterns in data (e.g., edges, shapes, and textures in images). These layers enable a model to capture local dependencies and spatial features, which are crucial for tasks like image classification, object detection, and more. Convolutional layers reduce the need for manual feature engineering, making the model more effective at recognizing features.

3. **How are sizes of the original image, the filter, and the resultant convoluted image related?**

   The size of the convoluted image depends on the original image size, the filter size, and any padding and stride used. Without padding, the output image size is reduced. The formula to calculate the output dimension O is:
   O=(I−F+2P)/S+1

   where:
   - **I** is the size of the input (image),
   - **F** is the size of the filter,
   - **P** is the padding, and
   - **S** is the stride.

   Without padding and stride of 1, the output size decreases by the filter’s size minus one.

4. **What is padding and why is it needed?**

   Padding involves adding a border of zeros around the input image. It is needed to control the spatial size of the output feature map and helps preserve the original input dimensions when applying convolutions. Padding allows edge pixels to receive equal treatment and ensures that feature extraction near image boundaries is as effective as within the image center.

5. **What is strided convolution and why is it needed?**

   Strided convolution means shifting the filter across the image by steps greater than one (stride > 1). This reduces the spatial dimensions of the output feature map, decreasing computational requirements and controlling the receptive field. Strided convolutions provide a downsampling effect, which can be useful for reducing data dimensionality while maintaining important spatial features.
