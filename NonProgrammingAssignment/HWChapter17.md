### Non-Programming Assignment

1. **What are anchor boxes and how do they work?**

   Anchor boxes are predefined bounding boxes of various sizes and aspect ratios that are used in object detection models to detect objects of different scales within an image. Each anchor box is placed at different locations across the image, and the model learns to predict which anchor box best matches the object in that region. The model adjusts the size and location of the anchor boxes during training to align with the ground truth bounding boxes, allowing it to detect objects with varying shapes and sizes.

2. **What is bounding box prediction and how does it work?**

   Bounding box prediction is the process by which an object detection model predicts the coordinates of a bounding box around an object within an image. The model learns to output the (x, y) coordinates of the box’s center, its width, and height relative to the anchor box or grid cell. During training, the model minimizes the error between predicted and actual bounding boxes, often using a loss function based on **IoU (Intersection over Union)** or regression-based losses like L1 or smooth L1 loss.

3. **Describe R-CNN**

   **R-CNN (Regions with Convolutional Neural Networks)** is a two-stage object detection model that combines region proposals with convolutional neural networks for object classification and localization. The R-CNN pipeline involves:
   
   - Generating **region proposals** using methods like selective search to identify potential object locations.
   - Extracting **features** from each region proposal using a CNN.
   - Classifying each proposal using a separate classifier (e.g., SVM) and refining bounding boxes to localize objects.
   
   R-CNN improved accuracy in object detection but is computationally expensive and slow due to the need to process each region proposal independently.

4. **What are advantages and disadvantages of R-CNN?**

   - **Advantages**:
     - R-CNN provides accurate object detection by focusing on region proposals, which reduces false positives.
     - It paved the way for modern object detection methods, inspiring faster and more efficient models like Fast R-CNN and Faster R-CNN.
   
   - **Disadvantages**:
     - R-CNN is computationally expensive because each region proposal is processed independently through the CNN, making it unsuitable for real-time applications.
     - The model has a complex pipeline and requires multiple steps (feature extraction, classification, bounding box regression) which are slow to train and deploy.

5. **What is semantic segmentation?**

   Semantic segmentation is a computer vision task that involves classifying each pixel in an image into a specific category, thereby segmenting the image into meaningful regions. Unlike object detection, which provides bounding boxes, semantic segmentation provides a pixel-level understanding of the scene, identifying objects’ exact shapes and boundaries. It is used in applications like autonomous driving, medical imaging, and scene understanding.

6. **How does deep learning work for semantic segmentation?**

   Deep learning models for semantic segmentation use fully convolutional networks (FCNs) to predict pixel-wise class labels. Typically, these models consist of:
   
   - **Encoder**: A convolutional neural network that extracts high-level features from the input image.
   - **Decoder**: A series of transposed convolutions (or upsampling layers) that expand feature maps back to the original image size to produce pixel-wise predictions.
   
   Deep learning models are trained to minimize a pixel-wise classification loss (like cross-entropy), enabling the network to learn fine-grained details for each class.

7. **What is transposed convolution?**

   Transposed convolution, also known as **deconvolution** or **upsampling convolution**, is a type of convolution operation used to increase the spatial resolution of feature maps. It works by reversing the downsampling effect of standard convolutions, allowing the model to generate high-resolution output from lower-resolution feature maps. Transposed convolution is commonly used in the decoder stage of segmentation models to upsample feature maps back to the original input size.

8. **Describe U-Net**

   **U-Net** is a deep learning architecture for semantic segmentation that was initially designed for biomedical image analysis. It has an encoder-decoder structure with skip connections that directly link corresponding layers in the encoder and decoder. This structure allows U-Net to capture both local and global features, preserving fine-grained spatial information while increasing spatial resolution in the output. Key characteristics of U-Net include:
   
   - **Encoder**: Downsamples the input image by extracting features through a series of convolutions and pooling layers.
   - **Decoder**: Upsamples the feature maps using transposed convolutions, gradually reconstructing the original image resolution.
   - **Skip Connections**: Connects each encoder layer to the corresponding decoder layer, allowing the model to retain spatial details for more accurate segmentation.
   
   U-Net is widely used in medical imaging and other applications that require precise, pixel-level segmentation.
