import numpy as np

def depthwise_convolution(image, kernel):
    """
    Perform depthwise convolution where each channel of the image is convolved
    with its corresponding channel in the kernel.
    """
    # Check that the number of channels in the image and kernel match
    num_channels = image.shape[-1]
    assert kernel.shape[-1] == num_channels, "Kernel and image must have the same number of channels"
    
    # Get dimensions for the output
    image_height, image_width, _ = image.shape
    kernel_height, kernel_width, _ = kernel.shape
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    
    # Initialize output with zeros for each channel
    output = np.zeros((output_height, output_width, num_channels))
    
    # Perform depthwise convolution for each channel separately
    for c in range(num_channels):
        for i in range(output_height):
            for j in range(output_width):
                region = image[i:i + kernel_height, j:j + kernel_width, c]
                output[i, j, c] = np.sum(region * kernel[:, :, c])
    
    return output

def pointwise_convolution(image, kernel):
    """
    Perform pointwise convolution where a 1x1 kernel is applied across all channels of the image.
    """
    # Get the number of channels in the image
    num_channels = image.shape[-1]
    
    # Ensure kernel has 1x1 spatial size and matches input channels
    assert kernel.shape[:2] == (1, 1), "Pointwise kernel should be of size 1x1"
    assert kernel.shape[2] == num_channels, "Kernel and image channels must match for pointwise convolution"
    
    # Initialize output, which will have as many channels as there are filters
    output_height, output_width = image.shape[0], image.shape[1]
    num_filters = kernel.shape[-1]
    output = np.zeros((output_height, output_width, num_filters))
    
    # Perform pointwise convolution
    for f in range(num_filters):
        output[:, :, f] = np.sum(image * kernel[0, 0, :, f], axis=-1)
    
    return output

def convolution(image, kernel, mode="depthwise"):
    """
    Main function to choose between depthwise and pointwise convolution based on mode.
    """
    if mode == "depthwise":
        return depthwise_convolution(image, kernel)
    elif mode == "pointwise":
        return pointwise_convolution(image, kernel)
    else:
        raise ValueError("Mode should be either 'depthwise' or 'pointwise'")

# Example input
# 3x3x3 image with 3 channels (e.g., RGB image)
image = np.array([
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
    [[1, 3, 5], [7, 9, 11], [13, 15, 17]]
])

# Depthwise 2x2 kernel for each channel
depthwise_kernel = np.array([
    [[1, -1, 2], [0, 1, -1]],
    [[-1, 0, 1], [2, -1, 0]]
])

# Pointwise 1x1 kernel with multiple filters (for example, converting 3 channels to 2)
pointwise_kernel = np.array([[[[1, 0.5], [-1, 0.3], [0.2, -0.4]]]])

# Perform depthwise convolution
depthwise_output = convolution(image, depthwise_kernel, mode="depthwise")
print("Depthwise Convolution Output:\n", depthwise_output)

# Perform pointwise convolution
pointwise_output = convolution(image, pointwise_kernel, mode="pointwise")
print("Pointwise Convolution Output:\n", pointwise_output)
