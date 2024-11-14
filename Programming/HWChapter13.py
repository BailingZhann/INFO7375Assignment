import numpy as np

def convolve_2d(image, kernel):
    # Get dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calculate dimensions of the output
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    
    # Initialize output matrix with zeros
    output = np.zeros((output_height, output_width))
    
    # Perform convolution
    for i in range(output_height):
        for j in range(output_width):
            # Extract the region of interest in the image
            region = image[i:i + kernel_height, j:j + kernel_width]
            # Perform element-wise multiplication and sum the result
            output[i, j] = np.sum(region * kernel)
    
    return output

# Define a 6x6 example image
image = np.array([
    [1, 2, 3, 0, 1, 2],
    [4, 5, 6, 1, 2, 3],
    [7, 8, 9, 2, 3, 4],
    [1, 0, 1, 3, 4, 5],
    [2, 1, 2, 4, 5, 6],
    [3, 2, 3, 5, 6, 7]
])

# Define a 3x3 filter (kernel)
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

# Perform the convolution
output = convolve_2d(image, kernel)

# Print the resulting convoluted output
print("Convolved Output:\n", output)
