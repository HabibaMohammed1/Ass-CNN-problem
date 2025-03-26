import numpy as np

# ReLU activation function
def relu(x):
    """Applies the ReLU activation function."""
    return np.maximum(0, x)

# 2D Convolution with stride and padding
def conv2d(image, kernel, stride=2, padding=1):
    """
    Applies a 2D convolution operation on an image with a given kernel.
    
    Args:
        image (ndarray): Input image matrix.
        kernel (ndarray): Filter kernel.
        stride (int, optional): The stride of the convolution. Default is 2.
        padding (int, optional): Padding to be added to the image. Default is 1.
    
    Returns:
        ndarray: The output after applying convolution and ReLU activation.
    """
    # Pad the image to ensure proper output size after convolution
    image_padded = np.pad(image, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)
    
    kernel_size = kernel.shape[0]
    output_size = (image_padded.shape[0] - kernel_size) // stride + 1
    
    # Initialize output matrix
    output = np.zeros((output_size, output_size))
    
    # Perform convolution operation
    for i in range(output_size):
        for j in range(output_size):
            output[i, j] = np.sum(
                image_padded[i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size] * kernel
            )
    
    # Apply ReLU activation function
    return relu(output)

# Max pooling operation
def max_pooling(image, pool_size=2, stride=2):
    """
    Applies max pooling on the image.
    
    Args:
        image (ndarray): Input image matrix.
        pool_size (int, optional): The size of the pooling window. Default is 2.
        stride (int, optional): The stride of the pooling operation. Default is 2.
    
    Returns:
        ndarray: The output after applying max pooling.
    """
    output_size = (image.shape[0] - pool_size) // stride + 1
    output = np.zeros((output_size, output_size))
    
    # Perform max pooling operation
    for i in range(output_size):
        for j in range(output_size):
            output[i, j] = np.max(image[i * stride:i * stride + pool_size, j * stride:j * stride + pool_size])
    
    return output

# Example input image matrices (R, G, B channels)
R = np.array([
    [112, 125, 25, 80, 220, 110],
    [150, 95, 15, 100, 115, 152],
    [200, 100, 48, 90, 70, 175],
    [187, 56, 43, 86, 180, 200],
    [190, 87, 70, 37, 24, 35],
    [80, 75, 65, 45, 32, 20]
])

G = np.array([
    [150, 125, 38, 80, 20, 10],
    [130, 95, 25, 100, 115, 152],
    [80, 100, 148, 90, 70, 175],
    [170, 160, 43, 160, 170, 180],
    [100, 150, 70, 37, 124, 135],
    [85, 75, 65, 45, 232, 120]
])

B = np.array([
    [200, 125, 25, 80, 220, 150],
    [50, 95, 15, 150, 115, 152],
    [90, 110, 48, 190, 70, 175],
    [180, 135, 43, 106, 180, 110],
    [55, 98, 70, 37, 24, 35],
    [78, 150, 65, 45, 32, 80]
])

# Filter kernels for each color channel
filter_R = np.array([
    [1, 1, 1, 0],
    [0, 1, 1, 1],
    [-1, 0, 0, 1],
    [-1, 0, 1, -1]
])

filter_G = np.array([
    [0, -1, -1, 0],
    [1, -1, 1, -1],
    [1, 0, 0, 1],
    [1, 0, 1, 1]
])

filter_B = np.array([
    [1, 1, 1, 0],
    [-1, 1, 1, 1],
    [0, 1, 0, 1],
    [-1, -1, 1, 1]
])

# Display input images and filters
print("Input Image Matrices:")
print("Red Channel (R):")
print(R)
print("\nGreen Channel (G):")
print(G)
print("\nBlue Channel (B):")
print(B)

print("\nFilter Kernels:")
print("Red Channel Kernel:")
print(filter_R)
print("\nGreen Channel Kernel:")
print(filter_G)
print("\nBlue Channel Kernel:")
print(filter_B)

# Perform convolution and pooling for each channel and store results
final_matrices = {}

print("\nConvolution and Max Pooling Results:")
for name, image, kernel in zip(["Red", "Green", "Blue"], [R, G, B], [filter_R, filter_G, filter_B]):
    conv_output = conv2d(image, kernel)
    pooled_output = max_pooling(conv_output)
    final_matrices[name] = pooled_output
    
    # Display results for each channel
    print(f"\n{name} Channel Convolution Output:")
    print(conv_output)
    print(f"\n{name} Channel Max Pooling Output:")
    print(pooled_output)
    print(f"\n{name} Channel Flatten Layer Size:", pooled_output.size)

# Display final output matrices for each channel
print("\nFinal 3x3 Output Matrices:")
for name, matrix in final_matrices.items():
    print(f"\n{name} Channel Final 3x3 Matrix:")
    print(matrix)
