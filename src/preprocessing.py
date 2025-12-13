"""
Image Preprocessing Module

This module provides optimized image preprocessing functions for the animal classifier.
All functions are designed with efficiency in mind and use TensorFlow operations
for GPU acceleration when available.

Time Complexity Notes:
- Image resizing: O(H*W) where H, W are original height and width
- Normalization: O(n) where n is total number of pixels (224*224*3)
- File I/O operations are I/O bound, not CPU bound
"""

import tensorflow as tf
import numpy as np

# Constants
IMG_SIZE = (224, 224)  # MobileNetV2 standard input size


def preprocess_image(image):
    """
    Preprocesses a loaded image for model inference.
    
    This function applies essential preprocessing steps required by MobileNetV2:
    1. Resizes image to 224x224 (model input size)
    2. Normalizes pixel values from [0, 255] to [0, 1]
    
    Args:
        image: Input image as numpy array or TensorFlow tensor.
               Expected shape: (H, W, 3) with values in range [0, 255]
    
    Returns:
        tf.Tensor: Preprocessed image with shape (224, 224, 3) and values in [0, 1]
    
    Time Complexity:
        O(H*W) for resize + O(n) for normalization where n = 224*224*3 = 150,528
        Total: O(H*W + 150528) ≈ O(H*W) for large images
    
    Space Complexity:
        O(1) - In-place operations, no additional memory allocation
    
    Example:
        >>> img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        >>> processed = preprocess_image(img)
        >>> print(processed.shape)  # (224, 224, 3)
        >>> print(processed.numpy().min(), processed.numpy().max())  # 0.0, 1.0
    """
    # Resize using bilinear interpolation (default, fastest method)
    # Time: O(H*W) using optimized TensorFlow operations
    image = tf.image.resize(image, IMG_SIZE)
    
    # Normalize to [0, 1] range
    # Time: O(n) where n = 224*224*3 = 150,528 pixels
    # This is efficient as it's a simple element-wise division
    image = image / 255.0
    
    return image


def load_and_preprocess_image(path):
    """
    Loads an image from disk and applies preprocessing pipeline.
    
    This function combines file I/O, decoding, and preprocessing into a single
    optimized pipeline. Suitable for batch processing and can be used with
    tf.data.Dataset.map() for parallel processing.
    
    Args:
        path: String path to the image file (JPEG format supported)
    
    Returns:
        tf.Tensor: Preprocessed image ready for model inference
                   Shape: (224, 224, 3), Values: [0, 1]
    
    Time Complexity:
        O(I/O) + O(decode) + O(H*W) + O(n)
        - I/O: Disk read time (I/O bound, not CPU bound)
        - decode: JPEG decompression, depends on compression ratio
        - H*W: Resize operation
        - n: Normalization (150,528 operations)
    
    Space Complexity:
        O(H*W) - Stores original decoded image temporarily
    
    Note:
        When used with tf.data.Dataset.map(num_parallel_calls=AUTOTUNE),
        this function benefits from parallel processing across multiple CPU cores.
    
    Example:
        >>> img_tensor = load_and_preprocess_image('data/raw-img/cane/dog1.jpg')
        >>> print(img_tensor.shape)  # (224, 224, 3)
    """
    # Read file from disk (I/O bound operation)
    image = tf.io.read_file(path)
    
    # Decode JPEG to tensor
    # Time: Depends on image compression and size
    # Forces 3 channels (RGB) even if image is grayscale
    image = tf.image.decode_jpeg(image, channels=3)
    
    # Apply preprocessing pipeline
    image = preprocess_image(image)
    
    return image
