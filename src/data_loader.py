"""
Data Loading and Dataset Creation Module

This module handles efficient data loading using TensorFlow's optimized data pipeline.
It implements best practices for performance including:
- Parallel data loading
- Dataset caching
- Prefetching
- Data normalization

Performance Optimizations:
1. AUTOTUNE: Dynamically adjusts parallelism based on available CPU cores
2. Cache: Stores preprocessed data in memory after first epoch
3. Shuffle: Randomizes training data to prevent overfitting
4. Prefetch: Loads next batch while GPU processes current batch

Time Complexity:
- Initial loading: O(n) where n is number of images
- Subsequent epochs with cache: O(1) for data loading, only model training time
"""

import tensorflow as tf
import pathlib
import os

# Dataset Configuration
# These paths and parameters are optimized for the Animals-10 dataset
DATA_DIR = pathlib.Path('data/raw-img')  # Root directory containing class subdirectories

# Batch size of 32 provides good balance between:
# - Training speed (larger batches = fewer updates)
# - Memory usage (GPU/CPU RAM constraints)
# - Gradient stability (not too large, not too small)
BATCH_SIZE = 32

# MobileNetV2 standard input dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224


def get_class_names():
    """
    Retrieves class names from the dataset directory structure.
    
    This function scans the data directory and returns sorted class names.
    The sorting ensures consistent label assignment across different runs.
    
    Returns:
        list: Sorted list of class names (subdirectory names)
              For Animals-10: ['cane', 'cavallo', 'elefante', 'farfalla', 
                               'gallina', 'gatto', 'mucca', 'pecora', 
                               'ragno', 'scoiattolo']
    
    Raises:
        FileNotFoundError: If DATA_DIR does not exist
    
    Time Complexity:
        O(c log c) where c is number of classes (10 in Animals-10)
        - O(c) for directory scanning
        - O(c log c) for sorting
        In practice: O(10 log 10) = O(33) ≈ O(1) for fixed dataset
    
    Space Complexity:
        O(c) for storing class names list
    
    Example:
        >>> classes = get_class_names()
        >>> print(classes)
        ['cane', 'cavallo', 'elefante', ...]
        >>> print(len(classes))
        10
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found at {DATA_DIR.absolute()}")
    
    # List comprehension with filter: O(c) where c = number of items in directory
    # Only directories are kept (each class has its own subdirectory)
    class_list = [item.name for item in DATA_DIR.glob('*') if item.is_dir()]
    
    # Sorting ensures consistent class-to-label mapping
    # Critical for model training and inference
    return sorted(class_list)


def create_datasets():
    """
    Creates optimized TensorFlow datasets for training and validation.
    
    This function implements a high-performance data pipeline with:
    - Automatic 80/20 train-validation split
    - Image loading and resizing
    - Normalization to [0, 1] range
    - Memory caching for faster epochs
    - Data shuffling for training
    - Prefetching for GPU/CPU overlap
    
    Returns:
        tuple: (train_ds, val_ds, class_names)
            - train_ds: tf.data.Dataset for training (80% of data)
            - val_ds: tf.data.Dataset for validation (20% of data)
            - class_names: list of class labels
    
    Raises:
        FileNotFoundError: If DATA_DIR does not exist
    
    Time Complexity:
        Initial epoch: O(n) where n is total number of images (~26,000 for Animals-10)
        - Each image is loaded, decoded, resized: O(H*W) per image
        - With caching, subsequent epochs: O(1) for data loading
    
    Space Complexity:
        O(n * 224 * 224 * 3) for cached dataset in memory
        ≈ 26,000 * 150KB = 3.9GB for full Animals-10 dataset
        (Cached as preprocessed tensors, not raw images)
    
    Performance Notes:
        1. cache(): First epoch loads and caches data. Subsequent epochs reuse cache.
        2. shuffle(1000): Maintains 1000-image buffer for randomization.
           - Buffer size balances memory usage vs randomness
           - 1000 is good for datasets with 10K-100K samples
        3. prefetch(AUTOTUNE): Overlaps data loading with model training.
           - While GPU trains on batch N, CPU loads batch N+1
           - AUTOTUNE dynamically adjusts based on hardware
        4. Normalization via map(): Applied per-batch, very fast O(batch_size * pixels)
    
    Example:
        >>> train_ds, val_ds, classes = create_datasets()
        >>> print(f"Classes: {classes}")
        >>> print(f"Train batches: {len(train_ds)}")
        >>> print(f"Val batches: {len(val_ds)}")
        >>> for images, labels in train_ds.take(1):
        ...     print(f"Batch shape: {images.shape}")  # (32, 224, 224, 3)
        ...     print(f"Label shape: {labels.shape}")  # (32,)
        ...     print(f"Pixel range: [{images.numpy().min()}, {images.numpy().max()}]")
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found at {DATA_DIR.absolute()}")

    # Training Dataset (80% split)
    # image_dataset_from_directory automatically:
    # 1. Finds all images in subdirectories
    # 2. Assigns labels based on subdirectory names
    # 3. Resizes images to (IMG_HEIGHT, IMG_WIDTH)
    # 4. Creates batches of size BATCH_SIZE
    # Time: O(n) for initial scan, but actual loading is lazy
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,  # 20% for validation
        subset="training",
        seed=123,  # Fixed seed for reproducibility
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    # Validation Dataset (20% split)
    # Uses same seed to ensure no overlap between train and val
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,  # Same seed as training for consistent split
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    
    # Performance Optimization Pipeline
    # Order matters: cache -> shuffle -> prefetch for best performance
    AUTOTUNE = tf.data.AUTOTUNE  # Let TensorFlow optimize parallelism
    
    # Training dataset optimizations
    # cache(): O(n) first time, O(1) subsequent times
    # shuffle(1000): O(1) per batch (maintains 1000-sample buffer)
    # prefetch(AUTOTUNE): O(1) overlap, no additional time cost
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    
    # Validation dataset optimizations
    # No shuffle needed for validation (deterministic evaluation)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Normalization Layer
    # image_dataset_from_directory returns pixels in [0, 255] (uint8)
    # We need [0, 1] (float32) for neural network training
    # Rescaling(1./255) is equivalent to: pixel_value / 255.0
    # Time: O(batch_size * 224 * 224 * 3) per batch ≈ O(32 * 150K) ≈ O(4.8M) ops
    # This is very fast (microseconds) on modern CPUs
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    
    # Apply normalization to both datasets
    # map() applies function to each batch in parallel
    # Time: O(1) per batch with AUTOTUNE parallelization
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds, val_ds, class_names
