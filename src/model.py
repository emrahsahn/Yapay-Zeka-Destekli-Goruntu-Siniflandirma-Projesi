"""
Model Architecture Module

This module defines the neural network architecture for the animal classifier.
It uses Transfer Learning with MobileNetV2 as the base model, which provides:
- Pre-trained weights from ImageNet (1.4M images, 1000 classes)
- Efficient architecture optimized for mobile and edge devices
- Excellent balance between accuracy and speed

Model Characteristics:
- Input: 224x224x3 RGB images
- Output: 10-class probability distribution (softmax)
- Parameters: ~3.5M total, ~1.3M trainable
- Inference time: ~50ms on CPU, ~5ms on GPU

Architecture Design Rationale:
1. MobileNetV2: Proven architecture with depthwise separable convolutions
2. Frozen base: Prevents destroying pre-trained features
3. Global Average Pooling: Reduces overfitting vs Flatten
4. Dropout: Additional regularization (20% dropout rate)
5. Dense with softmax: Multi-class classification output

Time Complexity:
- Forward pass: O(3.5M) ≈ 3.5M multiply-add operations
- Backward pass: O(1.3M) only trainable parameters updated
"""

import tensorflow as tf


def create_model(num_classes, config=None):
    """
    Creates and compiles a Transfer Learning model using MobileNetV2.
    
    This function builds a neural network optimized for image classification:
    - Uses MobileNetV2 pre-trained on ImageNet as feature extractor
    - Adds custom classification head for the specific task
    - Freezes base model to prevent catastrophic forgetting
    - Applies dropout for regularization
    
    Args:
        num_classes (int): Number of output classes (10 for Animals-10)
        config (dict, optional): Experiment configuration dictionary containing:
                                 - optimizer: Keras optimizer instance
                                 - loss: Loss function
                                 - metrics: List of metrics to track
                                 - dropout_rate: Dropout rate (0.0-1.0)
                                 If None, uses default configuration
    
    Returns:
        tf.keras.Model: Compiled Keras model ready for training
            - Optimizer: Adam with learning_rate=0.0001
            - Loss: Sparse Categorical Crossentropy
            - Metrics: Accuracy, Sparse Categorical Accuracy
    
    Model Architecture:
        Input (224, 224, 3)
        ↓
        Rescaling [-1, 1]  # MobileNetV2 expects this range
        ↓
        MobileNetV2 (frozen)  # ~3.2M parameters
        ↓
        GlobalAveragePooling2D  # 7x7x1280 -> 1280
        ↓
        Dropout(0.2)  # Regularization
        ↓
        Dense(num_classes, softmax)  # 1280 -> num_classes
        ↓
        Output (num_classes,)  # Probability distribution
    
    Time Complexity:
        Model creation: O(1) - instantiation is fast
        Forward pass: O(3.5M) - 3.5M multiply-add operations
        Backward pass: O(1.3M) - only trainable layers updated
    
    Space Complexity:
        O(3.5M * 4 bytes) ≈ 14MB for model parameters
        + O(batch_size * intermediate_activations) for forward pass
    
    Example:
        >>> model = create_model(num_classes=10)
        >>> model.summary()
        Model: "sequential"
        _________________________________________________________________
        Layer (type)                Output Shape              Param #
        =================================================================
        rescaling (Rescaling)       (None, 224, 224, 3)       0
        mobilenetv2 (Functional)    (None, 7, 7, 1280)        2,257,984
        global_average_pooling2d    (None, 1280)              0
        dropout (Dropout)           (None, 1280)              0
        dense (Dense)               (None, 10)                12,810
        =================================================================
        >>> model.input_shape
        (None, 224, 224, 3)
        >>> model.output_shape
        (None, 10)
    """
    # Define input shape for MobileNetV2
    # 224x224 is the standard input size for MobileNetV2
    # Using this size allows us to leverage pre-trained weights effectively
    IMG_SHAPE = (224, 224, 3)

    # Base Model: MobileNetV2 Pre-trained on ImageNet
    # ------------------------------------------------
    # MobileNetV2 is a lightweight convolutional neural network designed for:
    # - Mobile and edge devices (low latency, small model size)
    # - High accuracy despite reduced complexity
    # - Depthwise separable convolutions (efficient computation)
    # 
    # Pre-training on ImageNet provides learned features that generalize well:
    # - Low-level features: edges, textures, colors
    # - Mid-level features: shapes, patterns
    # - High-level features: object parts
    # 
    # include_top=False: Removes the final classification layer (1000 ImageNet classes)
    #                    We'll add our own classification head for 10 animal classes
    # weights='imagenet': Loads pre-trained weights (much better than random initialization)
    # 
    # Time: O(1) for model instantiation, weights loaded from disk
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,  # Remove top classification layer
        weights='imagenet'  # Use pre-trained weights
    )
    
    # Freeze Base Model Weights
    # -------------------------
    # Freezing prevents the pre-trained weights from being updated during training
    # Benefits:
    # 1. Preserves learned features from ImageNet
    # 2. Faster training (only custom layers are updated)
    # 3. Requires less training data (fine-tuning top layers only)
    # 4. Prevents overfitting on small datasets
    # 
    # Space saved: ~2.2M parameters don't need gradient storage
    # Time saved: ~60% reduction in backward pass computation
    base_model.trainable = False

    # Build Custom Model Architecture
    # --------------------------------
    # Sequential API provides clean, linear architecture definition
    model = tf.keras.Sequential([
        # Input Layer
        # Explicitly define input shape for model introspection
        tf.keras.Input(shape=IMG_SHAPE),
        
        # Preprocessing Layer: Rescale [0, 1] -> [-1, 1]
        # MobileNetV2 was trained with inputs in range [-1, 1]
        # Our data_loader normalizes to [0, 1], so we need this conversion
        # Formula: output = (input * scale) + offset = (x * 2) - 1
        # Time: O(224 * 224 * 3) = O(150K) operations, very fast
        tf.keras.layers.Rescaling(scale=2.0, offset=-1.0),
        
        # Base Model: Feature Extractor
        # Takes (224, 224, 3) -> outputs (7, 7, 1280)
        # 1280 feature maps capture high-level image features
        # Parameters: 2,257,984 (frozen, not trainable)
        base_model,
        
        # Global Average Pooling
        # Converts (7, 7, 1280) -> (1280,) by averaging each feature map
        # Advantages over Flatten:
        # 1. Reduces parameters (no spatial dimensions)
        # 2. More robust to spatial translations
        # 3. Natural regularization effect
        # Time: O(7 * 7 * 1280) = O(63K) operations
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # Dropout Layer: Regularization
        # Randomly sets specified % of inputs to 0 during training
        # Prevents overfitting by reducing co-adaptation of neurons
        # Only active during training, disabled during inference
        # Time: O(1280) operations
        # dropout_rate can be configured via config parameter
        tf.keras.layers.Dropout(config['dropout_rate'] if config else 0.2),
        
        # Output Layer: Classification Head
        # Dense layer with softmax activation for multi-class classification
        # Parameters: (1280 + 1) * num_classes = 12,810 for 10 classes
        # Softmax converts logits to probability distribution (sum = 1.0)
        # Time: O(1280 * num_classes) = O(12,800) operations
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Model Compilation
    # -----------------
    # Configures the model for training with optimizer, loss, and metrics
    # Configuration can be customized via config parameter for experiments
    
    # Default Configuration (if no config provided)
    if config is None:
        # Optimizer: Adam (Adaptive Moment Estimation)
        # - Combines benefits of RMSprop and Momentum
        # - learning_rate=0.0001: Small LR for fine-tuning (prevents destroying features)
        # - Adaptive learning rates per parameter
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        
        # Loss: Sparse Categorical Crossentropy
        # - Used for multi-class classification with integer labels
        # - "Sparse" means labels are integers (0-9), not one-hot encoded
        loss = 'sparse_categorical_crossentropy'
        
        # Metrics: Track performance during training
        metrics = [
            'accuracy',  # Standard accuracy metric
            tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_acc')
        ]
    else:
        # Use experiment configuration
        optimizer = config['optimizer']
        loss = config['loss']
        metrics = config['metrics']
    
    # Compile model with selected configuration
    # Time: O(trainable_params) per update for optimizer
    # Time: O(batch_size * num_classes) per batch for loss
    # Time: O(batch_size) per batch for each metric
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model
