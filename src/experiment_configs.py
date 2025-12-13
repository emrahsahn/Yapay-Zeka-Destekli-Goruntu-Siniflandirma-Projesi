"""
Experiment Configuration Module

This module provides different training configurations to experiment with various:
- Optimizers (Adam, SGD, RMSprop, AdamW)
- Learning rates
- Loss functions
- Metrics combinations
- Model architectures

Each configuration represents a different experiment to find optimal hyperparameters.
"""

import tensorflow as tf


# Experiment 1: Baseline (Current Configuration)
# ===============================================
CONFIG_BASELINE = {
    'name': 'baseline',
    'description': 'Current configuration - Adam optimizer with lr=0.0001',
    'optimizer': tf.keras.optimizers.Adam(learning_rate=0.0001),
    'loss': 'sparse_categorical_crossentropy',
    'metrics': [
        'accuracy',
        tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')
    ],
    'epochs': 10,
    'dropout_rate': 0.2
}


# Experiment 2: Higher Learning Rate
# ===================================
CONFIG_HIGH_LR = {
    'name': 'high_lr',
    'description': 'Adam optimizer with higher learning rate (0.001)',
    'optimizer': tf.keras.optimizers.Adam(learning_rate=0.001),
    'loss': 'sparse_categorical_crossentropy',
    'metrics': [
        'accuracy',
        tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')
    ],
    'epochs': 10,
    'dropout_rate': 0.2
}


# Experiment 3: Lower Learning Rate (Fine-tuning)
# ================================================
CONFIG_LOW_LR = {
    'name': 'low_lr',
    'description': 'Adam optimizer with very low learning rate (0.00001) for fine-tuning',
    'optimizer': tf.keras.optimizers.Adam(learning_rate=0.00001),
    'loss': 'sparse_categorical_crossentropy',
    'metrics': [
        'accuracy',
        tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')
    ],
    'epochs': 15,
    'dropout_rate': 0.2
}


# Experiment 4: SGD with Momentum
# ================================
CONFIG_SGD = {
    'name': 'sgd_momentum',
    'description': 'SGD optimizer with momentum (0.9) and Nesterov acceleration',
    'optimizer': tf.keras.optimizers.SGD(
        learning_rate=0.01,
        momentum=0.9,
        nesterov=True
    ),
    'loss': 'sparse_categorical_crossentropy',
    'metrics': [
        'accuracy',
        tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')
    ],
    'epochs': 15,
    'dropout_rate': 0.3
}


# Experiment 5: RMSprop
# =====================
CONFIG_RMSPROP = {
    'name': 'rmsprop',
    'description': 'RMSprop optimizer - good for RNNs and adaptive learning',
    'optimizer': tf.keras.optimizers.RMSprop(learning_rate=0.001),
    'loss': 'sparse_categorical_crossentropy',
    'metrics': [
        'accuracy',
        tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')
    ],
    'epochs': 10,
    'dropout_rate': 0.2
}


# Experiment 6: AdamW (Adam with Weight Decay)
# =============================================
CONFIG_ADAMW = {
    'name': 'adamw',
    'description': 'AdamW optimizer with weight decay for better regularization',
    'optimizer': tf.keras.optimizers.AdamW(
        learning_rate=0.0001,
        weight_decay=0.0001
    ),
    'loss': 'sparse_categorical_crossentropy',
    'metrics': [
        'accuracy',
        tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')
    ],
    'epochs': 15,
    'dropout_rate': 0.2
}


# Experiment 4: SGD with Momentum
# ==========================
CONFIG_ALL_METRICS = {
    'name': 'all_metrics',
    'description': 'Baseline config but with comprehensive metrics tracking',
    'optimizer': tf.keras.optimizers.Adam(learning_rate=0.0001),
    'loss': 'sparse_categorical_crossentropy',
    'metrics': [
        'accuracy',
        tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')
    ],
    'epochs': 10,
    'dropout_rate': 0.2
}


# Experiment 8: High Dropout (Prevent Overfitting)
# =================================================
CONFIG_HIGH_DROPOUT = {
    'name': 'high_dropout',
    'description': 'Baseline config with higher dropout (0.5) to prevent overfitting',
    'optimizer': tf.keras.optimizers.Adam(learning_rate=0.0001),
    'loss': 'sparse_categorical_crossentropy',
    'metrics': [
        'accuracy',
        tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')
    ],
    'epochs': 12,
    'dropout_rate': 0.5
}


# Experiment 9: Learning Rate Schedule
# =====================================
CONFIG_LR_SCHEDULE = {
    'name': 'lr_schedule',
    'description': 'Adam with exponential learning rate decay',
    'optimizer': tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.96
        )
    ),
    'loss': 'sparse_categorical_crossentropy',
    'metrics': [
        'accuracy',
        tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')
    ],
    'epochs': 15,
    'dropout_rate': 0.2
}


# Experiment 10: Label Smoothing
# ===============================
CONFIG_LABEL_SMOOTHING = {
    'name': 'label_smoothing',
    'description': 'Use label smoothing to improve generalization (note: requires custom implementation)',
    'optimizer': tf.keras.optimizers.Adam(learning_rate=0.0001),
    'loss': 'sparse_categorical_crossentropy',  # Label smoothing would need custom loss
    'metrics': [
        'accuracy',
        tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')
    ],
    'epochs': 10,
    'dropout_rate': 0.3  # Use higher dropout instead
}


# All configurations dictionary for easy access
ALL_CONFIGS = {
    'baseline': CONFIG_BASELINE,
    'high_lr': CONFIG_HIGH_LR,
    'low_lr': CONFIG_LOW_LR,
    'sgd': CONFIG_SGD,
    'rmsprop': CONFIG_RMSPROP,
    'adamw': CONFIG_ADAMW,
    'all_metrics': CONFIG_ALL_METRICS,
    'high_dropout': CONFIG_HIGH_DROPOUT,
    'lr_schedule': CONFIG_LR_SCHEDULE,
    'label_smoothing': CONFIG_LABEL_SMOOTHING
}


def get_config(name):
    """
    Retrieves a configuration by name.
    
    Args:
        name (str): Configuration name
    
    Returns:
        dict: Configuration dictionary
    
    Raises:
        ValueError: If configuration name is not found
    """
    if name not in ALL_CONFIGS:
        raise ValueError(
            f"Configuration '{name}' not found. "
            f"Available configs: {list(ALL_CONFIGS.keys())}"
        )
    return ALL_CONFIGS[name]


def list_configs():
    """
    Prints all available configurations with descriptions.
    """
    print("Available Experiment Configurations:")
    print("=" * 70)
    for name, config in ALL_CONFIGS.items():
        print(f"\n{name.upper()}")
        print(f"  Description: {config['description']}")
        print(f"  Optimizer: {config['optimizer'].__class__.__name__}")
        print(f"  Epochs: {config['epochs']}")
        print(f"  Dropout: {config['dropout_rate']}")
        print(f"  Metrics: {len(config['metrics'])} metrics")
    print("\n" + "=" * 70)
