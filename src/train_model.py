"""
Model Training Script

This module handles the complete training pipeline for the animal classifier:
- Data loading and preprocessing
- Model creation and compilation
- Training with callbacks (checkpointing, early stopping)
- Performance evaluation and metrics calculation
- Results visualization and export

Usage:
    python src/train_model.py

Outputs:
    - model_artifacts/animal_classifier.keras: Trained model
    - model_artifacts/metrics_table.csv: Performance metrics table
    - model_artifacts/performance_summary.md: Detailed performance report

Performance Notes:
- Training time: ~5-10 minutes with GPU, ~30-60 minutes with CPU
- Memory usage: ~2-4GB RAM for data caching
- GPU recommended for faster training
"""

import tensorflow as tf
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Add parent directory to path if running from src/
if os.path.basename(os.getcwd()) == 'src':
    sys.path.insert(0, os.path.dirname(os.getcwd()))

try:
    from src.data_loader import create_datasets
    from src.model import create_model
    from src.experiment_configs import get_config, list_configs
except ImportError:
    # If running from src/ directory
    from data_loader import create_datasets
    from model import create_model
    from experiment_configs import get_config, list_configs


def train(config_name='baseline'):
    """
    Main training function that orchestrates the entire training pipeline.
    
    This function:
    1. Loads and preprocesses the dataset
    2. Creates the model architecture
    3. Trains the model with optimization callbacks
    4. Evaluates performance on validation set
    5. Generates comprehensive metrics report
    6. Saves trained model and results
    
    The training uses:
    - Early stopping to prevent overfitting
    - Model checkpointing to save best weights
    - Comprehensive metrics (accuracy, precision, recall, F1-score)
    - Confusion matrix for detailed error analysis
    
    Time Complexity:
        O(epochs * batches * (forward_pass + backward_pass))
        - Forward pass: O(model_params * batch_size)
        - Backward pass: O(model_params * batch_size)
        - With early stopping, typically converges in 3-7 epochs
    
    Space Complexity:
        O(model_params + dataset_cache)
        - Model: ~3.5M parameters for MobileNetV2
        - Dataset cache: ~3-4GB for Animals-10
    
    Raises:
        FileNotFoundError: If data directory doesn't exist
        ValueError: If dataset is empty or invalid
    
    Example:
        >>> train()
        Loading Data...
        Detected 10 classes: ['cane', 'cavallo', ...]
        Building Model...
        Model: "sequential"
        ...
        Training Complete. Model saved to 'model_artifacts/animal_classifier.keras'
    """
    # Load experiment configuration
    print("="*60)
    print(f"Loading Experiment Configuration: {config_name.upper()}")
    print("="*60)
    config = get_config(config_name)
    print(f"Description: {config['description']}")
    print(f"Optimizer: {config['optimizer'].__class__.__name__}")
    print(f"Epochs: {config['epochs']}")
    print(f"Dropout Rate: {config['dropout_rate']}")
    print(f"Metrics: {[m if isinstance(m, str) else m.name for m in config['metrics']]}")
    print("="*60)
    
    print("\nLoading Data...")
    # Time: O(n) for first epoch with caching
    train_ds, val_ds, class_names = create_datasets()
    
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    print("\nBuilding Model with experiment configuration...")
    model = create_model(num_classes, config=config)
    model.summary()

    # Create artifacts directory if not exists
    os.makedirs('model_artifacts', exist_ok=True)

    # Training Callbacks for Optimization
    # Callbacks are executed at specific points during training to improve performance
    
    # 1. ModelCheckpoint: Saves model only when validation accuracy improves
    #    This ensures we keep the best model even if training continues past optimal point
    #    Time: O(1) per epoch (only saves when improvement detected)
    # Save model with config name for comparison
    model_filename = f"model_artifacts/animal_classifier_{config_name}.keras"
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_filename,
        save_best_only=True,  # Only save when val_accuracy improves
        monitor='val_accuracy',  # Metric to monitor
        mode='max',  # We want to maximize accuracy
        verbose=1  # Print message when saving
    )
    
    # 2. EarlyStopping: Stops training if validation accuracy doesn't improve
    #    Prevents overfitting and saves training time
    #    patience=3: Waits 3 epochs before stopping (allows for temporary plateaus)
    #    Time saved: Can reduce training from 10-15 epochs to 5-7 epochs
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=3,  # Stop after 3 epochs without improvement
        restore_best_weights=True,  # Restore weights from best epoch
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )

    print("\nStarting Training...")
    print(f"Epochs: {config['epochs']} (may stop early if convergence detected)")
    print("="*60)
    
    # Training Configuration
    # Use epochs from config
    # Typical convergence: 5-7 epochs for transfer learning with MobileNetV2
    epochs = config['epochs']
    
    # Main Training Loop
    # Time: O(epochs * batches * model_complexity)
    # - batches ≈ 650 for Animals-10 (20,800 images / 32 batch_size)
    # - model_complexity: Forward + Backward pass through ~3.5M parameters
    # Estimated time: 5-10 minutes with GPU, 30-60 minutes with CPU
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint_cb, early_stopping_cb],
        verbose=1  # Show progress bar
    )

    # Performance Evaluation and Metrics Calculation
    print("\n" + "="*60)
    print("Evaluating Model on Validation Set...")
    print("="*60)
    
    # Collect predictions for all validation samples
    # Time: O(val_samples) = O(5,200) for Animals-10 validation set
    # This is much faster than training as we only do forward pass (no backprop)
    y_true = []  # Ground truth labels
    y_pred = []  # Model predictions

    # Iterate through validation dataset
    # Time: O(num_batches * forward_pass) ≈ O(163 batches * 50ms) ≈ 8 seconds
    for images, labels in val_ds:
        # Forward pass through model
        # Shape: (batch_size, 10) with probabilities for each class
        preds = model.predict(images, verbose=0)
        
        # Store true labels (shape: batch_size,)
        y_true.extend(labels.numpy())
        
        # Get predicted class (argmax over 10 classes)
        # Time: O(batch_size * num_classes) = O(32 * 10) = O(320) per batch
        y_pred.extend(np.argmax(preds, axis=1))

    # Generate Comprehensive Performance Report
    print("\n--- Generating Comprehensive Performance Report ---")
    
    # Calculate detailed metrics using scikit-learn
    # classification_report computes:
    # - Precision: TP / (TP + FP) - How many predicted positives are correct
    # - Recall: TP / (TP + FN) - How many actual positives are found
    # - F1-Score: Harmonic mean of Precision and Recall
    # - Support: Number of samples per class
    # Time: O(n * c) where n=5200 samples, c=10 classes
    import pandas as pd
    report_dict = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names, 
        output_dict=True,
        zero_division=0  # Handle edge cases
    )
    
    # Convert to DataFrame for better visualization
    # This allows easy export to CSV and Markdown
    report_df = pd.DataFrame(report_dict).transpose()
    
    print("\nPerformance Metrics Table:")
    print(report_df.to_string())  # Print full table
    
    # Save metrics to files for documentation
    # CSV format: Easy to import into Excel, Google Sheets, etc.
    csv_filename = f'model_artifacts/metrics_table_{config_name}.csv'
    report_df.to_csv(csv_filename)
    print(f"\n✓ Metrics saved to '{csv_filename}'")
    
    # Markdown format: Easy to include in README or reports
    # Also includes confusion matrix for error analysis
    md_filename = f'model_artifacts/performance_summary_{config_name}.md'
    with open(md_filename, 'w', encoding='utf-8') as f:
        f.write("# Model Performance Metrics\n\n")
        f.write("## Classification Report\n\n")
        f.write(report_df.to_markdown())
        f.write("\n\n## Confusion Matrix\n\n")
        f.write("```\n")
        f.write(str(confusion_matrix(y_true, y_pred)))
        f.write("\n```\n")
        f.write("\n## Class Names\n\n")
        for idx, name in enumerate(class_names):
            f.write(f"{idx}: {name}\n")
    
    print(f"✓ Performance summary saved to '{md_filename}'")

    print("\n" + "="*60)
    print(f"Training Complete! [Config: {config_name.upper()}]")
    print("="*60)
    print(f"Model saved to: '{model_filename}'")
    print(f"Configuration: {config['description']}")
    print(f"Total classes: {num_classes}")
    print(f"Training epochs: {len(history.history['loss'])}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print("\nTo run the application:")
    print("  python src/app.py")
    print("="*60)


if __name__ == '__main__':
    import sys
    
    # Check if config name provided as argument
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
        print(f"\nUsing configuration: {config_name}")
    else:
        print("\nNo configuration specified, using 'baseline'")
        print("Available configurations:")
        list_configs()
        print("\nUsage: python src/train_model.py [config_name]")
        print("Example: python src/train_model.py high_lr")
        print("\nContinuing with 'baseline' configuration...\n")
        config_name = 'baseline'
    
    train(config_name)
