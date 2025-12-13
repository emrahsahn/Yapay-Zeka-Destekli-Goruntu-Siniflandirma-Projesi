"""
Experiment Runner Script

This script helps you run multiple experiments with different configurations
and compare their results.

Usage:
    # List all available configurations
    python src/run_experiments.py --list
    
    # Train with a specific configuration
    python src/run_experiments.py --config baseline
    python src/run_experiments.py --config high_lr
    python src/run_experiments.py --config sgd
    
    # Train multiple configs sequentially (one by one)
    python src/run_experiments.py --sequential baseline high_lr sgd
    
    # Compare results from multiple experiments
    python src/run_experiments.py --compare baseline high_lr sgd
"""

import sys
import os
import argparse
import pandas as pd
import time
from pathlib import Path

# Add parent directory to path if running from src/
if os.path.basename(os.getcwd()) == 'src':
    sys.path.insert(0, os.path.dirname(os.getcwd()))

try:
    from src.experiment_configs import list_configs, ALL_CONFIGS
    from src.train_model import train
except ImportError:
    from experiment_configs import list_configs, ALL_CONFIGS
    from train_model import train


def compare_results(config_names):
    """
    Compare results from multiple experiment configurations.
    
    Args:
        config_names (list): List of configuration names to compare
    """
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON")
    print("="*80)
    
    results = []
    
    for config_name in config_names:
        csv_file = f"model_artifacts/metrics_table_{config_name}.csv"
        
        if not Path(csv_file).exists():
            print(f"⚠ Warning: Results not found for '{config_name}'")
            print(f"  Please train this configuration first:")
            print(f"  python src/train_model.py {config_name}\n")
            continue
        
        # Load metrics
        df = pd.read_csv(csv_file, index_col=0)
        
        # Extract key metrics
        accuracy = df.loc['accuracy', 'f1-score'] if 'accuracy' in df.index else 0
        weighted_avg = df.loc['weighted avg']
        
        results.append({
            'Config': config_name,
            'Accuracy': f"{accuracy:.4f}",
            'Precision': f"{weighted_avg['precision']:.4f}",
            'Recall': f"{weighted_avg['recall']:.4f}",
            'F1-Score': f"{weighted_avg['f1-score']:.4f}",
            'Support': int(weighted_avg['support'])
        })
    
    if not results:
        print("No results to compare. Train some models first!")
        return
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    
    print("\nResults Summary:")
    print("-" * 80)
    print(comparison_df.to_string(index=False))
    print("-" * 80)
    
    # Find best performing configuration
    comparison_df['Accuracy_float'] = comparison_df['Accuracy'].astype(float)
    best_idx = comparison_df['Accuracy_float'].idxmax()
    best_config = comparison_df.loc[best_idx, 'Config']
    best_acc = comparison_df.loc[best_idx, 'Accuracy']
    
    print(f"\n🏆 Best Configuration: {best_config.upper()}")
    print(f"   Accuracy: {best_acc}")
    
    # Save comparison
    comparison_df.drop('Accuracy_float', axis=1, inplace=True)
    comparison_file = 'model_artifacts/experiments_comparison.csv'
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\n✓ Comparison saved to '{comparison_file}'")
    print("="*80)


def train_sequential(config_names):
    """
    Train multiple configurations sequentially (one at a time).
    This avoids overloading the system.
    
    Args:
        config_names (list): List of configuration names to train
    """
    print("=" * 70)
    print("SEQUENTIAL TRAINING - Sıralı Eğitim")
    print("=" * 70)
    print(f"\nEğitilecek konfigürasyonlar: {', '.join(config_names)}")
    print(f"Toplam: {len(config_names)} model")
    print("\nHer model eğitimi bitince sonraki başlayacak.")
    print("Bilgisayarınızı kullanmaya devam edebilirsiniz.\n")
    
    results = []
    
    for i, config_name in enumerate(config_names, 1):
        if config_name not in ALL_CONFIGS:
            print(f"\n⚠ {config_name} bulunamadı, atlanıyor...")
            results.append((config_name, "SKIPPED", 0))
            continue
            
        print(f"\n{'='*70}")
        print(f"[{i}/{len(config_names)}] {config_name.upper()} eğitimi başlıyor...")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        try:
            train(config_name)
            
            elapsed = time.time() - start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            
            print(f"\n✓ {config_name} tamamlandı! Süre: {minutes}dk {seconds}sn")
            results.append((config_name, "SUCCESS", elapsed))
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n✗ {config_name} hata verdi: {str(e)}")
            results.append((config_name, "FAILED", elapsed))
        
        # Kısa bir ara ver (sistem rahatlar)
        if i < len(config_names):
            print("\n5 saniye ara veriliyor...")
            time.sleep(5)
    
    # Final summary
    print("\n" + "=" * 70)
    print("TÜM EĞİTİMLER TAMAMLANDI!")
    print("=" * 70)
    print("\nÖzet:")
    for config_name, status, elapsed in results:
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        icon = "✓" if status == "SUCCESS" else ("⚠" if status == "SKIPPED" else "✗")
        print(f"{icon} {config_name:15s} - {status:10s} ({minutes}dk {seconds}sn)")
    
    # Karşılaştırma önerisi
    successful = [name for name, status, _ in results if status == "SUCCESS"]
    if len(successful) > 1:
        print(f"\n📊 Karşılaştırma için komut:")
        print(f"python src/run_experiments.py --compare {' '.join(successful)}")


def main():
    parser = argparse.ArgumentParser(
        description='Run and compare training experiments with different configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available configurations
  python src/run_experiments.py --list
  
  # Train with specific configuration
  python src/run_experiments.py --config baseline
  python src/run_experiments.py --config high_lr
  
  # Compare multiple experiments
  python src/run_experiments.py --compare baseline high_lr sgd
        """
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available configurations'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Configuration name to use for training'
    )
    
    parser.add_argument(
        '--sequential',
        nargs='+',
        help='Train multiple configurations sequentially (one by one)'
    )
    
    parser.add_argument(
        '--compare',
        nargs='+',
        help='Compare results from multiple configurations'
    )
    
    args = parser.parse_args()
    
    # List configurations
    if args.list:
        list_configs()
        return
    
    # Train sequentially
    if args.sequential:
        train_sequential(args.sequential)
        return
    
    # Compare results
    if args.compare:
        compare_results(args.compare)
        return
    
    # Train with configuration
    if args.config:
        if args.config not in ALL_CONFIGS:
            print(f"❌ Error: Configuration '{args.config}' not found!")
            print("\nAvailable configurations:")
            list_configs()
            sys.exit(1)
        
        print(f"\n🚀 Starting training with configuration: {args.config.upper()}")
        train(args.config)
        return
    
    # No arguments provided
    parser.print_help()
    print("\n" + "="*80)
    print("Quick Start:")
    print("  1. List configurations:  python src/run_experiments.py --list")
    print("  2. Train a model:        python src/run_experiments.py --config baseline")
    print("  3. Train multiple:       python src/run_experiments.py --sequential baseline high_lr sgd")
    print("  4. Compare results:      python src/run_experiments.py --compare baseline high_lr")
    print("="*80)


if __name__ == '__main__':
    main()
