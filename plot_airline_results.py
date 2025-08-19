#!/usr/bin/env python3
"""
Plot airline passenger forecasting results with confidence intervals.
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def plot_airline_forecasting_results(results_path, output_path):
    """Plot airline passenger forecasting results with confidence intervals."""
    
    # Load results
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    # Extract data
    x_train = results['data']['x_train'].flatten()
    y_train = results['data']['y_train'].flatten()
    x_test = results['data']['x_test'].flatten()
    y_test_true = results['data']['y_test'].flatten()
    
    # Extract predictions
    y_test_mean = np.array(results['y_test_mean'])
    y_test_median = np.array(results['y_test_median'])
    y_test_std = np.array(results['y_test_std'])
    y_test_lower = np.array(results['y_test_lower'])
    y_test_upper = np.array(results['y_test_upper'])
    
    # True function if available
    x_true = results['data']['x_true']
    y_true = results['data']['y_true']
    
    # Create the plot
    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot true function
    axes.plot(x_true, y_true, label='True Function', c='black', linewidth=2)
    
    # Plot training points
    axes.scatter(x_train, y_train, label='Training Points', c='black', marker='v', s=50, zorder=5)
    
    # Plot ground truth test points
    axes.scatter(x_test, y_test_true, label='Ground Truth (Test)', c='red', marker='o', s=50, zorder=5)
    
    # Sort test points for proper line plotting
    sort_indices = np.argsort(x_test)
    x_test_sorted = x_test[sort_indices]
    y_median_sorted = y_test_median[sort_indices]
    y_lower_sorted = y_test_lower[sort_indices]
    y_upper_sorted = y_test_upper[sort_indices]
    
    # Plot median predictions
    axes.plot(x_test_sorted, y_median_sorted, alpha=1.0, c='tab:red', label='Predicted Median', 
              marker='.', linewidth=2, markersize=8)
    
    # Plot confidence intervals
    axes.fill_between(x_test_sorted, y_lower_sorted, y_upper_sorted, 
                     color='tab:orange', alpha=0.4, label='95% Confidence Interval')
    
    # Plot individual samples (trajectories) if available
    if 'y_test' in results:
        y_samples = np.array(results['y_test'])
        if len(y_samples.shape) > 1:
            # Plot a few sample trajectories
            for i in range(min(5, y_samples.shape[1])):
                y_sample_sorted = y_samples[sort_indices, i]
                axes.plot(x_test_sorted, y_sample_sorted, 
                         alpha=0.3, c='tab:blue', linewidth=1, 
                         label='Sample Trajectory' if i == 0 else '')
    
    # Add vertical line to separate training and test regions
    if len(x_train) > 0 and len(x_test) > 0:
        separation_point = max(x_train)
        axes.axvline(x=separation_point, color='gray', linestyle='--', alpha=0.7, 
                    label='Train/Test Split')
    
    # Formatting
    axes.set_xlabel('Time (Months)', fontsize=12)
    axes.set_ylabel('Passengers', fontsize=12)
    axes.set_title(f'Airline Passenger Forecasting with LLM\nMAE = {results["mae"]:.2f}', fontsize=14)
    axes.grid(True, alpha=0.3)
    axes.legend(fontsize=10)
    
    # Set reasonable y-limits
    all_y_values = np.concatenate([y_train, y_test_true, y_test_median, y_true])
    y_min, y_max = np.min(all_y_values), np.max(all_y_values)
    y_range = y_max - y_min
    axes.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    
    print(f"Enhanced plot saved to: {output_path}")
    print(f"Also saved as PNG: {output_path.replace('.pdf', '.png')}")
    
    # Print statistics
    print(f"\nForecasting Results:")
    print(f"MAE: {results['mae']:.2f}")
    print(f"MSE: {results['mse']:.2f}")
    print(f"Number of training points: {len(x_train)}")
    print(f"Number of test points: {len(x_test)}")
    y_test_array = np.array(results['y_test'])
    print(f"Number of samples per test point: {y_test_array.shape[1] if len(y_test_array.shape) > 1 else 1}")
    
    return fig, axes

if __name__ == "__main__":
    results_path = './output/airline_passengers_llama_cpp.pkl'
    output_path = './plots/airline_passengers_forecasting_enhanced.pdf'
    
    plot_airline_forecasting_results(results_path, output_path)
    plt.show()