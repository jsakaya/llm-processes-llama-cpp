#!/usr/bin/env python3
"""
Create enhanced plots for the airline passenger forecasting experiment.
Shows individual trajectories, confidence intervals, training/test data, and ground truth.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def create_enhanced_plot(results_path, output_name, plot_dir="./plots"):
    """Create enhanced plot with trajectories."""
    
    # Load results
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    # Extract data
    data = results['data']
    x_train = data['x_train']
    y_train = data['y_train'] 
    x_test = data['x_test']
    y_test_true = data['y_test']
    x_true = data['x_true']
    y_true = data['y_true']
    
    # Extract predictions
    y_test_samples = np.array(results['y_test'])  # Shape: (n_test, n_samples)
    y_test_mean = np.array(results['y_test_mean'])
    y_test_std = np.array(results['y_test_std'])
    y_test_lower = np.array(results['y_test_lower'])
    y_test_upper = np.array(results['y_test_upper'])
    
    mae = results['mae']
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot ground truth function
    ax.plot(x_true, y_true, 'k-', linewidth=2, label='Ground Truth', alpha=0.8)
    
    # Plot training data
    ax.scatter(x_train, y_train, c='blue', marker='o', s=50, 
               label=f'Training Data (n={len(x_train)})', zorder=5)
    
    # Plot test data (ground truth)
    ax.scatter(x_test, y_test_true, c='red', marker='s', s=50,
               label=f'Test Data (Ground Truth)', zorder=5)
    
    # Plot individual trajectory samples (5 trajectories as requested)
    num_trajectories = min(5, y_test_samples.shape[1])
    colors_traj = plt.cm.tab10(np.linspace(0, 1, num_trajectories))
    
    for i in range(num_trajectories):
        ax.plot(x_test, y_test_samples[:, i], '--', alpha=0.6, 
                color=colors_traj[i], linewidth=1.5,
                label=f'Sample {i+1}' if i < 3 else None)  # Only label first 3 for legend clarity
    
    # Plot mean prediction
    ax.plot(x_test, y_test_mean, 'orange', linewidth=3, 
            label='Mean Prediction', zorder=4)
    
    # Plot confidence interval
    ax.fill_between(x_test, y_test_lower, y_test_upper, 
                    alpha=0.3, color='orange', 
                    label='95% Confidence Interval')
    
    # Add vertical line separating train/test
    train_test_boundary = x_train[-1] + (x_test[0] - x_train[-1]) / 2
    ax.axvline(x=train_test_boundary, color='gray', linestyle=':', alpha=0.7,
               label='Train/Test Split')
    
    # Formatting
    ax.set_xlabel('Time (months)', fontsize=12)
    ax.set_ylabel('Scaled Passenger Numbers (0-10)', fontsize=12)
    ax.set_title(f'Airline Passenger Forecasting with TinyLlama\n' + 
                f'52 Training Points, 0-10 Scaling, MAE = {mae:.4f}\n' + 
                f'Individual Trajectories (5 samples shown)', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save plots
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save PDF
    pdf_path = os.path.join(plot_dir, f"{output_name}_enhanced.pdf")
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    print(f"Enhanced plot saved to: {pdf_path}")
    
    # Save PNG
    png_path = os.path.join(plot_dir, f"{output_name}_enhanced.png")
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    print(f"Enhanced plot saved to: {png_path}")
    
    # Show some statistics
    print(f"\nExperiment Results:")
    print(f"MAE: {mae:.4f}")
    print(f"Training points: {len(x_train)}")
    print(f"Test points: {len(x_test)}")
    print(f"Y scaling range: 0-10")
    print(f"Mean prediction range: {np.min(y_test_mean):.4f} to {np.max(y_test_mean):.4f}")
    print(f"Ground truth test range: {np.min(y_test_true):.4f} to {np.max(y_test_true):.4f}")
    print(f"Individual trajectories shown: {num_trajectories}")
    
    plt.tight_layout()
    plt.close()  # Close the figure to prevent display

if __name__ == "__main__":
    # Create enhanced plot for the airline experiment
    create_enhanced_plot('./output/airline_52train_scale10.pkl', 'airline_52train_scale10')