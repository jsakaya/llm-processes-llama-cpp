#!/usr/bin/env python3
"""
Create a synthetic airline passenger dataset similar to the classic airline passenger dataset.
This dataset will have seasonal patterns and an increasing trend.
"""

import numpy as np
import pickle
import os

def create_airline_passenger_data(n_train=50, n_test=20, noise_level=0.1):
    """Create synthetic airline passenger data with seasonal patterns."""
    
    # Total time series length
    total_length = n_train + n_test
    
    # Create time points (representing months)
    x_all = np.arange(total_length).astype(float)
    
    # Base trend (increasing over time)
    base_trend = 100 + 2 * x_all
    
    # Seasonal component (annual cycle with 12-month period)
    seasonal = 20 * np.sin(2 * np.pi * x_all / 12) + 10 * np.sin(4 * np.pi * x_all / 12)
    
    # Long-term exponential growth
    growth = np.exp(0.02 * x_all)
    
    # Combine components
    y_true = (base_trend + seasonal) * growth
    
    # Add noise
    np.random.seed(42)
    y_all = y_true + noise_level * np.std(y_true) * np.random.randn(total_length)
    
    # Split into train and test
    x_train = x_all[:n_train].reshape(-1, 1)
    y_train = y_all[:n_train].reshape(-1, 1)
    x_test = x_all[n_train:].reshape(-1, 1)
    y_test = y_all[n_train:].reshape(-1, 1)
    
    # Also include the true function for plotting
    x_true = x_all
    y_true = y_true
    
    return {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test,
        'x_true': x_true,
        'y_true': y_true
    }

if __name__ == "__main__":
    # Create the dataset
    data = create_airline_passenger_data(n_train=50, n_test=20, noise_level=0.1)
    
    # Ensure data directory exists
    os.makedirs('./data', exist_ok=True)
    
    # Save the dataset
    with open('./data/airline_passengers.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    print("Created airline passenger dataset with:")
    print(f"Training points: {data['x_train'].shape[0]}")
    print(f"Test points: {data['x_test'].shape[0]}")
    print(f"X range: {data['x_train'][0,0]:.1f} to {data['x_test'][-1,0]:.1f}")
    print(f"Y range: {np.min(data['y_train']):.1f} to {np.max(data['y_train']):.1f}")
    print("Saved to: ./data/airline_passengers.pkl")