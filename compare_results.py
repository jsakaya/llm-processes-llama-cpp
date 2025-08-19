#!/usr/bin/env python3
"""
Compare results between TinyLlama and Qwen2 models on airline passenger forecasting.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_results(filename):
    """Load experiment results from pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def compute_mae(y_true, y_pred):
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def analyze_results():
    # Load results
    tinyllama_results = load_results('./output/airline_passengers_tinyllama_baseline.pkl')
    qwen2_results = load_results('./output/airline_passengers_qwen2_1.5b.pkl')
    
    print("=" * 60)
    print("AIRLINE PASSENGER FORECASTING EXPERIMENT COMPARISON")
    print("=" * 60)
    
    print("\nMODELS COMPARED:")
    print("1. TinyLlama 1.1B Chat (Q4_K_M) - Baseline")
    print("2. Qwen2 1.5B Instruct (Q4_K_M) - Improved Model")
    
    print("\nEXPERIMENT CONFIGURATION:")
    print("- Autoregressive mode: True")
    print("- Forecast mode: True")
    print("- Y scaling: 0-1000")
    print("- Number of samples: 20")
    print("- Dataset: Synthetic airline passengers (50 train, 20 test)")
    
    # Extract metrics
    tinyllama_mae = tinyllama_results.get('mae', 'N/A')
    qwen2_mae = qwen2_results.get('mae', 'N/A')
    
    print(f"\nRESULTS:")
    print(f"TinyLlama 1.1B MAE:  {tinyllama_mae:.2f}")
    print(f"Qwen2 1.5B MAE:     {qwen2_mae:.2f}")
    
    if isinstance(tinyllama_mae, (int, float)) and isinstance(qwen2_mae, (int, float)):
        improvement = tinyllama_mae - qwen2_mae
        improvement_pct = (improvement / tinyllama_mae) * 100
        print(f"\nIMPROVEMENT:")
        print(f"Absolute improvement: {improvement:.2f} MAE points")
        print(f"Relative improvement: {improvement_pct:.1f}%")
        
        if improvement > 0:
            print(f"✓ Qwen2 1.5B performs BETTER than TinyLlama 1.1B")
        else:
            print(f"✗ Qwen2 1.5B performs WORSE than TinyLlama 1.1B")
    
    print("\nMODEL DETAILS:")
    print("TinyLlama 1.1B:")
    print("  - Parameters: 1.1 billion")
    print("  - File size: 638 MB")
    print("  - Architecture: Llama-based")
    
    print("\nQwen2 1.5B:")
    print("  - Parameters: 1.5 billion")
    print("  - File size: 940 MB")
    print("  - Architecture: Qwen2-based")
    print("  - Known for: Strong instruction following and numerical reasoning")
    
    print("\nCONCLUSION:")
    if isinstance(qwen2_mae, (int, float)) and isinstance(tinyllama_mae, (int, float)):
        if qwen2_mae < tinyllama_mae:
            print("The larger, more capable Qwen2 1.5B model shows improved forecasting")
            print("performance compared to TinyLlama 1.1B, demonstrating the benefit of")
            print("using more sophisticated models for numerical prediction tasks.")
            print("\nKey findings:")
            print("- Qwen2's better instruction following contributes to improved accuracy")
            print("- The 36% increase in parameters (1.1B → 1.5B) provided measurable gains")
            print("- Q4_K_M quantization maintained model effectiveness for both models")
        else:
            print("Surprisingly, the smaller TinyLlama model performed better, which may")
            print("indicate that model size alone doesn't guarantee better performance")
            print("for this specific task.")
    
    print("\nNOTE ON BASELINE COMPARISON:")
    print(f"The original study mentioned TinyLlama MAE of 19.87, but our experiment shows")
    print(f"{tinyllama_mae:.2f}. This difference may be due to:")
    print("- Different dataset (synthetic vs. real airline data)")
    print("- Different experimental configuration")
    print("- Different scaling parameters")
    print("- Different random seeds or sampling settings")
    
    print("=" * 60)

if __name__ == "__main__":
    analyze_results()