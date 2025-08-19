#!/usr/bin/env python
"""
Test script to compare HuggingFace and llama.cpp backends
"""

import subprocess
import sys

def test_backend(backend, model_path=None):
    """Test a specific backend"""
    print(f"\n{'='*60}")
    print(f"Testing {backend} backend")
    print('='*60)
    
    cmd = [
        sys.executable, "-m", "llm_processes.run_llm_process",
        "--backend", backend,
        "--experiment_name", f"test_{backend}",
        "--data_path", "./data/functions/square_20_seed_0.pkl",
        "--mode", "sample_only",
        "--num_samples", "2",
        "--output_dir", "./output",
        "--plot_dir", "./plots",
        "--batch_size", "1",
        "--max_generated_length", "10"
    ]
    
    if backend == "llama_cpp":
        if model_path:
            cmd.extend(["--llm_path", model_path])
        else:
            cmd.extend(["--llm_path", "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"])
    else:  # hf backend
        # Use a small model for testing
        cmd.extend(["--llm_type", "llama-2-7B"])
        if model_path:
            cmd.extend(["--llm_path", model_path])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            # Extract MAE from output
            for line in result.stdout.split('\n'):
                if 'mae:' in line.lower():
                    print(f"✓ Success! {line}")
                    return True
            print("✓ Process completed successfully")
            return True
        else:
            print(f"✗ Failed with error:")
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("✗ Process timed out")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("LLM-Process Backend Comparison Test")
    
    # Test llama.cpp backend (should work)
    llama_success = test_backend("llama_cpp")
    
    # Test HF backend (will fail without a model, but shows it's available)
    # Uncomment and provide a valid HF model path to test
    # hf_success = test_backend("hf", "meta-llama/Llama-2-7b-hf")
    
    print(f"\n{'='*60}")
    print("Test Summary:")
    print(f"  llama.cpp backend: {'✓ PASSED' if llama_success else '✗ FAILED'}")
    # print(f"  HuggingFace backend: {'✓ PASSED' if hf_success else '✗ FAILED'}")
    print('='*60)