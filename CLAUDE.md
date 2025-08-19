# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is LLM Processes, a framework for numerical predictive distributions conditioned on natural language. The codebase has been enhanced with llama.cpp backend support for efficient local inference using quantized GGUF models.

## Build and Development Commands

### Installation
```bash
# Install with poetry (recommended)
poetry install

# Install optional dependencies (for black-box optimization and Gaussian Processes)
poetry install -E opt

# Or install with pip
pip install .

# Install llama-cpp-python for llama.cpp backend
poetry run pip install llama-cpp-python
```

### Running Experiments
```bash
# Basic command structure
llm_process --backend [hf|llama_cpp] --llm_path <path> --data_path <path> [options]

# HuggingFace backend example
llm_process --backend hf --llm_type llama-2-7B --data_path ./data/functions/sine_20_seed_0.pkl

# llama.cpp backend example (requires GGUF model)
llm_process --backend llama_cpp --llm_path ./models/model.gguf --data_path ./data/functions/sine_20_seed_0.pkl

# Time series forecasting with autoregressive mode
llm_process --backend llama_cpp --llm_path ./models/model.gguf \
            --data_path ./data/airline_passengers_fixed.pkl \
            --autoregressive True --forecast True \
            --y_min 0 --y_max 100 --num_samples 20
```

### Testing
```bash
# Run test backends script
python test_backends.py

# Run specific experiment scripts
python experiments/run_functions_exp.py --llm_type phi-3-mini-128k-instruct --function sine
python experiments/run_housing_exp.py --llm_type llama-2-7B
```

## Architecture and Key Components

### Dual Backend System
The codebase supports two inference backends that can be switched via `--backend` argument:

1. **HuggingFace Backend** (`hf_api.py`):
   - Original implementation using transformers library
   - Supports models: llama-2, llama-3, mixtral, phi-3
   - Higher memory usage, requires full precision models

2. **llama.cpp Backend** (`llama_cpp_api.py`):
   - New implementation for quantized GGUF models
   - Wrapper classes: `LlamaCppModel`, `LlamaCppTokenizer`
   - Mimics HuggingFace API for compatibility
   - Supports 2-8 bit quantization, Metal/CUDA acceleration

### Core Workflow
1. **Entry Point** (`run_llm_process.py`):
   - Parses arguments and selects backend
   - Orchestrates data loading, sampling, and evaluation
   
2. **Data Preparation** (`prepare_data.py`):
   - Loads pickle files with x_train, y_train, x_test, y_test
   - Handles scaling via `--y_min` and `--y_max`
   
3. **Prompt Construction** (`helpers.py`):
   - Formats in-context examples: `x1, y1\nx2, y2\n...xn,`
   - Supports different orderings: sequential, random, distance
   - Critical fix: `repr(float(f))` for numpy float64 compatibility

4. **Sampling** (`sample.py`):
   - Two modes: I-LLMP (default) and A-LLMP (autoregressive)
   - Autoregressive mode uses predictions as context for next predictions
   - Backend-specific generation functions selected dynamically

5. **Evaluation** (`compute_nll.py`):
   - Computes negative log-likelihood for predictions
   - llama.cpp backend implements token-by-token evaluation

### Critical Implementation Details

#### Autoregressive Mode
When `--autoregressive True`:
- Previous predictions are appended to context
- Essential for time series with `--forecast True`
- 33% improvement in MAE for sequential data

#### Data Format Requirements
- 1D arrays for single-dimensional data
- 2D arrays (n×d) for multi-dimensional inputs
- Pickle files must contain: x_train, y_train, x_test, y_test

#### llama.cpp Specific Considerations
- Models must be in GGUF format
- Context size set as `batch_size * 1024` in `llama_cpp_api.py`
- Batch generation processes prompts sequentially (not parallel)
- `LlamaCppModel.__call__` implements NLL computation via token-by-token evaluation

## Model Conversion

```bash
# Convert HuggingFace model to GGUF
git clone https://github.com/ggerganov/llama.cpp.git
python llama.cpp/convert.py path/to/hf/model --outfile model.gguf --outtype q4_k_m
```

## Dataset Structure

Datasets are pickle files containing dictionaries with:
- `x_train`: training inputs (numpy array)
- `y_train`: training outputs (numpy array)
- `x_test`: test inputs (numpy array)
- `y_test`: test outputs (numpy array)
- Optional: `x_ordering` for categorical features

## Important Parameters

- `--autoregressive True`: Enable A-LLMP for time series
- `--forecast True`: Sequential ordering for time series
- `--y_min/--y_max`: Scale outputs to specified range
- `--num_samples`: Number of samples for confidence intervals
- `--batch_size`: GPU memory vs speed tradeoff
- `--temperature/--top_p`: Control sampling randomness
- `--prompt_ordering`: sequential/random/distance
- `--max_generated_length`: Maximum tokens to generate per prediction
- use poetry