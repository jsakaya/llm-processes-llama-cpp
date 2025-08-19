# LLM Processes with llama.cpp Backend

An enhanced version of [LLM Processes](https://github.com/requeima/llm_processes) with added support for [llama.cpp](https://github.com/ggerganov/llama.cpp) inference backend. This enables efficient local inference using quantized GGUF models with CPU and GPU acceleration.

## Features

### New llama.cpp Backend
- 🚀 **Fast local inference** with quantized GGUF models
- 💾 **Memory efficient** - 4-bit quantization reduces memory usage by ~75%
- 🖥️ **Hardware acceleration** - Metal (Apple Silicon), CUDA, ROCm support
- 🔄 **Seamless switching** between HuggingFace and llama.cpp backends
- 📊 **Full feature parity** with original implementation

### Original Features
- In-context learning for numerical predictions
- Support for various function types (linear, sine, exponential, etc.)
- Autoregressive time series forecasting
- Confidence interval estimation
- Comprehensive evaluation metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/jsakaya/llm-processes-llama-cpp.git
cd llm-processes-llama-cpp

# Install dependencies with poetry
poetry install

# Or with pip
pip install -r requirements.txt

# Install llama-cpp-python
pip install llama-cpp-python
```

## Quick Start

### 1. Download a GGUF Model

```bash
# Example: Download TinyLlama (638MB)
mkdir models
wget -O models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

### 2. Run an Experiment

```bash
# Using llama.cpp backend
llm_process --backend llama_cpp \
            --llm_path ./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
            --experiment_name "sine_test" \
            --data_path ./data/functions/sine_20_seed_0.pkl \
            --mode "sample_only" \
            --num_samples 10
```

### 3. Time Series Forecasting (Autoregressive)

```bash
# Airline passenger forecasting with confidence intervals
llm_process --backend llama_cpp \
            --llm_path ./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
            --experiment_name "airline_forecast" \
            --data_path ./data/airline_passengers_fixed.pkl \
            --autoregressive True \
            --forecast True \
            --y_min 0 --y_max 100 \
            --mode "sample_only" \
            --num_samples 20
```

## Backend Comparison

| Feature | HuggingFace | llama.cpp |
|---------|-------------|-----------|
| Model Format | Safetensors/PyTorch | GGUF |
| Quantization | Limited | 2-8 bit |
| Memory Usage | High | Low |
| Speed | Moderate | Fast |
| GPU Support | CUDA | Metal/CUDA/ROCm |
| Model Size (7B) | ~13GB | ~4GB (Q4) |

## Performance Results

Using TinyLlama-1.1B on Apple M1:

| Function Type | MAE (llama.cpp) | Samples/sec |
|---------------|-----------------|-------------|
| Linear | 0.0805 | ~4.0 |
| Sigmoid | 0.0640 | ~4.0 |
| Logarithm | 0.0768 | ~4.0 |
| Sine | 0.6221 | ~2.1 |
| Airline (Autoregressive) | 19.87 | ~8.6 |

## Key Improvements

### 1. Autoregressive Time Series Forecasting
- Enable with `--autoregressive True --forecast True`
- 33% improvement in MAE for time series tasks
- Uses previous predictions as context for future predictions

### 2. Data Scaling
- Use `--y_min` and `--y_max` to scale outputs
- Improves numerical stability for LLM predictions

### 3. Confidence Intervals
- Generate multiple samples with `--num_samples`
- Automatic confidence interval calculation in plots

## API Usage

```python
from llm_processes.parse_args import parse_command_line
from llm_processes.run_llm_process import run_llm_process

# Parse arguments
args = parse_command_line([
    '--backend', 'llama_cpp',
    '--llm_path', './models/model.gguf',
    '--data_path', './data/functions/sine_20_seed_0.pkl',
    '--mode', 'sample_only'
])

# Run experiment
run_llm_process(args)
```

## Converting Models to GGUF

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git

# Convert HuggingFace model to GGUF
python llama.cpp/convert.py path/to/hf/model \
  --outfile model.gguf \
  --outtype q4_k_m  # 4-bit quantization
```

## Supported Models

Any model supported by llama.cpp:
- Llama 2/3
- Mistral/Mixtral
- Phi-2/3
- TinyLlama
- Qwen
- And many more...

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{requeima2024llm,
  title={LLM Processes: Numerical Predictive Distributions Conditioned on Natural Language},
  author={Requeima, James and Bronskill, John and Choi, Dami and Turner, Richard E and Hernández-Lobato, José Miguel},
  journal={arXiv preprint arXiv:2405.12856},
  year={2024}
}
```

## License

MIT License (same as original repository)

## Acknowledgments

- Original [LLM Processes](https://github.com/requeima/llm_processes) by Requeima et al.
- [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) by Andrei Betlen