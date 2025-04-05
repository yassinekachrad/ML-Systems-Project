#!/usr/bin/env python
"""
Benchmark Inference Speed of Selected Mistral Models

This script loads a set of Mistral text-generation models, measures their
inference speed (tokens per second) by generating a fixed number of tokens from
a common prompt, and plots model size (in billions of parameters) vs.
throughput. The script defaults to the MPS device (Apple Silicon) but allows
overriding with CPU or CUDA.

Models Benchmarked:
  - mistralai/Mistral-7B-v0.3         (~7B params)
  - mistralai/Ministral-8B-Instruct-2410 (~8B params)
  - mistralai/Mistral-Nemo-Base-2407   (~12B params)
  - mistralai/Mistral-Small-24B-Base-2501 (~24B params)
  - mistralai/Mixtral-8x7B-v0.3        (~56B params; 8x7B)

Note: Ensure you have accepted any model licenses required by Hugging Face.
"""

import time
import gc
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

def get_device(device_name: str) -> torch.device:
    """
    Determine the computation device based on user input.
    Defaults to MPS if available, but allows 'cuda' or 'cpu'.
    """
    device_name = device_name.lower()
    if device_name == "mps":
        if not torch.backends.mps.is_available() or not torch.backends.mps.is_built():
            print("Warning: MPS device not available or not built. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("mps")
    elif device_name == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA device not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    elif device_name == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError(f"Unsupported device '{device_name}'. Choose 'mps', 'cuda', or 'cpu'.")

def load_model_and_tokenizer(model_name: str, device: torch.device):
    """
    Load a pretrained model and tokenizer from Hugging Face.
    Uses fp16 on GPU/MPS for speed/efficiency; defaults to fp32 on CPU.
    """
    if device.type in ("cuda", "mps"):
        dtype = torch.float16
    else:
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()  # disable dropout, etc.
    return model, tokenizer

def measure_inference_speed(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int = 50) -> float:
    """
    Generate text from the model and measure inference throughput (tokens per second).
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    # Warm-up generation to initialize caches/kernels
    with torch.no_grad():
        model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=1)
    
    # Clear cache if needed
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()

    start_time = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False  # deterministic generation for consistency
        )
    elapsed = time.perf_counter() - start_time

    # Calculate throughput: number of new tokens generated per second
    new_tokens = output_ids.size(1) - input_ids.size(1)
    tokens_per_sec = new_tokens / elapsed if elapsed > 0 else float("inf")
    return tokens_per_sec

def main():
    parser = argparse.ArgumentParser(description="Benchmark Inference Speed of Selected Mistral Models")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps",
                        help="Device to run the models on (default: mps).")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Number of tokens to generate for the speed test (default: 50).")
    args = parser.parse_args()
    device = get_device(args.device)
    max_new_tokens = args.max_new_tokens

    # Define Mistral models to benchmark with approximate parameter counts (in billions)
    models_info = [
        {"name": "mistralai/Mistral-7B-v0.3",           "label": "Mistral-7B-v0.3",            "size": 7.0},
        {"name": "mistralai/Ministral-8B-Instruct-2410",   "label": "Ministral-8B-Instruct-2410",   "size": 8.0},
        {"name": "mistralai/Mistral-Nemo-Base-2407",       "label": "Mistral-Nemo-Base-2407",       "size": 12.0},
        # {"name": "mistralai/Mistral-Small-24B-Base-2501",  "label": "Mistral-Small-24B-Base-2501",  "size": 24.0},
        # {"name": "mistralai/Mixtral-8x7B-v0.1",            "label": "Mixtral-8x7B-v0.1",            "size": 56.0},  # 8 x 7B
    ]

    # Sample prompt used for text generation (ensure a similar context for all models)
    prompt_text = (
        "The advancements in artificial intelligence have led to rapid changes in technology, "
        "driving both innovation and ethical debates across multiple industries. "
    )

    sizes = []
    speeds = []
    labels = []

    for info in models_info:
        model_name = info["name"]
        print(f"\nLoading model: {model_name} on device: {device}")
        model, tokenizer = load_model_and_tokenizer(model_name, device)
        print(f"Benchmarking {info['label']}...")
        tps = measure_inference_speed(model, tokenizer, prompt_text, device, max_new_tokens=max_new_tokens)
        print(f"  {info['label']}: {tps:.2f} tokens/sec")
        sizes.append(info["size"])
        speeds.append(tps)
        labels.append(info["label"])
        # Free resources between models
        del model, tokenizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if device.type == "mps":
            torch.mps.empty_cache()

    # Plot the model sizes vs. throughput
    plt.figure(figsize=(6, 4))
    plt.plot(sizes, speeds, marker='o', linestyle='--', color='steelblue')
    plt.title(f"Mistral Model Inference Speed on {device.type.upper()}")
    plt.xlabel("Model Size (Billion Parameters)")
    plt.ylabel("Throughput (Tokens per Second)")
    plt.grid(True)
    # Annotate each point with its label
    for x, y, label in zip(sizes, speeds, labels):
        plt.text(x, y, label, fontsize=8, ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
