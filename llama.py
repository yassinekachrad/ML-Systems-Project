# Llama 3.1 and 3.2 Models Inference Speed Benchmarking Script
# 
# This script loads all Llama 3.1 and Llama 3.2 models under 10B parameters (preferring Llama 3.2 when available)
# and measures their inference speed in tokens per second. It then plots model size vs tokens/sec.
#
# Note: Ensure you have access to the Llama 3.1/3.2 model weights on Hugging Face (accept the model license if required)
#       before running this script.

import time
import gc
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

def get_device(device_name: str) -> torch.device:
    """
    Resolve the desired computation device.
    Defaults to MPS (Metal) on macOS if available, or falls back to CPU/CUDA as specified.
    """
    device_name = device_name.lower()
    if device_name == "mps":
        if not torch.backends.mps.is_available() or not torch.backends.mps.is_built():
            print("Warning: MPS device not found or not built. Falling back to CPU.")
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
        raise ValueError(f"Unsupported device type '{device_name}'. Choose from 'mps', 'cuda', or 'cpu'.")

def load_model_and_tokenizer(model_name: str, device: torch.device):
    """
    Load a pretrained model and tokenizer from Hugging Face for text generation.
    Uses half precision (fp16) on GPU/MPS for efficiency, full precision on CPU.
    """
    # Determine appropriate torch dtype for model weights (fp16 for GPU/MPS, fp32 for CPU)
    if device.type in ("cuda", "mps"):
        dtype = torch.float16
    else:
        dtype = torch.float32
    # Load tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    # Move model to target device and set to eval mode
    model.to(device)
    model.eval()  # disable dropout and other training-specific layers
    return model, tokenizer

def measure_inference_speed(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int = 50) -> float:
    """
    Generate text from the model and measure tokens-per-second throughput.
    Returns the throughput in tokens per second for generating `max_new_tokens` tokens.
    """
    # Tokenize the input prompt and move input IDs (and mask if present) to the device
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device) if "attention_mask" in inputs else None

    # Warm-up generation (generate 1 token) to initialize model caches and GPU kernels
    with torch.no_grad():
        model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=1)
    # Clear GPU/MPS cache if applicable to ensure clean measurement
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()

    # Timed generation for the desired number of new tokens
    start_time = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            max_new_tokens=max_new_tokens, 
            do_sample=False  # deterministic generation (no sampling) for consistency
        )
    elapsed = time.perf_counter() - start_time

    # Calculate how many new tokens were generated (output length minus input length)
    new_tokens = output_ids.size(1) - input_ids.size(1)
    # Throughput in tokens per second
    tokens_per_sec = new_tokens / elapsed if elapsed > 0 else float("inf")
    return tokens_per_sec

def main():
    # Parse command-line arguments for device and generation length
    parser = argparse.ArgumentParser(description="Benchmark Llama 3.1 and 3.2 model inference speed.")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps",
                        help="Device to run the models on (default: mps for Apple Silicon, or specify cpu/cuda).")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Number of tokens to generate for measuring speed (default: 50).")
    args = parser.parse_args()
    device = get_device(args.device)
    max_new_tokens = args.max_new_tokens

    # List of (model_name, model_version_label, model_size) for Llama 3.1 and 3.2 under 10B
    models_info = [
        {"name": "meta-llama/Llama-3.2-1B", "label": "Llama3.2-1B", "size": 1.0},
        {"name": "meta-llama/Llama-3.2-3B", "label": "Llama3.2-3B", "size": 3.0},
        {"name": "meta-llama/Llama-3.1-8B", "label": "Llama3.1-8B", "size": 8.0}
    ]

    # Sample prompt (a sentence from a Wikipedia-style context) to use for all models
    prompt_text = (
        "The Apollo program was the third United States human spaceflight program "
        "carried out by NASA, which succeeded in landing the first humans on the Moon. "
    )
    # Using the same prompt across models ensures a fair comparison of speed.

    sizes = []
    speeds = []
    labels = []
    for info in models_info:
        model_name = info["name"]
        print(f"Loading model: {model_name} (device: {device})")
        model, tokenizer = load_model_and_tokenizer(model_name, device)
        # Measure throughput on the current model
        print(f"Measuring inference speed for {info['label']}...")
        tps = measure_inference_speed(model, tokenizer, prompt_text, device, max_new_tokens=max_new_tokens)
        print(f"  {info['label']}: {tps:.2f} tokens/sec")
        sizes.append(info["size"])
        speeds.append(tps)
        labels.append(info["label"])
        # Free memory before loading the next model
        del model, tokenizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if device.type == "mps":
            torch.mps.empty_cache()

    # Plot model size vs throughput
    plt.figure(figsize=(6, 4))
    plt.plot(sizes, speeds, marker='o', linestyle='--', color='steelblue')
    plt.title(f"Llama 3.1/3.2 Inference Speed on {device.type.upper()}")
    plt.xlabel("Model Size (Billion Parameters)")
    plt.ylabel("Throughput (Tokens per Second)")
    plt.grid(True)
    # Annotate each point with the model label for clarity
    for x, y, label in zip(sizes, speeds, labels):
        plt.text(x, y, label, fontsize=8, ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
