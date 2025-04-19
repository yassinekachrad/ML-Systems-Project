## To run this (example run): 
# python spec_decoding_main.py \
#     --model-family llama \
#     --use-env-token \
#     --benchmark \
#     --prompts-file prompts.txt \
#     --output-dir $SCRATCH/llama_benchmark_results \
#     --max-tokens 200 \
#     --spec-tokens 8 \
#     --run-baseline \
#     --detailed-timing \
#     --verbose


def run_benchmark(
    decoder,
    prompts,
    max_tokens=100,
    n_speculative_tokens=8,
    run_baseline=True,
    verbose=True,
    detailed_timing=True,
    output_dir=None
):
    """
    Run a benchmark comparing speculative decoding with baseline generation.
    
    Args:
        decoder: Initialized HierarchicalSpeculativeDecoder instance
        prompts: List of prompts to use for testing
        max_tokens: Maximum tokens to generate for each prompt
        n_speculative_tokens: Number of tokens to speculate at each step
        run_baseline: Whether to run baseline for comparison
        verbose: Whether to print detailed results
        detailed_timing: Whether to measure detailed timing information
        output_dir: Directory to save results (None = don't save)
        
    Returns:
        dict: Aggregated benchmark results
    """
    results = []
    
    # Track overall stats
    total_speculative_time = 0
    total_baseline_time = 0
    total_speculative_tokens = 0
    total_baseline_tokens = 0
    total_speedup = 0
    
    for i, prompt in enumerate(prompts):
        prompt_result = {"prompt": prompt, "prompt_id": i}
        
        print(f"\n======= Running benchmark {i+1}/{len(prompts)} =======")
        print(f"Prompt: {prompt[:50]}...")
        
        # Run speculative decoding
        print("\nRunning speculative decoding...")
        speculative_text, speculative_stats = decoder.generate(
            prompt, 
            max_tokens=max_tokens, 
            n_speculative_tokens=n_speculative_tokens,
            verbose=verbose,
            detailed_timing=detailed_timing
        )
        
        prompt_result["speculative"] = {
            "text": speculative_text,
            "stats": speculative_stats
        }
        
        total_speculative_time += speculative_stats["elapsed_time"]
        total_speculative_tokens += speculative_stats["total_tokens"]
        
        # Run baseline if requested
        if run_baseline:
            print("\nRunning baseline generation...")
            baseline_text, baseline_stats = run_baseline_generation(
                decoder.model_paths["large"],
                prompt,
                max_tokens=max_tokens,
                device=decoder.device,
                verbose=verbose,
                token=decoder.token if hasattr(decoder, 'token') else None,
                detailed_timing=detailed_timing
            )
            
            prompt_result["baseline"] = {
                "text": baseline_text,
                "stats": baseline_stats
            }
            
            total_baseline_time += baseline_stats["elapsed_time"]
            total_baseline_tokens += baseline_stats["total_tokens"]
            
            # Compare outputs
            output_file = None
            if output_dir:
                import os
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"comparison_{i}.json")
                
            comparison = compare_outputs(
                prompt,
                speculative_text,
                baseline_text,
                speculative_stats,
                baseline_stats,
                verbose=verbose,
                save_to_file=output_file
            )
            
            prompt_result["comparison"] = comparison
            total_speedup += comparison["speedup"]
        
        results.append(prompt_result)
    
    # Calculate aggregate stats
    avg_speedup = total_speedup / len(prompts) if run_baseline and len(prompts) > 0 else 0
    
    speculative_tokens_per_sec = total_speculative_tokens / total_speculative_time if total_speculative_time > 0 else 0
    
    if run_baseline:
        baseline_tokens_per_sec = total_baseline_tokens / total_baseline_time if total_baseline_time > 0 else 0
        overall_speedup = baseline_tokens_per_sec / speculative_tokens_per_sec if speculative_tokens_per_sec > 0 else 0
    else:
        baseline_tokens_per_sec = 0
        overall_speedup = 0
    
    # Compile benchmark summary
    benchmark_summary = {
        "num_prompts": len(prompts),
        "total_tokens": {
            "speculative": total_speculative_tokens,
            "baseline": total_baseline_tokens
        },
        "total_time": {
            "speculative": total_speculative_time,
            "baseline": total_baseline_time
        },
        "tokens_per_second": {
            "speculative": speculative_tokens_per_sec,
            "baseline": baseline_tokens_per_sec
        },
        "speedup": {
            "average": avg_speedup,
            "overall": overall_speedup
        },
        "detailed_results": results
    }
    
    # Save overall benchmark results if output directory provided
    if output_dir:
        import os
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "benchmark_summary.json")
        
        # Extract a simplified version of summary for saving
        save_summary = {k: v for k, v in benchmark_summary.items() if k != "detailed_results"}
        save_summary["prompt_ids"] = list(range(len(prompts)))
        
        with open(summary_path, 'w') as f:
            json.dump(save_summary, f, indent=2)
            
        if verbose:
            print(f"\nBenchmark summary saved to {summary_path}")
    
    # Print overall benchmark summary
    if verbose:
        print("\n========== Benchmark Summary ==========")
        print(f"Number of prompts: {len(prompts)}")
        print(f"Total tokens generated (speculative): {total_speculative_tokens}")
        if run_baseline:
            print(f"Total tokens generated (baseline): {total_baseline_tokens}")
        
        print(f"\nTotal time (speculative): {total_speculative_time:.2f}s")
        if run_baseline:
            print(f"Total time (baseline): {total_baseline_time:.2f}s")
        
        print(f"\nAverage tokens/second (speculative): {speculative_tokens_per_sec:.2f}")
        if run_baseline:
            print(f"Average tokens/second (baseline): {baseline_tokens_per_sec:.2f}")
            print(f"\nAverage speedup per prompt: {avg_speedup:.2f}x")
            print(f"Overall speedup: {overall_speedup:.2f}x")
    
    return benchmark_summaryimport torch
import time
import argparse
import os  # Added for environment variable access
from transformers import AutoModelForCausalLM, AutoTokenizer
from enum import Enum
from typing import List, Dict, Tuple, Optional, Union, Any


class ModelFamily(Enum):
    """Supported model families for hierarchical speculative decoding."""
    LLAMA = "llama"
    GEMMA = "gemma"


class ModelConfig:
    """Configuration for model paths and parameters."""
    
    # Detect available device
    @staticmethod
    def detect_default_device():
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    # Model family configurations
    FAMILY_CONFIGS = {
        ModelFamily.LLAMA: {
            "small": "meta-llama/Meta-Llama-3-8B",  # Updated from Llama-3.2-1B
            "medium": "meta-llama/Llama-3.1-8B",    # Updated from Llama-3.2-3B
            "large": "meta-llama/Llama-3.1-70B",    # Updated from Llama-3.2-8B
            "default_device": detect_default_device.__func__(),
            "dtype": torch.float16
        },
        ModelFamily.GEMMA: {
            "small": "google/gemma-3-1b",
            "medium": "google/gemma-3-4b",
            "large": "google/gemma-3-12b",
            "default_device": detect_default_device.__func__(),
            "dtype": torch.float16
        }
    }
    
    @staticmethod
    def get_model_paths(family: ModelFamily) -> Dict[str, str]:
        """Get model paths for a specific model family."""
        if family not in ModelConfig.FAMILY_CONFIGS:
            raise ValueError(f"Unsupported model family: {family}")
        
        config = ModelConfig.FAMILY_CONFIGS[family]
        return {
            "small": config["small"],
            "medium": config["medium"],
            "large": config["large"]
        }
    
    @staticmethod
    def get_default_device(family: ModelFamily) -> str:
        """Get default device for a model family."""
        return ModelConfig.FAMILY_CONFIGS[family]["default_device"]
    
    @staticmethod
    def get_default_dtype(family: ModelFamily) -> torch.dtype:
        """Get default dtype for a model family."""
        return ModelConfig.FAMILY_CONFIGS[family]["dtype"]


class HierarchicalSpeculativeDecoder:
    """
    Implements hierarchical speculative decoding using three models of different sizes.
    Verification follows a cascade: small model drafts → medium model verifies → large model verifies
    """
    
    def __init__(
        self, 
        model_family: ModelFamily,
        custom_model_paths: Optional[Dict[str, str]] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        token: Optional[str] = None  # Added token parameter
    ):
        """
        Initialize hierarchical speculative decoder with models from a specific family.
        
        Args:
            model_family: Which model family to use (LLAMA or GEMMA)
            custom_model_paths: Optional custom paths for models, must include 'small', 'medium', 'large' keys
            device: Device to run models on (defaults to family's default device)
            dtype: Data type for model weights (defaults to family's default dtype)
            load_in_8bit: Whether to load models in 8-bit precision (saves memory)
            load_in_4bit: Whether to load models in 4-bit precision (saves more memory)
            token: Hugging Face access token for gated models
        """
        self.model_family = model_family
        self.device = device or ModelConfig.get_default_device(model_family)
        self.dtype = dtype or ModelConfig.get_default_dtype(model_family)
        self.token = token  # Store token for authentication
        
        # MPS compatibility check - quantization not supported on MPS
        if self.device == "mps" and (load_in_8bit or load_in_4bit):
            print("Warning: Quantization (8-bit/4-bit) is not supported on MPS. Disabling quantization.")
            load_in_8bit = False
            load_in_4bit = False
        
        # Get model paths (either custom or default)
        if custom_model_paths:
            required_keys = {'small', 'medium', 'large'}
            if not required_keys.issubset(custom_model_paths.keys()):
                missing = required_keys - set(custom_model_paths.keys())
                raise ValueError(f"Missing required model paths: {missing}")
            self.model_paths = custom_model_paths
        else:
            self.model_paths = ModelConfig.get_model_paths(model_family)
        
        # Load models and tokenizers
        self.models = {}
        self.tokenizers = {}
        
        # Set quantization config if needed and available
        quantization_config = None
        if load_in_8bit or load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=self.dtype
            )
        
        # Load models with proper sizes (small, medium, large)
        for size, path in self.model_paths.items():
            print(f"Loading {size} model from {path}...")
            
            # Load tokenizer with authentication token if provided
            tokenizer_kwargs = {}
            if self.token:
                tokenizer_kwargs["token"] = self.token
            
            self.tokenizers[size] = AutoTokenizer.from_pretrained(path, **tokenizer_kwargs)
            
            # Prepare model loading kwargs with authentication
            model_kwargs = {"torch_dtype": self.dtype}
            if self.token:
                model_kwargs["token"] = self.token
            
            # MPS and CPU need different loading methods from CUDA
            if self.device == "mps" or self.device == "cpu":
                # For MPS or CPU we need to load first then move to device
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    **model_kwargs
                )
                # Move model to MPS device
                self.models[size] = model.to(self.device)
            else:
                # For CUDA we can use device_map
                if quantization_config:
                    model_kwargs["quantization_config"] = quantization_config
                    model_kwargs["device_map"] = self.device
                else:
                    model_kwargs["device_map"] = self.device
                
                self.models[size] = AutoModelForCausalLM.from_pretrained(
                    path,
                    **model_kwargs
                )
        
        # Give models readable names for logging
        self.model_names = {
            "small": f"{model_family.value} small ({self.model_paths['small'].split('/')[-1]})",
            "medium": f"{model_family.value} medium ({self.model_paths['medium'].split('/')[-1]})",
            "large": f"{model_family.value} large ({self.model_paths['large'].split('/')[-1]})"
        }
        
        # Ensure all models share compatible tokenization
        self._verify_compatible_tokenizers()
    
    def _verify_compatible_tokenizers(self):
        """Verify that all tokenizers are compatible enough for our use case."""
        special_tokens = {
            "small": set(self.tokenizers["small"].all_special_tokens),
            "medium": set(self.tokenizers["medium"].all_special_tokens),
            "large": set(self.tokenizers["large"].all_special_tokens),
        }
        
        # Check for major differences in special tokens
        if len(special_tokens["small"] ^ special_tokens["medium"]) > 0:
            print(f"Warning: Special token mismatch between small and medium models")
        
        if len(special_tokens["medium"] ^ special_tokens["large"]) > 0:
            print(f"Warning: Special token mismatch between medium and large models")
        
        # Check if tokenizers have the same vocabulary size
        vocab_sizes = {
            "small": len(self.tokenizers["small"]),
            "medium": len(self.tokenizers["medium"]),
            "large": len(self.tokenizers["large"]),
        }
        
        if len(set(vocab_sizes.values())) > 1:
            print(f"Warning: Different vocabulary sizes: {vocab_sizes}")
            print("This might lead to inconsistent token prediction across models")
    
    def _generate_candidates(self, input_ids, n_tokens=8):
        """
        Generate candidate tokens from the smallest model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            n_tokens (int): Maximum number of tokens to generate
            
        Returns:
            torch.Tensor: Candidate tokens from the smallest model
        """
        # Generate with small model
        with torch.no_grad():
            draft_output = self.models["small"].generate(
                input_ids,
                max_new_tokens=n_tokens,
                do_sample=False  # Greedy decoding
            )
            
        # Extract new tokens (excluding input)
        draft_new_tokens = draft_output[:, input_ids.shape[1]:]
        
        return draft_new_tokens
    
    def _verify_candidates(self, input_ids, candidates):
        """
        Verify candidates using hierarchical verification with medium and largest models.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            candidates (torch.Tensor): Candidate tokens from the smallest model
            
        Returns:
            torch.Tensor: Accepted token IDs
            Dict: Statistics about verification
        """
        if candidates.shape[1] == 0:
            return input_ids, {"medium_verified": 0, "large_verified": 0}
        
        # First verification step: Use medium model to verify smallest model's candidates
        medium_verified, medium_stats = self._verify_with_model(
            input_ids, candidates, self.models["medium"], "medium"
        )
        
        if medium_verified.shape[1] == input_ids.shape[1]:
            # Medium model rejected all tokens
            return input_ids, {"medium_verified": 0, "large_verified": 0}
        
        # Extract the tokens accepted by medium model
        medium_accepted_tokens = medium_verified[:, input_ids.shape[1]:]
        
        # Second verification step: Use largest model to verify medium model's accepted tokens
        large_verified, large_stats = self._verify_with_model(
            input_ids, medium_accepted_tokens, self.models["large"], "large"
        )
        
        # Combine statistics
        stats = {
            "medium_verified": medium_stats["verified_count"],
            "large_verified": large_stats["verified_count"]
        }
        
        return large_verified, stats
    
    def _verify_with_model(self, input_ids, candidate_tokens, verification_model, model_name):
        """
        Verify candidate tokens with a specific model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            candidate_tokens (torch.Tensor): Candidate tokens to verify
            verification_model: Model to use for verification
            model_name (str): Name of the model for logging
            
        Returns:
            torch.Tensor: Accepted token IDs
            Dict: Statistics about verification
        """
        # Prepare for verification
        n_candidates = candidate_tokens.shape[1]
        full_sequence = torch.cat([input_ids, candidate_tokens], dim=1)
        
        # Get verification model logits for the sequence
        with torch.no_grad():
            outputs = verification_model(full_sequence)
            logits = outputs.logits
        
        # Validate each candidate token
        accepted_tokens = input_ids.clone()
        verified_count = 0
        
        for i in range(n_candidates):
            position = input_ids.shape[1] + i
            if position >= logits.shape[1]:
                break
            
            # Get verification model prediction for this position
            verifier_logits = logits[:, position - 1, :]
            verifier_probs = torch.softmax(verifier_logits, dim=-1)
            best_verifier_token = torch.argmax(verifier_probs, dim=-1)
            
            # Get candidate token at this position
            candidate_token = full_sequence[:, position]
            
            # If they match, accept the candidate token
            if best_verifier_token == candidate_token:
                accepted_tokens = torch.cat([accepted_tokens, candidate_token.unsqueeze(0).unsqueeze(0)], dim=1)
                verified_count += 1
            else:
                # Mismatch - stop accepting and return what we've got
                break
        
        stats = {
            "verified_count": verified_count,
            "total_candidates": n_candidates,
            "acceptance_rate": verified_count / n_candidates if n_candidates > 0 else 0
        }
        
        return accepted_tokens, stats
    
    def generate(self, prompt, max_tokens=100, n_speculative_tokens=8, verbose=True, detailed_timing=False):
        """
        Generate text using hierarchical speculative decoding.
        
        Args:
            prompt (str): Input text prompt
            max_tokens (int): Maximum number of tokens to generate
            n_speculative_tokens (int): Number of tokens to speculate at each step
            verbose (bool): Whether to print detailed stats during generation
            detailed_timing (bool): Whether to measure detailed timing information
            
        Returns:
            str: Generated text
            dict: Performance statistics
        """
        input_ids = self.tokenizers["large"].encode(prompt, return_tensors="pt").to(self.device)
        generated_ids = input_ids.clone()
        
        total_new_tokens = 0
        start_time = time.time()
        
        # Create timer for detailed timing if requested
        timer = InferenceTimer() if detailed_timing else None
        
        # Track metrics for analysis
        total_draft_tokens_proposed = 0
        total_medium_tokens_accepted = 0
        total_large_tokens_accepted = 0
        total_fallback_tokens = 0
        
        while total_new_tokens < max_tokens:
            # 1. Generate candidates from small model
            if detailed_timing:
                timer.start("small_model_generation")
            
            candidates = self._generate_candidates(generated_ids, n_tokens=n_speculative_tokens)
            total_draft_tokens_proposed += candidates.shape[1]
            
            if detailed_timing:
                timer.end()
            
            # 2. Hierarchical verification (medium then large)
            prev_length = generated_ids.shape[1]
            
            if detailed_timing:
                timer.start("hierarchical_verification")
                
            generated_ids, verification_stats = self._verify_candidates(generated_ids, candidates)
            
            if detailed_timing:
                timer.end()
            
            # Update metrics
            medium_verified = verification_stats["medium_verified"]
            large_verified = verification_stats["large_verified"]
            total_medium_tokens_accepted += medium_verified
            total_large_tokens_accepted += large_verified
            
            # If no tokens were accepted, generate one token with the large model (fallback)
            accepted_tokens = generated_ids.shape[1] - prev_length
            if accepted_tokens == 0:
                if detailed_timing:
                    timer.start("fallback_generation")
                    
                with torch.no_grad():
                    target_output = self.models["large"].generate(
                        generated_ids,
                        max_new_tokens=1,
                        do_sample=False
                    )
                generated_ids = target_output
                total_fallback_tokens += 1
                
                if detailed_timing:
                    timer.end()
            
            # Update total tokens
            total_new_tokens = generated_ids.shape[1] - input_ids.shape[1]
            
            # Print progress
            if verbose and total_new_tokens % 10 == 0:
                print(f"Generated {total_new_tokens} tokens")
                
            # Check if EOS token was generated
            if self.tokenizers["large"].eos_token_id in generated_ids[0, input_ids.shape[1]:]:
                break
        
        elapsed_time = time.time() - start_time
        tokens_per_second = total_new_tokens / elapsed_time if elapsed_time > 0 else 0
        
        # Compile performance metrics
        stats = {
            "total_tokens": total_new_tokens,
            "elapsed_time": elapsed_time,
            "tokens_per_second": tokens_per_second,
            "draft_tokens_proposed": total_draft_tokens_proposed,
            "medium_tokens_accepted": total_medium_tokens_accepted,
            "large_tokens_accepted": total_large_tokens_accepted,
            "fallback_tokens": total_fallback_tokens,
            "medium_acceptance_rate": total_medium_tokens_accepted / total_draft_tokens_proposed if total_draft_tokens_proposed > 0 else 0,
            "large_acceptance_rate": total_large_tokens_accepted / total_medium_tokens_accepted if total_medium_tokens_accepted > 0 else 0,
            "overall_acceptance_rate": total_large_tokens_accepted / total_draft_tokens_proposed if total_draft_tokens_proposed > 0 else 0,
            "fallback_rate": total_fallback_tokens / total_new_tokens if total_new_tokens > 0 else 0,
            "search_ratio": total_draft_tokens_proposed / total_new_tokens if total_new_tokens > 0 else 0
        }
        
        # Add detailed timing stats if available
        if detailed_timing:
            timer_stats = timer.get_stats()
            stats["detailed_timing"] = timer_stats
        
        # Print summary statistics
        if verbose:
            print(f"\nGeneration complete: {total_new_tokens} tokens in {elapsed_time:.2f}s ({tokens_per_second:.2f} tokens/s)")
            print(f"Draft tokens proposed: {total_draft_tokens_proposed}")
            print(f"Medium model accepted: {total_medium_tokens_accepted} ({stats['medium_acceptance_rate']*100:.2f}%)")
            print(f"Large model accepted: {total_large_tokens_accepted} "
                 f"({stats['large_acceptance_rate']*100:.2f}% of medium, "
                 f"{stats['overall_acceptance_rate']*100:.2f}% of draft)")
            print(f"Fallback tokens: {total_fallback_tokens} ({stats['fallback_rate']*100:.2f}%)")
            print(f"Tree search ratio: {stats['search_ratio']:.2f}x")
            
            # Log detailed timing if available
            if detailed_timing:
                timer.log_stats()
        
        # Decode the generated text
        generated_text = self.tokenizers["large"].decode(generated_ids[0], skip_special_tokens=True)
        return generated_text, stats


def run_baseline_generation(
    model_path, 
    prompt, 
    max_tokens=100, 
    device="cuda", 
    dtype=torch.float16,
    verbose=True,
    token=None,  # Added token parameter
    detailed_timing=False  # Added detailed timing parameter
):
    """
    Run generation with just the target (large) model for baseline comparison.
    
    Args:
        model_path: Path to the target model
        prompt: Input text prompt
        max_tokens: Maximum number of tokens to generate
        device: Device to run on
        dtype: Data type for model weights
        verbose: Whether to print details
        token: Hugging Face access token for gated models
        detailed_timing: Whether to collect detailed timing information
        
    Returns:
        str: Generated text
        dict: Performance statistics
    """
    if verbose:
        print(f"\nGenerating baseline response with just target model ({model_path})...")
    
    # Create timer for detailed timing if requested
    timer = InferenceTimer() if detailed_timing else None
    
    # Prepare arguments with authentication if provided
    tokenizer_kwargs = {}
    model_kwargs = {"torch_dtype": dtype}
    
    if token:
        tokenizer_kwargs["token"] = token
        model_kwargs["token"] = token
    
    # Load model and tokenizer
    if detailed_timing:
        timer.start("model_loading")
        
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    
    # Handle different devices appropriately
    if device == "mps" or device == "cpu":
        # For MPS or CPU we need to load first then move to device
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        model = model.to(device)
    else:
        # For CUDA we can use device_map
        model_kwargs["device_map"] = device
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
    
    if detailed_timing:
        timer.end()
    
    # Generate text
    start_time = time.time()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    if detailed_timing:
        timer.start("tokenization")
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        timer.end()
        
        timer.start("to_device")
        input_ids = input_ids.to(device)
        timer.end()
    
    if detailed_timing:
        timer.start("generation")
        
    with torch.no_grad():
        baseline_output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False
        )
    
    if detailed_timing:
        timer.end()
    
    # Calculate metrics
    elapsed_time = time.time() - start_time
    new_tokens = baseline_output.shape[1] - input_ids.shape[1]
    tokens_per_second = new_tokens / elapsed_time if elapsed_time > 0 else 0
    
    if verbose:
        print(f"Baseline: {new_tokens} tokens in {elapsed_time:.2f}s ({tokens_per_second:.2f} tokens/s)")
    
    # Decode text
    if detailed_timing:
        timer.start("decoding")
    generated_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
    if detailed_timing:
        timer.end()
    
    # Compile stats
    stats = {
        "total_tokens": new_tokens,
        "elapsed_time": elapsed_time,
        "tokens_per_second": tokens_per_second
    }
    
    # Add detailed timing stats if available
    if detailed_timing:
        timer_stats = timer.get_stats()
        stats["detailed_timing"] = timer_stats
        timer.log_stats()
    
    return generated_text, stats
    
    # Generate text
    start_time = time.time()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        baseline_output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False
        )
    
    # Calculate metrics
    elapsed_time = time.time() - start_time
    new_tokens = baseline_output.shape[1] - input_ids.shape[1]
    tokens_per_second = new_tokens / elapsed_time if elapsed_time > 0 else 0
    
    if verbose:
        print(f"Baseline: {new_tokens} tokens in {elapsed_time:.2f}s ({tokens_per_second:.2f} tokens/s)")
    
    # Decode text
    generated_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
    
    stats = {
        "total_tokens": new_tokens,
        "elapsed_time": elapsed_time,
        "tokens_per_second": tokens_per_second
    }
    
    return generated_text, stats


def compare_outputs(
    prompt, 
    speculative_text, 
    baseline_text, 
    speculative_stats, 
    baseline_stats,
    verbose=True,
    save_to_file=None
):
    """
    Compare the outputs and performance of speculative decoding vs baseline.
    
    Args:
        prompt: Input prompt
        speculative_text: Text generated with speculative decoding
        baseline_text: Text generated with baseline approach
        speculative_stats: Stats from speculative decoding
        baseline_stats: Stats from baseline generation
        verbose: Whether to print outputs
        save_to_file: Path to save the comparison results (None = don't save)
        
    Returns:
        dict: Comparison statistics
    """
    # Calculate speedup and other metrics
    speedup = baseline_stats["elapsed_time"] / speculative_stats["elapsed_time"] if speculative_stats["elapsed_time"] > 0 else 0
    speedup_per_token = baseline_stats["tokens_per_second"] / speculative_stats["tokens_per_second"] if speculative_stats["tokens_per_second"] > 0 else 0
    
    # Build detailed comparison
    comparison = {
        "speedup": speedup,
        "speedup_per_token": speedup_per_token,
        "speculative_tokens_per_second": speculative_stats["tokens_per_second"],
        "baseline_tokens_per_second": baseline_stats["tokens_per_second"],
        "speculative_time": speculative_stats["elapsed_time"],
        "baseline_time": baseline_stats["elapsed_time"],
        "speculative_tokens": speculative_stats["total_tokens"],
        "baseline_tokens": baseline_stats["total_tokens"],
        "outputs_match": speculative_text == baseline_text,
        "draft_token_stats": {
            "proposed": speculative_stats.get("draft_tokens_proposed", 0),
            "medium_accepted": speculative_stats.get("medium_tokens_accepted", 0),
            "large_accepted": speculative_stats.get("large_tokens_accepted", 0),
            "fallback": speculative_stats.get("fallback_tokens", 0)
        }
    }
    
    # Calculate tokenization overlap (how much of the outputs are identical)
    if baseline_text != speculative_text:
        baseline_tokens = baseline_text.split()
        speculative_tokens = speculative_text.split()
        min_len = min(len(baseline_tokens), len(speculative_tokens))
        
        matching_tokens = 0
        for i in range(min_len):
            if baseline_tokens[i] == speculative_tokens[i]:
                matching_tokens += 1
        
        if min_len > 0:
            comparison["token_similarity"] = matching_tokens / min_len
        else:
            comparison["token_similarity"] = 0
    else:
        comparison["token_similarity"] = 1.0
    
    if verbose:
        print("\n============= Performance Comparison =============")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Speculative: {speculative_stats['tokens_per_second']:.2f} tokens/s, {speculative_stats['elapsed_time']:.2f}s total")
        print(f"Baseline: {baseline_stats['tokens_per_second']:.2f} tokens/s, {baseline_stats['elapsed_time']:.2f}s total")
        
        print("\n=================== Token Stats ==================")
        print(f"Speculative tokens: {speculative_stats['total_tokens']}")
        print(f"Baseline tokens: {baseline_stats['total_tokens']}")
        
        print("\n================ Speculative Stats ===============")
        print(f"Draft tokens proposed: {speculative_stats.get('draft_tokens_proposed', 0)}")
        print(f"Medium model accepted: {speculative_stats.get('medium_tokens_accepted', 0)} " 
              f"({speculative_stats.get('medium_acceptance_rate', 0)*100:.2f}%)")
        print(f"Large model accepted: {speculative_stats.get('large_tokens_accepted', 0)} "
              f"({speculative_stats.get('overall_acceptance_rate', 0)*100:.2f}% of all proposed)")
        print(f"Fallback tokens: {speculative_stats.get('fallback_tokens', 0)} " 
              f"({speculative_stats.get('fallback_rate', 0)*100:.2f}% of total)")
        
        print("\n================= Output Comparison ================")
        print(f"Outputs match: {comparison['outputs_match']}")
        if not comparison['outputs_match']:
            print(f"Token similarity: {comparison['token_similarity']*100:.2f}%")
            
            # Show outputs with diff markers for first 500 chars
            print("\nSpeculative Decoding output:")
            print("-" * 40)
            print(speculative_text[:500] + ("..." if len(speculative_text) > 500 else ""))
            print("\nBaseline output:")
            print("-" * 40)
            print(baseline_text[:500] + ("..." if len(baseline_text) > 500 else ""))
    
    # Save comparison to file if requested
    if save_to_file:
        import json
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_to_file) or '.', exist_ok=True)
        
        # Prepare data for saving
        save_data = {
            "prompt": prompt,
            "comparison": comparison,
            "speculative_stats": speculative_stats,
            "baseline_stats": baseline_stats,
            "speculative_text": speculative_text[:1000],  # Truncate long text
            "baseline_text": baseline_text[:1000],        # Truncate long text
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save to file
        with open(save_to_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        if verbose:
            print(f"\nResults saved to {save_to_file}")
    
    return comparison


def main():
    """Command-line interface for hierarchical speculative decoding."""
    parser = argparse.ArgumentParser(description="Hierarchical Speculative Decoding")
    
    # Model configuration
    parser.add_argument("--model-family", type=str, choices=["llama", "gemma"], required=True,
                      help="Model family to use (llama or gemma)")
    parser.add_argument("--small-model", type=str, help="Path to small model (if custom)")
    parser.add_argument("--medium-model", type=str, help="Path to medium model (if custom)")
    parser.add_argument("--large-model", type=str, help="Path to large model (if custom)")
    
    # Decoding configuration
    parser.add_argument("--prompt", type=str, 
                      default="Explain how hierarchical speculative decoding works in language models:",
                      help="Prompt for text generation")
    parser.add_argument("--max-tokens", type=int, default=200,
                      help="Maximum number of tokens to generate")
    parser.add_argument("--spec-tokens", type=int, default=8,
                      help="Number of tokens to speculate at each step")
    
    # Hardware configuration
    parser.add_argument("--device", type=str, default=None,
                      help="Device to run models on (default: auto-detect)")
    parser.add_argument("--load-in-8bit", action="store_true",
                      help="Load models in 8-bit precision")
    parser.add_argument("--load-in-4bit", action="store_true",
                      help="Load models in 4-bit precision")
    
    # Authentication
    parser.add_argument("--token", type=str, help="Hugging Face access token for gated models")
    parser.add_argument("--use-env-token", action="store_true",
                      help="Use HF_TOKEN environment variable for authentication")
    
    # Run options
    parser.add_argument("--run-baseline", action="store_true",
                      help="Run baseline generation for comparison")
    parser.add_argument("--verbose", action="store_true", default=True,
                      help="Print detailed information")
    parser.add_argument("--detailed-timing", action="store_true",
                      help="Measure detailed timing information for each phase")
    
    # Benchmark options
    parser.add_argument("--benchmark", action="store_true",
                      help="Run benchmark on multiple prompts")
    parser.add_argument("--prompts-file", type=str,
                      help="Path to file containing prompts for benchmarking (one per line)")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                      help="Directory to save benchmark results")
    parser.add_argument("--num-runs", type=int, default=1,
                      help="Number of times to run each prompt (for averaging)")
    
    args = parser.parse_args()
    
    # Check for token in environment variable if requested
    token = args.token
    if args.use_env_token and not token:
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("Warning: --use-env-token specified but HF_TOKEN environment variable not found.")
    
    # Set up model family
    model_family = ModelFamily(args.model_family)
    
    # Set up custom model paths if provided
    custom_model_paths = None
    if args.small_model or args.medium_model or args.large_model:
        custom_model_paths = {}
        if args.small_model:
            custom_model_paths["small"] = args.small_model
        if args.medium_model:
            custom_model_paths["medium"] = args.medium_model
        if args.large_model:
            custom_model_paths["large"] = args.large_model
            
        # Fill in any missing paths with defaults
        for size in ["small", "medium", "large"]:
            if size not in custom_model_paths:
                custom_model_paths[size] = ModelConfig.get_model_paths(model_family)[size]
    
    # Create speculative decoder
    decoder = HierarchicalSpeculativeDecoder(
        model_family=model_family,
        custom_model_paths=custom_model_paths,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        token=token  # Pass the token for authentication
    )
    
    # Handle benchmark mode
    if args.benchmark:
        # Load prompts from file or use default
        prompts = []
        if args.prompts_file:
            try:
                with open(args.prompts_file, 'r') as f:
                    prompts = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print(f"Error loading prompts file: {e}")
                return
        else:
            # Use some default prompts
            prompts = [
                "Explain how hierarchical speculative decoding works in language models:",
                "What are the key differences between transformer and RNN architectures?",
                "Describe the process of training a neural network using backpropagation.",
                "What are the ethical considerations when deploying large language models?",
                "Explain the concept of attention mechanisms in transformers."
            ]
        
        if not prompts:
            print("Error: No prompts found for benchmarking.")
            return
        
        print(f"Running benchmark with {len(prompts)} prompts, {args.num_runs} run(s) each...")
        
        all_results = []
        for run in range(args.num_runs):
            output_dir = args.output_dir
            if args.num_runs > 1:
                output_dir = os.path.join(args.output_dir, f"run_{run+1}")
                
            print(f"\n===== Starting benchmark run {run+1}/{args.num_runs} =====")
            
            run_results = run_benchmark(
                decoder,
                prompts,
                max_tokens=args.max_tokens,
                n_speculative_tokens=args.spec_tokens,
                run_baseline=args.run_baseline,
                verbose=args.verbose,
                detailed_timing=args.detailed_timing,
                output_dir=output_dir
            )
            
            all_results.append(run_results)
        
        # If multiple runs, compute average stats
        if args.num_runs > 1:
            avg_speculative_tokens_per_sec = sum(res["tokens_per_second"]["speculative"] for res in all_results) / args.num_runs
            
            if args.run_baseline:
                avg_baseline_tokens_per_sec = sum(res["tokens_per_second"]["baseline"] for res in all_results) / args.num_runs
                avg_speedup = sum(res["speedup"]["overall"] for res in all_results) / args.num_runs
                
                print("\n===== Average Results Across All Runs =====")
                print(f"Average speculative tokens/sec: {avg_speculative_tokens_per_sec:.2f}")
                print(f"Average baseline tokens/sec: {avg_baseline_tokens_per_sec:.2f}")
                print(f"Average speedup: {avg_speedup:.2f}x")
            else:
                print("\n===== Average Results Across All Runs =====")
                print(f"Average speculative tokens/sec: {avg_speculative_tokens_per_sec:.2f}")
    else:
        # Regular single prompt mode
        # Generate text with speculative decoding
        print(f"\nPrompt: {args.prompt}")
        print(f"\nGenerating response with hierarchical speculative decoding...")
        speculative_text, speculative_stats = decoder.generate(
            args.prompt, 
            max_tokens=args.max_tokens, 
            n_speculative_tokens=args.spec_tokens,
            verbose=args.verbose,
            detailed_timing=args.detailed_timing
        )
        
        print("\nGenerated response:")
        print(speculative_text)
        
        # Optional: Run baseline for comparison
        if args.run_baseline:
            baseline_text, baseline_stats = run_baseline_generation(
                decoder.model_paths["large"],
                args.prompt,
                max_tokens=args.max_tokens,
                device=args.device,
                verbose=args.verbose,
                token=token,  # Pass the token for authentication
                detailed_timing=args.detailed_timing
            )
            
            print("\nBaseline response:")
            print(baseline_text)
            
            # Compare outputs
            compare_outputs(
                args.prompt,
                speculative_text,
                baseline_text,
                speculative_stats,
                baseline_stats,
                verbose=args.verbose
            )


if __name__ == "__main__":
    main()