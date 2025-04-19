#!/bin/bash
#SBATCH -J llama_spec_decode    # Job name
#SBATCH -p GPU                  # Partition (queue) name
#SBATCH -N 1                    # Number of nodes
#SBATCH -n 1                    # Number of tasks
#SBATCH --gpus=4                # Number of GPUs (max 8 per node)
#SBATCH -t 4:00:00              # Run time (hh:mm:ss)
#SBATCH -A abc123               # Project to charge against (replace with yours)
#SBATCH -o llama_spec_%j.out    # Standard output log
#SBATCH -e llama_spec_%j.err    # Standard error log

# Load modules
module purge
module load python/3.9.0
module load cuda/11.7.0
module list

# Activate virtual environment
source $SCRATCH/spec_decode_env/bin/activate

# Set Hugging Face token
export HF_TOKEN=""  #TODO(all): Replace with your token (can't upload to github with my private token)

# Go to working directory
cd $SCRATCH/speculative_decoding

# Run benchmark with Llama models
python spec_decoding_main.py \
    --model-family llama \
    --use-env-token \
    --benchmark \
    --prompts-file prompts.txt \
    --output-dir $SCRATCH/llama_benchmark_results \
    --max-tokens 200 \
    --spec-tokens 8 \
    --run-baseline \
    --detailed-timing \
    --verbose