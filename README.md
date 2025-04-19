## Local Setup
1. Environment Setup
First, create a Python environment with the required packages:
```
bash# Create a virtual environment
python -m venv spec_decode_env
source spec_decode_env/bin/activate  # On Windows: spec_decode_env\Scripts\activate
```

2. Install required packages
```pip install torch transformers accelerate huggingface_hub```

3. Authenticate Hugging Face token
``` huggingface-cli login```

4. Run the script spec_decoding_main.py

## PSC Setup
Git clone the repo and test using an interactive session.
### Example Interactive Session

For development and testing, you can use an interactive session:
``` # Request an interactive session with 2 GPUs
interact -p GPU-shared --gres=gpu:2 -N 1 -n 1 -t 01:00:00

# When the session starts, you'll be on a compute node
# Load modules and run your code directly
module load python cuda
source $SCRATCH/spec_decode_env/bin/activate
cd $SCRATCH/speculative_decoding
python modular_speculative_decoding.py --model-family gemma --use-env-token
```