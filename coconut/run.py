import os
import sys
import argparse
import torch

# Add the project directory to the Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir)

# Detect available hardware
has_cuda = torch.cuda.is_available()
has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

# Print device information
if has_cuda:
    print(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    device = "cuda"
elif has_mps:
    print("Apple Silicon GPU (MPS) detected")
    device = "mps"
else:
    print("No GPU detected, using CPU (training will be slow)")
    device = "cpu"

# Apply GPU optimizations with memory constraints in mind
if device != "cpu":
    print(f"Applying GPU optimizations for {device} with memory efficiency...")
    
    # Common optimizations for any GPU type
    os.environ["TRANSFORMERS_GRADIENT_CHECKPOINTING"] = "1"
    
    # Enable GPU-specific environment variables
    if device == "cuda":
        # CUDA-specific optimizations
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        
        # Memory optimizations for CUDA
        # Enable TF32 for NVIDIA Ampere GPUs (A100, etc.)
        os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
        
        # Get GPU memory info
        try:
            free_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            print(f"Total GPU memory: {free_memory_mb:.0f}MB")
            
            # Adjust memory settings based on GPU RAM
            if free_memory_mb < 8000:  # Less than 8GB GPU
                print("Low memory GPU detected, applying extreme memory optimization")
                # Force 4-bit quantization
                if "--load_in_4bit" not in sys.argv:
                    sys.argv.extend(["--load_in_4bit", "true"])
            elif free_memory_mb < 12000:  # Less than 12GB GPU
                print("Mid-range GPU detected, applying moderate memory optimization")
                # Force 8-bit quantization if 4-bit not specified
                if "--load_in_4bit" not in sys.argv and "--load_in_8bit" not in sys.argv:
                    sys.argv.extend(["--load_in_8bit", "true"])
        except:
            print("Could not detect GPU memory, applying default memory optimizations")
            
    elif device == "mps":
        # MPS-specific optimizations
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        # Additional MPS optimizations
        # Keep more tensors in main memory to avoid MPS out-of-memory errors
        import psutil
        total_ram = psutil.virtual_memory().total / (1024**3)
        print(f"System memory: {total_ram:.1f}GB")
        
        # Set memory limit for MPS based on available system RAM
        if total_ram > 32:  # More than 32GB RAM
            print("High memory system detected for MPS")
        elif total_ram > 16:  # 16-32GB RAM
            print("Medium memory system detected for MPS")
            # Add MPS-specific optimizations here
        else:  # Less than 16GB RAM
            print("Low memory system detected for MPS, applying extreme memory optimization")
            # Add extreme MPS optimizations here

# Apply model-specific optimizations
if '--model_type' in sys.argv and 'gemma' in sys.argv[sys.argv.index('--model_type') + 1]:
    print("Applying Gemma model optimizations...")
    
    # If batch size is not specified, set it to 1 for CPU/MPS, 2 for CUDA
    if '--batch_size' not in sys.argv:
        batch_size = "2" if device == "cuda" else "1"
        sys.argv.extend(['--batch_size', batch_size])
    
    # If gradient accumulation is not specified
    if '--gradient_accumulation_steps' not in sys.argv:
        # Higher accumulation steps for CPU, lower for GPU
        accum_steps = "2" if device == "cuda" else "4"
        sys.argv.extend(['--gradient_accumulation_steps', accum_steps])
    
    # Set shorter sequence length for faster training
    if '--max_length' not in sys.argv:
        max_length = "256" if device != "cpu" else "128"
        sys.argv.extend(['--max_length', max_length])

from coconut.main import main

if __name__ == "__main__":
    main()