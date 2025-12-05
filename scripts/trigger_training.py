import sys
import os
import subprocess
import logging
import shutil

# Add the project root to the Python path to allow for absolute imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from lpm_kernel.common.logging import logger

def download_model_from_hf(hf_model_id: str, local_model_name: str) -> str:
    """
    Download model from Hugging Face and set up in base_models directory.
    
    Args:
        hf_model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-0.6B")
        local_model_name: Local folder name for the model
        
    Returns:
        Path to the local model directory
    """
    from huggingface_hub import snapshot_download
    
    base_model_dir = os.path.join(PROJECT_ROOT, "resources", "L2", "base_models", local_model_name)
    
    # Check if model already exists
    if os.path.exists(base_model_dir) and os.listdir(base_model_dir):
        logger.info(f"Model already exists at: {base_model_dir}")
        return base_model_dir
    
    logger.info(f"Downloading model from HuggingFace: {hf_model_id}")
    
    # Download to HF cache
    hf_cache_path = snapshot_download(
        repo_id=hf_model_id,
        resume_download=True,
    )
    logger.info(f"Downloaded to cache: {hf_cache_path}")
    
    # Create symlink or copy to base_models
    os.makedirs(os.path.dirname(base_model_dir), exist_ok=True)
    
    try:
        if os.path.exists(base_model_dir):
            os.rmdir(base_model_dir)
        os.symlink(hf_cache_path, base_model_dir, target_is_directory=True)
        logger.info(f"Created symlink: {base_model_dir} -> {hf_cache_path}")
    except (OSError, NotImplementedError) as e:
        logger.warning(f"Symlink failed ({e}), copying model files...")
        if os.path.exists(base_model_dir):
            shutil.rmtree(base_model_dir)
        shutil.copytree(hf_cache_path, base_model_dir)
        logger.info(f"Copied model to: {base_model_dir}")
    
    return base_model_dir

def direct_lora_training():
    """
    Directly run LoRA training, bypassing all previous stages.
    Assumes data is already prepared in resources/L2/data/merged.json
    """
    logger.info("Starting direct LoRA training...")
    
    # --- Check GPU availability ---
    import torch
    use_cuda = False
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
        use_cuda = True
    else:
        logger.warning("CUDA not available! Training will use CPU (much slower)")
        use_cuda = False

    # --- Model Configuration ---
    # HuggingFace model ID and local folder name
    # Using Qwen3-8B for efficient LoRA fine-tuning
    HF_MODEL_ID = "Qwen/Qwen3-8B"
    LOCAL_MODEL_NAME = "Qwen3-8B"
    
    logger.info(f"Preparing LoRA fine-tuning for: {HF_MODEL_ID}")
    logger.info("This model requires trust_remote_code=True to load custom modeling code")
    
    # --- Download/Setup Model ---
    base_model_path = download_model_from_hf(HF_MODEL_ID, LOCAL_MODEL_NAME)
    
    # --- Paths ---
    dataset_path = os.path.join(PROJECT_ROOT, "resources", "L2", "data", "merged.json")
    output_dir = os.path.join(PROJECT_ROOT, "resources", "model", "output", "personal_model", LOCAL_MODEL_NAME)
    log_path = os.path.join(PROJECT_ROOT, "logs", "training.log")
    
    # Verify paths exist
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Base model not found: {base_model_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Training data not found: {dataset_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    logger.info(f"Base model: {base_model_path}")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Output dir: {output_dir}")
    
    # --- Training Parameters ---
    learning_rate = 2e-5
    num_epochs = 1
    is_cot = False
    concurrency_threads = 4
    
    # --- Environment Setup ---
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MODEL_BASE_PATH"] = base_model_path
    env["MODEL_PERSONAL_DIR"] = output_dir
    env["PYTHONPATH"] = PROJECT_ROOT
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Faster downloads
    
    if use_cuda:
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["CUDA_LAUNCH_BLOCKING"] = "0"
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    
    if concurrency_threads > 1:
        env["OMP_NUM_THREADS"] = str(concurrency_threads)
        env["MKL_NUM_THREADS"] = str(concurrency_threads)
        env["NUMEXPR_NUM_THREADS"] = str(concurrency_threads)
    
    # Use bf16 only with CUDA
    use_half = "True" if use_cuda else "False"
    
    # --- Build Training Command ---
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "lpm_kernel", "L2", "train.py"),
        "--seed", "42",
        "--model_name_or_path", base_model_path,
        "--user_name", "User",
        "--dataset_name", dataset_path,
        "--chat_template_format", "chatml",
        "--add_special_tokens", "False",
        "--append_concat_token", "False",
        "--max_seq_length", "2048",
        "--num_train_epochs", str(num_epochs),
        "--save_total_limit", "2",
        "--logging_steps", "20",
        "--log_level", "info",
        "--logging_strategy", "steps",
        "--save_strategy", "steps",
        "--save_steps", "5",
        "--push_to_hub", "False",
        "--bf16", use_half,
        "--packing", "False",
        "--learning_rate", str(learning_rate),
        "--lr_scheduler_type", "cosine",
        "--weight_decay", "1e-4",
        "--max_grad_norm", "0.3",
        "--output_dir", output_dir,
        "--per_device_train_batch_size", "4",  # Can use larger batch for smaller model
        "--gradient_accumulation_steps", "2",  # Reduced accumulation steps
        "--gradient_checkpointing", "True",
        "--use_reentrant", "False",
        "--use_peft_lora", "True",
        "--lora_r", "8",  # Appropriate for 0.6B model
        "--lora_alpha", "16",  # Standard alpha for this size
        "--lora_dropout", "0.1",
        "--lora_target_modules", "all-linear",
        "--use_4bit_quantization", "False",
        "--use_nested_quant", "False",
        "--bnb_4bit_compute_dtype", "bfloat16",
        "--is_cot", str(is_cot),
        "--use_cuda", str(use_cuda),
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # --- Run Training ---
    with open(log_path, "ab") as log_file:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        
        # Stream output to both console and log file
        for line in iter(process.stdout.readline, b''):
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
            log_file.write(line)
        
        return_code = process.wait()
    
    if return_code == 0:
        logger.info(f"Training completed successfully! Output saved to: {output_dir}")
    else:
        logger.error(f"Training failed with return code: {return_code}")
        logger.error(f"Check log file: {log_path}")
    
    return return_code

if __name__ == "__main__":
    exit_code = direct_lora_training()
    sys.exit(exit_code)
