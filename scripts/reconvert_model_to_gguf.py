"""
Script to reconvert a merged model to GGUF format with better error handling.

This script:
1. Validates the merged model
2. Uses the llama.cpp convert script directly
3. Provides detailed error diagnostics

Usage:
    python scripts/reconvert_model_to_gguf.py --model-name Qwen2.5-7B-Instruct
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def validate_merged_model(model_dir: Path) -> bool:
    """Validate that the merged model has all required files"""
    required_files = [
        "config.json",
        "tokenizer_config.json",
    ]
    
    print(f"\nValidating merged model at: {model_dir}")
    
    # Check if directory exists
    if not model_dir.exists():
        print(f"[X] Model directory does not exist: {model_dir}")
        return False
    
    # Check for required files
    missing_files = []
    for file in required_files:
        file_path = model_dir / file
        if not file_path.exists():
            missing_files.append(file)
        else:
            print(f"  [OK] Found {file}")
    
    if missing_files:
        print(f"[X] Missing required files: {', '.join(missing_files)}")
        return False
    
    # Check for model weights (safetensors or bin files)
    safetensors_files = list(model_dir.glob("*.safetensors"))
    bin_files = list(model_dir.glob("*.bin"))
    
    if not safetensors_files and not bin_files:
        print(f"[X] No model weight files found (.safetensors or .bin)")
        return False
    
    if safetensors_files:
        print(f"  [OK] Found {len(safetensors_files)} safetensors file(s)")
    if bin_files:
        print(f"  [OK] Found {len(bin_files)} bin file(s)")
    
    # Read config to check model type
    import json
    try:
        with open(model_dir / "config.json", "r") as f:
            config = json.load(f)
            model_type = config.get("model_type", "unknown")
            print(f"  [OK] Model type: {model_type}")
            
            if model_type != "qwen2":
                print(f"  [!] Warning: Expected model_type 'qwen2', got '{model_type}'")
    except Exception as e:
        print(f"  [!] Warning: Could not read config.json: {e}")
    
    print("[OK] Model validation passed")
    return True


def find_converter_script() -> Path:
    """Locate the best available convert_hf_to_gguf.py script."""
    base_dir = Path(PROJECT_ROOT)

    preferred = base_dir / "lpm_kernel" / "L2" / "convert_hf_to_gguf.py"
    if preferred.exists():
        print(f"[OK] Using enhanced converter at: {preferred}")
        return preferred

    # Fall back to llama.cpp copy if the local helper is missing
    fallback_paths = [
        base_dir / "llama.cpp" / "convert_hf_to_gguf.py",
        base_dir / "llama.cpp" / "convert-hf-to-gguf.py",
    ]

    for path in fallback_paths:
        if path.exists():
            print(f"[OK] Using llama.cpp converter at: {path}")
            return path

    checked = [preferred] + fallback_paths
    raise FileNotFoundError(
        "Conversion script not found. Checked the following paths:\n" +
        "\n".join(f"  - {p}" for p in checked)
    )


def convert_to_gguf(merged_model_dir: Path, output_path: Path, quantization: str = "f16") -> bool:
    """Convert merged model to GGUF format using llama.cpp converter"""
    try:
        # Find converter script (prefer the repo's patched copy)
        converter_script = find_converter_script()
        
        print(f"\nConverting model to GGUF format...")
        print(f"  Input: {merged_model_dir}")
        print(f"  Output: {output_path}")
        print(f"  Quantization: {quantization}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build command - use llama.cpp's converter directly
        cmd = [
            sys.executable,
            str(converter_script),
            str(merged_model_dir),
            "--outfile", str(output_path),
            "--outtype", quantization,
        ]
        
        print(f"\nRunning conversion command:")
        print(f"  {' '.join(cmd)}")
        print("\nThis may take several minutes...\n")
        
        # Run conversion with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',  # Replace invalid characters instead of failing
            bufsize=1
        )
        
        # Stream output
        try:
            for line in process.stdout:
                print(f"  {line.rstrip()}")
        except UnicodeDecodeError:
            # If encoding fails, read as bytes and decode manually
            pass
        
        # Wait for completion
        return_code = process.wait()
        
        if return_code != 0:
            print(f"\n[X] Conversion failed with exit code {return_code}")
            return False
        
        # Verify output file exists
        if not output_path.exists():
            print(f"\n[X] Output file was not created: {output_path}")
            return False
        
        file_size = output_path.stat().st_size / (1024**3)  # GB
        print(f"\n[OK] Conversion completed successfully!")
        print(f"  Output file size: {file_size:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"\n[X] Conversion error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Reconvert merged model to GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reconvert with default settings (f16)
    python scripts/reconvert_model_to_gguf.py --model-name Qwen2.5-7B-Instruct
  
  # Reconvert with quantization
    python scripts/reconvert_model_to_gguf.py --model-name Qwen2.5-7B-Instruct --quantization q4_0
  
  # Use custom paths
  python scripts/reconvert_model_to_gguf.py --input-dir path/to/merged --output-file path/to/model.gguf
        """
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        help='Name of the model (e.g., Qwen2.5-7B-Instruct)'
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        help='Direct path to merged model directory. Overrides --model-name.'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        help='Direct path to output GGUF file. Overrides default location.'
    )
    
    parser.add_argument(
        '--quantization',
        type=str,
        default='f16',
        choices=['f32', 'f16', 'q8_0', 'q4_0', 'q4_1', 'q5_0', 'q5_1'],
        help='Quantization type (default: f16)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("MODEL TO GGUF RECONVERSION")
    print("="*70)
    
    # Determine input directory
    if args.input_dir:
        merged_model_dir = Path(args.input_dir)
    elif args.model_name:
        base_dir = Path(PROJECT_ROOT)
        merged_model_dir = base_dir / "resources" / "model" / "output" / "merged_model" / args.model_name
    else:
        print("[X] Error: Must specify either --model-name or --input-dir")
        return 1
    
    # Determine output path
    if args.output_file:
        output_path = Path(args.output_file)
    elif args.model_name:
        base_dir = Path(PROJECT_ROOT)
        output_path = base_dir / "resources" / "model" / "output" / "gguf" / args.model_name / "model.gguf"
    else:
        output_path = Path("model.gguf")
    
    # Validate merged model
    if not validate_merged_model(merged_model_dir):
        print("\n[X] Model validation failed. Cannot proceed with conversion.")
        return 1
    
    # Check if we can locate a converter script
    try:
        find_converter_script()
    except FileNotFoundError as e:
        print(f"\n[X] {e}")
        print("\nPlease ensure the repository (or llama.cpp submodule) is fully initialized:")
        print("  git submodule update --init --recursive")
        return 1
    
    # Perform conversion
    if convert_to_gguf(merged_model_dir, output_path, args.quantization):
        print("\n" + "="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"\nYour GGUF model is ready at:")
        print(f"  {output_path}")
        print(f"\nYou can now test it with:")
        print(f"  python scripts/test_llama_cpp_serving.py --model-path \"{output_path}\"")
        return 0
    else:
        print("\n" + "="*70)
        print("CONVERSION FAILED")
        print("="*70)
        print("\nTroubleshooting steps:")
        print("1. Check if the merged model is valid")
        print("2. Ensure llama.cpp is up to date:")
        print("   cd llama.cpp && git pull")
        print("3. Try with a different quantization:")
        print("   python scripts/reconvert_model_to_gguf.py --model-name Qwen2.5-7B-Instruct --quantization f32")
        return 1


if __name__ == "__main__":
    sys.exit(main())
