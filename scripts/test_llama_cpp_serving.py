"""
Test script to serve and interact with the personal baseline model using llama.cpp.

This script:
1. Locates the GGUF model from the training pipeline
2. Starts llama-server with the model
3. Tests the model with sample prompts
4. Displays performance metrics

Usage:
    python scripts/test_llama_cpp_serving.py [--model-name MODEL_NAME] [--no-gpu] [--port PORT]
"""

import sys
import os
import argparse
import subprocess
import time
import requests
import json
import signal
import threading
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Try to import torch for GPU detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. GPU detection disabled.")


class LlamaCppTester:
    """Test harness for llama.cpp server"""
    
    def __init__(self, model_path: str, port: int = 8080, use_gpu: bool = True):
        self.model_path = model_path
        self.port = port
        self.use_gpu = use_gpu
        self.server_process = None
        self.base_url = f"http://127.0.0.1:{port}"
        self.server_output = []
        self.server_ready = False
        
    def find_llama_server(self) -> str:
        """Find llama-server executable"""
        base_dir = Path(PROJECT_ROOT)
        
        # Common paths to check
        paths_to_check = [
            base_dir / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe",
            base_dir / "llama.cpp" / "build" / "bin" / "Release" / "llama-server",
            base_dir / "llama.cpp" / "build" / "bin" / "llama-server.exe",
            base_dir / "llama.cpp" / "build" / "bin" / "llama-server",
        ]
        
        for path in paths_to_check:
            if path.exists():
                print(f"✓ Found llama-server at: {path}")
                return str(path)
        
        raise FileNotFoundError(
            "llama-server executable not found. Please build llama.cpp first:\n"
            "  cd llama.cpp && mkdir build && cd build && cmake .. && cmake --build . --config Release"
        )
    
    def check_gpu_availability(self):
        """Check if GPU is available"""
        if not TORCH_AVAILABLE:
            return False, "PyTorch not available"
        
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return True, f"{gpu_name} ({gpu_memory:.1f} GB)"
    
    def stream_output(self, pipe, prefix):
        """Stream output from subprocess in a separate thread"""
        try:
            for line in iter(pipe.readline, ''):
                if line:
                    self.server_output.append(f"{prefix}: {line.rstrip()}")
                    # Check for ready signal
                    if "HTTP server is listening" in line or "server is listening" in line:
                        print(f"  Server binding to port {self.port}...")
                    elif "loading model" in line.lower():
                        print(f"  Loading model into memory...")
                    elif "model loaded" in line.lower() or "main: server started" in line.lower():
                        self.server_ready = True
                        print(f"  Model loaded successfully!")
        except Exception as e:
            self.server_output.append(f"{prefix} ERROR: {str(e)}")
    
    def start_server(self) -> bool:
        """Start llama-server"""
        try:
            server_path = self.find_llama_server()
            
            # Check GPU
            gpu_available, gpu_info = self.check_gpu_availability()
            
            if self.use_gpu and gpu_available:
                print(f"✓ GPU detected: {gpu_info}")
                print("  Starting server with GPU acceleration...")
            elif self.use_gpu and not gpu_available:
                print(f"✗ GPU requested but not available: {gpu_info}")
                print("  Falling back to CPU mode...")
                self.use_gpu = False
            else:
                print("  Starting server in CPU mode...")
            
            # Build command with optimized parameters for 8B model
            cmd = [
                server_path,
                "-m", self.model_path,
                "--host", "0.0.0.0",
                "--port", str(self.port),
                "--ctx-size", "2048",
                "--parallel", "1",           # Reduce parallelism for stability
                "--threads", "8",            # Limit threads
                "--n-predict", "512",        # Limit max prediction tokens
            ]
            
            # Add GPU parameters if available
            if self.use_gpu and gpu_available:
                cmd.extend([
                    "--n-gpu-layers", "19",  # Use all layers on GPU
                    "--tensor-split", "0",
                    "--main-gpu", "0"
                ])
            
            print(f"\nStarting server with command:")
            print(f"  {' '.join(cmd)}")
            print("\nWaiting for server to start (this may take 30-60 seconds for large models)...")
            
            # Start server process with separate stdout/stderr handling
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Start threads to capture output
            stdout_thread = threading.Thread(
                target=self.stream_output,
                args=(self.server_process.stdout, "STDOUT"),
                daemon=True
            )
            stderr_thread = threading.Thread(
                target=self.stream_output,
                args=(self.server_process.stderr, "STDERR"),
                daemon=True
            )
            
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for server to be ready
            max_wait = 120  # seconds - increase for large models
            start_time = time.time()
            dots_printed = 0
            
            while time.time() - start_time < max_wait:
                # Check if process died
                if self.server_process.poll() is not None:
                    # Give threads a moment to capture final output
                    time.sleep(1)
                    
                    print(f"\n\n✗ Server process died!")
                    print("\nServer output (last 50 lines):")
                    print("-" * 70)
                    for line in self.server_output[-50:]:
                        print(line)
                    print("-" * 70)
                    
                    # Check for common error patterns
                    output_str = "\n".join(self.server_output)
                    if "out of memory" in output_str.lower() or "oom" in output_str.lower():
                        print("\n⚠ Possible cause: Insufficient memory")
                        print("  Try reducing context size with: --ctx-size 1024")
                    elif "cannot allocate" in output_str.lower():
                        print("\n⚠ Possible cause: Memory allocation failed")
                        print("  The 8B model (~15GB) may be too large for your system")
                    elif "failed to load" in output_str.lower():
                        print("\n⚠ Possible cause: Model file may be corrupted")
                    
                    return False
                
                # Try to connect to check if server is ready
                try:
                    response = requests.get(f"{self.base_url}/health", timeout=2)
                    if response.status_code == 200:
                        elapsed = time.time() - start_time
                        print(f"\n✓ Server is ready! (took {elapsed:.1f}s)")
                        return True
                except requests.exceptions.RequestException:
                    pass
                
                # Print progress indicator
                if dots_printed < 60:
                    print(".", end="", flush=True)
                    dots_printed += 1
                
                time.sleep(2)
            
            print(f"\n✗ Server failed to start within {max_wait} seconds")
            print("\nServer output (last 50 lines):")
            print("-" * 70)
            for line in self.server_output[-50:]:
                print(line)
            print("-" * 70)
            return False
            
        except Exception as e:
            print(f"✗ Error starting server: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def stop_server(self):
        """Stop llama-server"""
        if self.server_process:
            print("\nStopping server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("  Server didn't stop gracefully, forcing...")
                self.server_process.kill()
            print("✓ Server stopped")
    
    def test_completion(self, prompt: str, max_tokens: int = 150) -> dict:
        """Test completion endpoint"""
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}", "details": response.text}
                
        except Exception as e:
            return {"error": str(e)}
    
    def run_tests(self):
        """Run a series of test prompts"""
        print("\n" + "="*70)
        print("TESTING PERSONAL BASELINE MODEL")
        print("="*70)
        
        test_prompts = [
            {
                "name": "Simple Greeting",
                "prompt": "Hello! Who are you?",
                "max_tokens": 100
            },
            {
                "name": "Personal Information",
                "prompt": "Tell me about yourself and your background.",
                "max_tokens": 200
            },
            {
                "name": "Preference Query",
                "prompt": "What are your interests and hobbies?",
                "max_tokens": 150
            }
        ]
        
        results = []
        
        for i, test in enumerate(test_prompts, 1):
            print(f"\n{'─'*70}")
            print(f"Test {i}/{len(test_prompts)}: {test['name']}")
            print(f"{'─'*70}")
            print(f"Prompt: {test['prompt']}")
            print("\nGenerating response...", end="", flush=True)
            
            start_time = time.time()
            result = self.test_completion(test['prompt'], test['max_tokens'])
            elapsed = time.time() - start_time
            
            print(f" done ({elapsed:.2f}s)")
            
            if 'error' in result:
                print(f"\n✗ Error: {result['error']}")
                if 'details' in result:
                    print(f"   Details: {result['details']}")
                results.append({
                    "test": test['name'],
                    "success": False,
                    "error": result['error']
                })
            else:
                # Extract response
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    print(f"\nResponse:\n{content}")
                    
                    # Display stats
                    if 'usage' in result:
                        usage = result['usage']
                        tokens_per_sec = usage.get('completion_tokens', 0) / elapsed if elapsed > 0 else 0
                        print(f"\nStats:")
                        print(f"  • Prompt tokens: {usage.get('prompt_tokens', 0)}")
                        print(f"  • Completion tokens: {usage.get('completion_tokens', 0)}")
                        print(f"  • Total tokens: {usage.get('total_tokens', 0)}")
                        print(f"  • Speed: {tokens_per_sec:.2f} tokens/sec")
                        print(f"  • Time: {elapsed:.2f}s")
                    
                    results.append({
                        "test": test['name'],
                        "success": True,
                        "time": elapsed,
                        "response_length": len(content)
                    })
                else:
                    print(f"\n✗ Unexpected response format: {result}")
                    results.append({
                        "test": test['name'],
                        "success": False,
                        "error": "Unexpected response format"
                    })
        
        # Summary
        print(f"\n{'='*70}")
        print("TEST SUMMARY")
        print(f"{'='*70}")
        successful = sum(1 for r in results if r['success'])
        print(f"Tests passed: {successful}/{len(results)}")
        
        if successful > 0:
            avg_time = sum(r['time'] for r in results if r['success']) / successful
            print(f"Average response time: {avg_time:.2f}s")
        
        return results


def find_model_path(model_name: str = None) -> str:
    """Find the GGUF model path"""
    base_dir = Path(PROJECT_ROOT)
    gguf_dir = base_dir / "resources" / "model" / "output" / "gguf"
    
    if not gguf_dir.exists():
        raise FileNotFoundError(
            f"GGUF directory not found: {gguf_dir}\n"
            "Please complete the training pipeline first."
        )
    
    # If model_name provided, look for that specific model
    if model_name:
        model_path = gguf_dir / model_name / "model.gguf"
        if model_path.exists():
            return str(model_path)
        else:
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Available models in {gguf_dir}:\n" +
                "\n".join(f"  • {d.name}" for d in gguf_dir.iterdir() if d.is_dir())
            )
    
    # Otherwise, find the most recent model
    model_dirs = [d for d in gguf_dir.iterdir() if d.is_dir()]
    
    if not model_dirs:
        raise FileNotFoundError(
            f"No models found in {gguf_dir}\n"
            "Please complete the training pipeline first."
        )
    
    # Find most recent model by checking modification time
    latest_model = max(model_dirs, key=lambda d: d.stat().st_mtime)
    model_path = latest_model / "model.gguf"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"model.gguf not found in {latest_model}\n"
            "Please complete the model conversion step."
        )
    
    return str(model_path)


def main():
    parser = argparse.ArgumentParser(
        description="Test serving personal baseline model with llama.cpp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with auto-detected model
  python scripts/test_llama_cpp_serving.py
  
  # Test specific model
    python scripts/test_llama_cpp_serving.py --model-name Qwen2.5-7B-Instruct
  
  # Test with CPU only
  python scripts/test_llama_cpp_serving.py --no-gpu
  
  # Use custom port
  python scripts/test_llama_cpp_serving.py --port 8081
        """
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        help='Name of the model to test (e.g., Qwen2.5-7B-Instruct). If not provided, uses most recent model.'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        help='Direct path to GGUF model file. Overrides --model-name.'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Force CPU mode (disable GPU acceleration)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port for llama-server (default: 8080)'
    )
    
    parser.add_argument(
        '--no-tests',
        action='store_true',
        help='Start server only, skip running test prompts'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("LLAMA.CPP PERSONAL MODEL SERVING TEST")
    print("="*70)
    
    # Find model
    try:
        if args.model_path:
            model_path = args.model_path
            print(f"\nUsing provided model path: {model_path}")
        else:
            print(f"\nSearching for model...")
            model_path = find_model_path(args.model_name)
            print(f"✓ Found model: {model_path}")
        
        # Check model file size
        model_size = Path(model_path).stat().st_size / (1024**3)  # GB
        print(f"  Model size: {model_size:.2f} GB")
        
        # Warn about large models
        if model_size > 10:
            print(f"\n⚠ WARNING: Large model detected ({model_size:.1f} GB)")
            print(f"  This may require significant RAM (typically 2x model size)")
            print(f"  Estimated RAM needed: ~{model_size * 2:.0f} GB")
            
            # Check available memory on Windows
            try:
                import psutil
                available_ram = psutil.virtual_memory().available / (1024**3)
                total_ram = psutil.virtual_memory().total / (1024**3)
                print(f"  System RAM: {available_ram:.1f} GB available / {total_ram:.1f} GB total")
                
                if available_ram < model_size * 1.5:
                    print(f"\n  ⚠ You may not have enough RAM to run this model!")
                    print(f"  Consider using a smaller model or quantizing further.")
            except ImportError:
                pass
        
    except FileNotFoundError as e:
        print(f"\n✗ {e}")
        return 1
    
    # Create tester
    tester = LlamaCppTester(
        model_path=model_path,
        port=args.port,
        use_gpu=not args.no_gpu
    )
    
    # Setup signal handler for clean shutdown
    def signal_handler(sig, frame):
        print("\n\nReceived interrupt signal...")
        tester.stop_server()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    # Start server
    try:
        if not tester.start_server():
            print("\n✗ Failed to start server")
            return 1
        
        if args.no_tests:
            print("\n" + "="*70)
            print("Server is running!")
            print("="*70)
            print(f"\nAPI endpoint: {tester.base_url}/v1/chat/completions")
            print("\nPress Ctrl+C to stop the server...")
            
            # Keep server running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        else:
            # Run tests
            results = tester.run_tests()
            
            # Check if all tests passed
            if all(r['success'] for r in results):
                print("\n✓ All tests passed!")
                return 0
            else:
                print("\n✗ Some tests failed")
                return 1
    
    finally:
        tester.stop_server()


if __name__ == "__main__":
    sys.exit(main())