"""
Debug script to run reinforce_identity step directly without running all previous steps.
Run this from the project root directory:
    python debug_reinforce_identity.py
"""

import os
import sys
import traceback

# Add the project root to the path
sys.path.insert(0, os.getcwd())

# Set environment variables before imports
os.environ["CONCURRENCY_THREADS"] = "2"
os.environ["DATA_SYNTHESIS_MODE"] = "low"

from lpm_kernel.api.domains.trainprocess.trainprocess_service import TrainProcessService
from lpm_kernel.api.domains.trainprocess.training_params_manager import TrainingParamsManager

def debug_reinforce_identity():
    """Debug the reinforce_identity step"""
    
    print("=" * 60)
    print("Debugging reinforce_identity step")
    print("=" * 60)
    
    # Check environment variables
    print(f"\nEnvironment variables:")
    print(f"  CONCURRENCY_THREADS = {os.environ.get('CONCURRENCY_THREADS', 'not set')}")
    print(f"  CONCURRENCY_THREADS type = {type(os.environ.get('CONCURRENCY_THREADS'))}")
    print(f"  DATA_SYNTHESIS_MODE = {os.environ.get('DATA_SYNTHESIS_MODE', 'not set')}")
    
    # Get training params
    try:
        params_manager = TrainingParamsManager()
        training_params = params_manager.get_latest_training_params()
        print(f"\nTraining params:")
        for key, value in training_params.items():
            print(f"  {key} = {value} (type: {type(value).__name__})")
    except Exception as e:
        print(f"Error getting training params: {e}")
        traceback.print_exc()
    
    # Try to get or create the service instance
    try:
        # Use a default model name - adjust this if needed
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        print(f"\nCreating TrainProcessService with model: {model_name}")
        
        service = TrainProcessService.get_instance(model_name)
        if service is None:
            print("Failed to get service instance")
            return
        
        print("Service instance created successfully")
        
        # Try to prepare L2 data first
        print("\n" + "=" * 60)
        print("Preparing L2 data...")
        print("=" * 60)
        
        try:
            l2_data = service._prepare_l2_data()
            print("L2 data prepared successfully")
            print(f"  notes count: {len(l2_data.get('notes', [])) if l2_data.get('notes') else 'None'}")
            print(f"  basic_info: {l2_data.get('basic_info', {}).keys() if l2_data.get('basic_info') else 'None'}")
        except Exception as e:
            print(f"Error preparing L2 data: {e}")
            traceback.print_exc()
            return
        
        # Now try the reinforce_identity step
        print("\n" + "=" * 60)
        print("Running reinforce_identity...")
        print("=" * 60)
        
        try:
            result = service.reinforce_identity()
            print(f"\nResult: {result}")
        except Exception as e:
            print(f"\nError in reinforce_identity: {e}")
            traceback.print_exc()
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_reinforce_identity()
