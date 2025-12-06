"""
Run only data synthesis stages (Biography Generation + Training Data Preparation)
Skips Memory Matrix (document processing) stage
"""
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from lpm_kernel.api.domains.trainprocess.trainprocess_service import TrainProcessService
from lpm_kernel.api.domains.trainprocess.training_params_manager import TrainingParamsManager
from lpm_kernel.configs.logging import get_train_process_logger

logger = get_train_process_logger()

def main():
    """Run data synthesis stages only"""
    
    # Get training parameters
    params_manager = TrainingParamsManager()
    training_params = params_manager.get_latest_training_params()
    model_name = training_params.get("model_name", "Qwen2.5-7B-Instruct")
    
    logger.info(f"Starting data synthesis for model: {model_name}")
    
    # Get service instance
    service = TrainProcessService.get_instance(current_model_name=model_name)
    
    # Stage 2: Generate Biography (uses existing L0 data from notes)
    logger.info("=" * 60)
    logger.info("Stage 2: Synthesize Your Life Narrative")
    logger.info("=" * 60)
    
    """ logger.info("Step 1: Generating biography from notes...")
    if not service.generate_biography():
        logger.error("Biography generation failed!")
        return False
    logger.info("✓ Biography generation completed")
     """
    logger.info("Step 2: Mapping entity network...")
    if not service.map_your_entity_network():
        logger.error("Entity network mapping failed!")
        return False
    logger.info("✓ Entity network mapping completed")
    
    # Stage 3: Prepare Training Data
    logger.info("=" * 60)
    logger.info("Stage 3: Prepare Training Data for Deep Comprehension")
    logger.info("=" * 60)
    
    logger.info("Step 1: Decoding preference patterns...")
    if not service.decode_preference_patterns():
        logger.error("Preference patterns decoding failed!")
        return False
    logger.info("✓ Preference patterns completed")
    
    logger.info("Step 2: Reinforcing identity (generating SelfQA and Diversity data)...")
    if not service.reinforce_identity():
        logger.error("Identity reinforcement failed!")
        return False
    logger.info("✓ Identity reinforcement completed")
    
    logger.info("=" * 60)
    logger.info("Data synthesis completed successfully!")
    logger.info("=" * 60)
    logger.info(f"Training data saved to: resources/L2/data/merged.json")
    logger.info("You can now run training with: python scripts/trigger_training.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
