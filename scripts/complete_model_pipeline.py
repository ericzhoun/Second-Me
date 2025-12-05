"""
Complete the model pipeline: merge LoRA weights and convert to GGUF.
Use this after training to prepare the model for inference.
"""
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from lpm_kernel.api.domains.trainprocess.trainprocess_service import TrainProcessService
from lpm_kernel.configs.logging import get_train_process_logger

logger = get_train_process_logger()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Complete model pipeline: merge + convert to GGUF")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen3-8B",
        help="Model name (must match training output folder)",
    )
    args = parser.parse_args()

    logger.info(f"Starting post-training pipeline for model: {args.model_name}")
    
    # Get service instance
    service = TrainProcessService.get_instance(current_model_name=args.model_name)
    
    # Step 1: Merge LoRA weights with base model
    logger.info("Step 1/2: Merging LoRA weights with base model...")
    if not service.merge_weights():
        logger.error("Failed to merge weights")
        sys.exit(1)
    logger.info("✓ Weight merge completed")
    
    # Step 2: Convert merged model to GGUF
    logger.info("Step 2/2: Converting merged model to GGUF format...")
    if not service.convert_model():
        logger.error("Failed to convert model to GGUF")
        sys.exit(1)
    logger.info("✓ GGUF conversion completed")
    
    logger.info("=" * 60)
    logger.info(f"Model pipeline complete for {args.model_name}!")
    logger.info(f"GGUF model location: resources/model/output/gguf/{args.model_name}/model.gguf")
    logger.info("You can now start the inference service.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
