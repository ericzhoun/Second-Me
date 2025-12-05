import argparse
import os
import sys
import subprocess

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lpm_kernel.api.domains.trainprocess.trainprocess_service import TrainProcessService
from lpm_kernel.api.domains.trainprocess.process_step import ProcessStep
from lpm_kernel.api.domains.trainprocess.progress_enum import Status
from lpm_kernel.configs.logging import get_train_process_logger

logger = get_train_process_logger()


def main():
    parser = argparse.ArgumentParser(description="Trigger only the base model download stage (model_download)")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen3-8B",
        help="Model identifier to download (used for progress tracking and local folder name)",
    )
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default="Qwen3-8B",
        help="Hugging Face repo id or short name (save_hf_model prefixes 'Qwen/'). If you pass 'org/name', we will strip the org to avoid double prefixing.",
    )
    parser.add_argument(
        "--run-training",
        action="store_true",
        help="After download, run the standard training script (scripts/trigger_training.py)",
    )
    parser.add_argument(
        "--mark-training-complete",
        action="store_true",
        help="After successful download (and optional training), mark training/merge/convert steps as completed in progress file",
    )
    args = parser.parse_args()

    # Force model_name expected by TrainProcessService; save_hf_model uses given id
    # We temporarily set TRAINING_MODEL to match for other components that may read it.
    os.environ["TRAINING_MODEL"] = args.model_name

    # Instantiate service and override model_name to use HF repo (save_hf_model prefixes 'Qwen/') while keeping
    # progress tracking keyed by model_name.
    service = TrainProcessService.get_instance(current_model_name=args.model_name)

    # save_hf_model prepends "Qwen/" to whatever model_name is given, so if the user passed
    # an org/name repo id we strip the org to avoid double prefix (Qwen/Qwen/...).
    download_name = args.hf_model_id.split("/")[-1]
    if "/" in args.hf_model_id:
        logger.warning(
            "Stripping org from hf-model-id to avoid double prefix: %s -> %s", args.hf_model_id, download_name
        )

    # Monkey-patch service.model_name to the stripped download name while progress is still keyed by args.model_name.
    original_name = service.model_name
    service.model_name = download_name
    try:
        success = service.model_download()
    finally:
        service.model_name = original_name

    if success:
        logger.info("Base model download stage completed successfully")
        print("Base model download completed successfully")
        if args.run_training:
            logger.info("Starting training via scripts/trigger_training.py ...")
            train_cmd = [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "trigger_training.py")]
            result = subprocess.run(train_cmd)
            if result.returncode != 0:
                logger.error("Training script failed")
                sys.exit(result.returncode)
        if args.mark_training_complete:
            logger.info("Marking training/merge/convert steps as completed in progress file")
            for step in [ProcessStep.TRAIN, ProcessStep.MERGE_WEIGHTS, ProcessStep.CONVERT_MODEL]:
                service.progress.mark_step_status(step, Status.COMPLETED)
            # Ensure overall status reflects completion and persist
            service.progress.progress.data["status"] = "completed"
            service.progress.progress.data["current_stage"] = None
            service.progress.progress.data["overall_progress"] = 100.0
            # Save the updated progress snapshot
            service.progress._save_progress()
        sys.exit(0)
    else:
        logger.error("Base model download stage failed")
        print("Base model download failed", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
