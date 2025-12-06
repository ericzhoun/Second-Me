"""
Ensure Qwen3 model configs retain their native metadata so downstream tools can
detect and load them accurately.

Usage:
    python scripts/fix_qwen3_config.py --model-name Qwen3-8B
"""

import sys
import os
import json
import argparse
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fix_model_config(model_dir: Path, backup: bool = True) -> bool:
    """Ensure Qwen3 configs advertise the Qwen3 architecture"""
    config_path = model_dir / "config.json"
    
    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        return False
    
    print(f"Reading config from: {config_path}")
    
    # Read config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    original_config = json.loads(json.dumps(config))
    
    original_type = config.get('model_type')
    print(f"  Current model_type: {original_type}")
    
    needs_update = False

    if original_type != 'qwen3':
        print(f"  Updating model_type from '{original_type}' to 'qwen3'")
        config['model_type'] = 'qwen3'
        needs_update = True
    else:
        print("  Model already marked as qwen3")

    if 'architectures' in config:
        original_arch = config['architectures']
        updated_arch = [arch.replace('Qwen2', 'Qwen3') for arch in original_arch]
        if updated_arch != original_arch:
            print(f"  Updating architectures: {original_arch} -> {updated_arch}")
            config['architectures'] = updated_arch
            needs_update = True

    if not needs_update:
        print("✓ Config already aligned with Qwen3 metadata")
        return True

    if backup:
        backup_path = model_dir / "config.json.backup"
        if not backup_path.exists():
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(original_config, f, indent=2, ensure_ascii=False)
            print(f"✓ Backup created: {backup_path}")

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("✓ Saved updated Qwen3 metadata to config.json")
    return True


def restore_model_config(model_dir: Path) -> bool:
    """Restore original config from backup"""
    config_path = model_dir / "config.json"
    backup_path = model_dir / "config.json.backup"
    
    if not backup_path.exists():
        print(f"✗ Backup file not found: {backup_path}")
        return False
    
    print(f"Restoring config from: {backup_path}")
    
    # Read backup
    with open(backup_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Write to config
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Config restored")
    print(f"  model_type: {config.get('model_type')}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fix Qwen3 model config for llama.cpp compatibility"
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        help='Name of the model (e.g., Qwen3-8B)'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        help='Direct path to merged model directory'
    )
    
    parser.add_argument(
        '--restore',
        action='store_true',
        help='Restore original config from backup'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup of original config'
    )
    
    args = parser.parse_args()
    
    # Determine model directory
    if args.model_dir:
        model_dir = Path(args.model_dir)
    elif args.model_name:
        base_dir = Path(PROJECT_ROOT)
        model_dir = base_dir / "resources" / "model" / "output" / "merged_model" / args.model_name
    else:
        print("✗ Error: Must specify either --model-name or --model-dir")
        return 1
    
    if not model_dir.exists():
        print(f"✗ Model directory not found: {model_dir}")
        return 1
    
    print("="*70)
    if args.restore:
        print("RESTORING MODEL CONFIG")
    else:
        print("FIXING MODEL CONFIG FOR LLAMA.CPP")
    print("="*70)
    print()
    
    if args.restore:
        success = restore_model_config(model_dir)
    else:
        success = fix_model_config(model_dir, backup=not args.no_backup)
    
    if success:
        print("\n✓ Done!")
        if not args.restore:
            print("\nYou can now run:")
            print(f"  python scripts/reconvert_model_to_gguf.py --model-name {args.model_name or 'MODEL'}")
        return 0
    else:
        print("\n✗ Failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
