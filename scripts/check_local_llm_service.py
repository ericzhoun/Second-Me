"""Utility script to sanity-check the local LLM service lifecycle.

This script exercises the `LocalLLMService` helper used by the backend to manage
llama.cpp. It can:

* Inspect the current server status (`--status-only`).
* Start the server with a chosen GGUF file and wait until it is detected.
* Optionally perform a lightweight HTTP health probe once the server reports
  running, mirroring the backend's fallbacks.
* Stop the server and confirm that it terminated.

Example usage (from the project root):

    # Status only
    poetry run python scripts/check_local_llm_service.py --status-only

    # Start the server with an explicit GGUF path and verify
    poetry run python scripts/check_local_llm_service.py \
        --model-path resources/model/output/gguf/Qwen3-8B/model.gguf \
        --start --wait-seconds 15

    # Stop any running server and verify shutdown
    poetry run python scripts/check_local_llm_service.py --stop
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpm_kernel.api.services.local_llm_service import LocalLLMService, local_llm_service
from lpm_kernel.api.domains.kernel2.dto.server_dto import ServerStatus
from lpm_kernel.configs.config import Config


def _format_status(status: ServerStatus) -> str:
    """Return a human-readable summary of the current server status."""
    if not status or not status.is_running:
        return "Server status: NOT RUNNING"

    info = status.process_info
    if not info:
        return "Server status: RUNNING (process info unavailable)"

    cmd_preview = " ".join(info.cmdline)
    return (
        "Server status: RUNNING\n"
        f"  pid: {info.pid}\n"
        f"  cpu%: {info.cpu_percent:.2f}  mem%: {info.memory_percent:.2f}\n"
        f"  started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(info.create_time))}\n"
        f"  cmd: {cmd_preview}"
    )


def _await_status(service: LocalLLMService, want_running: bool, timeout: float) -> ServerStatus:
    """Poll `get_server_status` until it matches the desired state or times out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = service.get_server_status()
        if bool(status.is_running) == want_running:
            return status
        time.sleep(1.0)
    return service.get_server_status()


def _http_probe(config: Config, timeout: float = 2.0) -> Optional[dict]:
    """Attempt a simple health probe against the configured llama.cpp endpoint."""
    base = config.get("LOCAL_LLM_SERVICE_URL")
    if not base:
        return None

    # Try /health first, then root.
    for suffix in ("/health", "/"):
        url = base.rstrip("/") + suffix
        try:
            response = requests.get(url, timeout=timeout)
            return {
                "url": url,
                "status_code": response.status_code,
                "body": response.text[:500],
            }
        except requests.RequestException:
            continue
    return None


def find_default_model_path() -> Optional[Path]:
    """Return the most recent GGUF under resources if available."""
    gguf_root = PROJECT_ROOT / "resources" / "model" / "output" / "gguf"
    if not gguf_root.exists():
        return None

    candidates = sorted(gguf_root.glob("**/model.gguf"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local_llm_service lifecycle")
    parser.add_argument("--model-path", type=Path, help="Path to GGUF model file used when starting")
    parser.add_argument("--start", action="store_true", help="Start llama-server if it is not running")
    parser.add_argument("--stop", action="store_true", help="Stop llama-server and verify shutdown")
    parser.add_argument("--wait-seconds", type=float, default=10.0, help="Seconds to wait for state changes (default: 10)")
    parser.add_argument("--status-only", action="store_true", help="Inspect status and exit without side effects")
    parser.add_argument("--use-gpu", action="store_true", help="Request GPU acceleration when starting")
    args = parser.parse_args()

    service = local_llm_service
    config = Config.from_env()

    status = service.get_server_status()
    print(_format_status(status))

    if args.status_only:
        return 0

    if args.start:
        model_path = args.model_path or find_default_model_path()
        if not model_path:
            print("\nNo model path provided and no GGUF discovered under resources/model/output/gguf.")
            return 1
        if not model_path.exists():
            print(f"\nModel path does not exist: {model_path}")
            return 1

        print(f"\nStarting llama-server with model: {model_path}")
        ok = service.start_server(str(model_path), use_gpu=args.use_gpu)
        if not ok:
            print("Start request returned False – see backend logs for details.")
            return 1

        status = _await_status(service, want_running=True, timeout=args.wait_seconds)
        print(_format_status(status))
        if not status.is_running:
            print("\nServer failed to report running within timeout.")
            return 2

        probe = _http_probe(config)
        if probe:
            print("\nHTTP probe succeeded:")
            print(json.dumps(probe, indent=2))
        else:
            print("\nHTTP probe did not succeed; check network reachability manually.")

    if args.stop:
        print("\nStopping llama-server…")
        status = service.stop_server()
        if status.is_running:
            print("Warning: stop_server reports the process is still running.")
        status = _await_status(service, want_running=False, timeout=args.wait_seconds)
        print(_format_status(status))
        if status.is_running:
            print("\nServer still appears to be running; manual inspection may be required.")
            return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
