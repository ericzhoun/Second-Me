import os
import sys
import requests
import yaml
import logging
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ObsidianSync")

# Constants
ALLOWED_EXTENSIONS = {
    ".txt", ".pdf", ".md",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"
}
STATE_FILE = ".obsidian_sync_state.json"

class ObsidianSync:
    def __init__(self, api_url, vault_path):
        self.api_url = api_url.rstrip("/")
        self.vault_path = Path(vault_path)
        self.state_file_path = self.vault_path / STATE_FILE
        self.state = self._load_state()

        if not self.vault_path.exists():
            raise ValueError(f"Vault path does not exist: {vault_path}")

    def _load_state(self):
        if self.state_file_path.exists():
            try:
                with open(self.state_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load state file: {e}")
        return {}

    def _save_state(self):
        try:
            with open(self.state_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state file: {e}")

    def _get_remote_memories(self):
        """Fetch all memories from the server."""
        try:
            response = requests.get(f"{self.api_url}/api/memories/list")
            response.raise_for_status()
            data = response.json()
            if data.get("code") == 200:
                return data.get("data", [])
            else:
                logger.error(f"API Error: {data.get('message')}")
                return []
        except Exception as e:
            logger.error(f"Failed to list remote memories: {e}")
            return []

    def _parse_frontmatter(self, file_path):
        """Parse frontmatter from markdown file manually to avoid extra dependencies."""
        metadata = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if content.startswith("---\n"):
                end_index = content.find("\n---\n", 4)
                if end_index != -1:
                    frontmatter = content[4:end_index]
                    try:
                        metadata = yaml.safe_load(frontmatter)
                        if not isinstance(metadata, dict):
                            metadata = {}
                    except yaml.YAMLError:
                        pass
        except Exception as e:
            logger.warning(f"Failed to parse frontmatter for {file_path}: {e}")

        return metadata

    def _calculate_file_hash(self, file_path):
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _upload_file(self, file_path, metadata):
        """Upload a file to the server."""
        url = f"{self.api_url}/api/memories/file"

        # Add source metadata
        metadata['source'] = 'obsidian'

        # Convert metadata values to strings/JSON as needed for multipart/form-data
        # Flatten metadata for simple key-value pairs
        form_data = {}
        for k, v in metadata.items():
            if isinstance(v, (dict, list)):
                form_data[k] = json.dumps(v)
            else:
                form_data[k] = str(v)

        try:
            with open(file_path, 'rb') as f:
                files = {'file': (file_path.name, f)}
                response = requests.post(url, files=files, data=form_data)
                response.raise_for_status()
                result = response.json()
                if result.get("code") == 200:
                    logger.info(f"Successfully uploaded: {file_path.name}")
                    return True
                else:
                    logger.error(f"Failed to upload {file_path.name}: {result.get('message')}")
                    return False
        except Exception as e:
            logger.error(f"Error uploading {file_path.name}: {e}")
            return False

    def _delete_remote_file(self, filename):
        """Delete a file from the server."""
        url = f"{self.api_url}/api/memories/file/{filename}"
        try:
            response = requests.delete(url)
            response.raise_for_status()
            result = response.json()
            if result.get("code") == 200:
                logger.info(f"Successfully deleted remote file: {filename}")
                return True
            else:
                # If 404, it's already gone, which is fine
                if "does not exist" in result.get("message", ""):
                    return True
                logger.error(f"Failed to delete {filename}: {result.get('message')}")
                return False
        except Exception as e:
            logger.error(f"Error deleting {filename}: {e}")
            return False

    def sync(self):
        logger.info("Starting sync...")

        # 1. Get remote files
        remote_memories = self._get_remote_memories()

        # Filter remote files that are managed by obsidian sync
        # We assume files with 'source': 'obsidian' in metadata are managed by us.
        remote_obsidian_files = {}
        for m in remote_memories:
            meta = m.get("meta_data") or {}
            # Check if source is obsidian.
            # Note: The server might return meta_data as a dictionary or string depending on DB
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except:
                    meta = {}

            if meta.get("source") == "obsidian":
                remote_obsidian_files[m["name"]] = m

        # 2. Scan local files
        local_files = {}
        for root, _, files in os.walk(self.vault_path):
            for file in files:
                if file == STATE_FILE:
                    continue

                file_path = Path(root) / file
                if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                    continue

                # We use filename as the key because the backend uses filename.
                # WARNING: Duplicate filenames in different folders will cause conflict in the current backend.
                # The backend flattens the directory structure.
                # We will just warn for now or process the first one found.
                if file in local_files:
                    logger.warning(f"Duplicate filename found: {file}. Backend does not support folders. Skipping {file_path}")
                    continue

                local_files[file] = file_path

        # 3. Process Uploads and Updates
        for filename, file_path in local_files.items():
            try:
                # Calculate hash and size
                current_hash = self._calculate_file_hash(file_path)
                current_size = file_path.stat().st_size

                # Extract metadata
                metadata = {}
                if file_path.suffix.lower() == ".md":
                    metadata = self._parse_frontmatter(file_path)

                    # Add creation date if not present
                    if 'created' not in metadata:
                        created_ts = file_path.stat().st_ctime
                        metadata['created'] = datetime.fromtimestamp(created_ts).isoformat()

                # Check against state
                state_entry = self.state.get(filename)
                needs_upload = False

                if filename not in remote_obsidian_files:
                    logger.info(f"New file found: {filename}")
                    needs_upload = True
                else:
                    # File exists remotely. Check if it needs update.
                    # We check if hash changed or if state says it's different
                    if not state_entry:
                        # No local state, but exists remotely. Trust remote?
                        # Or check if remote size matches.
                        # Ideally we assume if we don't have state, we might need to sync.
                        # But to save bandwidth, we can check remote size.
                        remote_size = remote_obsidian_files[filename].get("size")
                        if remote_size != current_size:
                            logger.info(f"Size mismatch for {filename}. Local: {current_size}, Remote: {remote_size}")
                            needs_upload = True
                        else:
                            # Assume synced if size matches and we have no state (initial sync of existing folder)
                            # Update state
                            self.state[filename] = {"hash": current_hash, "size": current_size}
                    else:
                        if state_entry.get("hash") != current_hash:
                            logger.info(f"File changed: {filename}")
                            needs_upload = True

                if needs_upload:
                    # If it exists remotely, delete it first (to update)
                    if filename in remote_obsidian_files:
                        self._delete_remote_file(filename)

                    if self._upload_file(file_path, metadata):
                        self.state[filename] = {"hash": current_hash, "size": current_size}
                        self._save_state()

            except Exception as e:
                logger.error(f"Error processing local file {filename}: {e}")

        # 4. Process Deletions
        for filename in remote_obsidian_files:
            if filename not in local_files:
                logger.info(f"File deleted locally, removing from server: {filename}")
                if self._delete_remote_file(filename):
                    if filename in self.state:
                        del self.state[filename]
                        self._save_state()

        logger.info("Sync completed.")

def main():
    parser = argparse.ArgumentParser(description="Sync Obsidian vault to LPM Memories")
    parser.add_argument("--api-url", help="LPM API URL (e.g. http://localhost:5000)")
    parser.add_argument("--vault-path", help="Path to Obsidian vault")

    args = parser.parse_args()

    api_url = args.api_url or os.environ.get("LPM_API_URL")
    vault_path = args.vault_path or os.environ.get("OBSIDIAN_VAULT_PATH")

    if not api_url:
        logger.error("API URL must be provided via --api-url or LPM_API_URL env var")
        sys.exit(1)

    if not vault_path:
        logger.error("Vault path must be provided via --vault-path or OBSIDIAN_VAULT_PATH env var")
        sys.exit(1)

    try:
        syncer = ObsidianSync(api_url, vault_path)
        syncer.sync()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
