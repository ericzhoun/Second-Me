#!/usr/bin/env python3
"""
Script to reset ChromaDB collections with the correct dimension for the current embedding model.
Run this when you change embedding models and need to reset the collections.
"""

import os
import sys
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    chroma_path = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
    
    # Get absolute path
    if not os.path.isabs(chroma_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        chroma_path = os.path.join(project_root, chroma_path)
    
    print(f"ChromaDB path: {chroma_path}")
    
    if os.path.exists(chroma_path):
        # Create backup
        backup_path = chroma_path + "_backup_reset"
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        shutil.copytree(chroma_path, backup_path)
        print(f"Created backup at: {backup_path}")
        
        # Delete the original
        shutil.rmtree(chroma_path)
        print(f"Deleted ChromaDB at: {chroma_path}")
        print("ChromaDB will be recreated with the correct dimension when you run the embedding step.")
    else:
        print(f"ChromaDB does not exist at: {chroma_path}")
        print("Nothing to reset.")

if __name__ == "__main__":
    main()
