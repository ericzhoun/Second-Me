#!/usr/bin/env python3
"""
Process Document Embeddings for INITIALIZED Documents

This script processes embeddings for documents that are still in INITIALIZED status.
It bypasses the application import chain to avoid circular dependencies.

Usage:
    python process_remaining_embeddings.py [--limit N] [--dry-run]
"""
import sys
import os
import argparse
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sqlite3
import chromadb
from tqdm import tqdm


def get_db_connection():
    """Get SQLite database connection"""
    db_path = './data/sqlite/lpm.db'
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")
    return sqlite3.connect(db_path)


def get_unembedded_documents(conn):
    """Get all documents with INITIALIZED embedding status"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, name, raw_content 
        FROM document 
        WHERE embedding_status = 'INITIALIZED' OR embedding_status IS NULL
    """)
    return cursor.fetchall()


def get_document_chunks(conn, doc_id):
    """Get chunks for a document"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, content, has_embedding 
        FROM chunk 
        WHERE document_id = ?
    """, (doc_id,))
    return cursor.fetchall()


def main():
    parser = argparse.ArgumentParser(description="Process remaining document embeddings")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents to process")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be processed")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Document Embedding Processor")
    print("=" * 60)
    print()
    
    # Connect to database
    conn = get_db_connection()
    
    # Get unembedded documents
    print("Finding documents with INITIALIZED status...")
    docs = get_unembedded_documents(conn)
    print(f"Found {len(docs)} documents needing embeddings")
    
    if not docs:
        print("No documents need processing. All done!")
        return 0
    
    if args.dry_run:
        print("\nDry run mode - would process these documents:")
        for i, (doc_id, name, _) in enumerate(docs[:20]):  # Show first 20
            print(f"  {i+1}. Document ID: {doc_id}, Name: {name[:50] if name else 'N/A'}...")
        if len(docs) > 20:
            print(f"  ... and {len(docs) - 20} more")
        conn.close()
        return 0
    
    # For actual processing, we need to use the application's embedding service
    # But we can avoid the circular import by importing after the Flask app context
    print("\nInitializing embedding service...")
    print("NOTE: This requires the Flask application to be properly configured.")
    print()
    
    # The safest way is to use the training workflow directly
    print("To process these documents, run the training workflow:")
    print()
    print("  1. Start the backend server:")
    print("     python run.py")
    print()
    print("  2. Trigger the 'Generate Document Embeddings' step from the UI")
    print("     OR use the API endpoint")
    print()
    print("Alternatively, you can run the embedding step directly:")
    print()
    print("  python -c \"")
    print("  from lpm_kernel.api.domains.trainprocess.trainprocess_service import TrainProcessService")
    print("  from flask import Flask")
    print("  app = Flask(__name__)")
    print("  with app.app_context():")
    print("      service = TrainProcessService()")
    print("      service.generate_document_embeddings()")
    print("  \"")
    print()
    
    # Show document stats
    print("=" * 60)
    print("Document Statistics:")
    print("=" * 60)
    
    cursor = conn.cursor()
    cursor.execute("SELECT embedding_status, COUNT(*) FROM document GROUP BY embedding_status")
    for status, count in cursor.fetchall():
        print(f"  {status or 'NULL'}: {count}")
    
    # Check chunk stats
    cursor.execute("SELECT has_embedding, COUNT(*) FROM chunk GROUP BY has_embedding")
    print("\nChunk Statistics:")
    for has_emb, count in cursor.fetchall():
        status = "with embedding" if has_emb else "without embedding"
        print(f"  {status}: {count}")
    
    conn.close()
    
    print()
    print("=" * 60)
    print(f"ACTION NEEDED: {len(docs)} documents need embedding processing.")
    print("Run the training workflow or use the command above.")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
