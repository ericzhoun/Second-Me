#!/usr/bin/env python3
"""
Generate embeddings for chunks that don't have embeddings yet.

This script processes chunk embeddings for documents that already have
document-level embeddings (SUCCESS status) but have chunks without embeddings.

Usage:
    python run_chunk_embeddings.py [--limit N]
"""
import sys
import os
import argparse
import sqlite3

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def get_docs_with_unembedded_chunks():
    """Get document IDs that have chunks without embeddings"""
    conn = sqlite3.connect('./data/sqlite/lpm.db')
    cursor = conn.cursor()
    
    # Find documents that have chunks without embeddings
    cursor.execute("""
        SELECT DISTINCT d.id
        FROM document d
        INNER JOIN chunk c ON d.id = c.document_id
        WHERE (c.has_embedding = 0 OR c.has_embedding IS NULL)
        AND d.embedding_status = 'SUCCESS'
        ORDER BY d.id
    """)
    docs = [row[0] for row in cursor.fetchall()]
    conn.close()
    return docs


def main():
    parser = argparse.ArgumentParser(description="Generate chunk embeddings")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents to process")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Chunk Embedding Processor")
    print("=" * 60)
    print()
    
    # Find documents with unembedded chunks
    print("Finding documents with chunks that need embeddings...")
    doc_ids = get_docs_with_unembedded_chunks()
    print(f"Found {len(doc_ids)} documents with unembedded chunks")
    
    if not doc_ids:
        print("All chunks already have embeddings. Nothing to do!")
        return 0
    
    if args.limit:
        doc_ids = doc_ids[:args.limit]
        print(f"Processing first {args.limit} documents only")
    
    # Create Flask app context
    from flask import Flask
    app = Flask(__name__)
    
    with app.app_context():
        from lpm_kernel.file_data.document_service import document_service
        
        print()
        
        # Process each document
        success_count = 0
        failed_count = 0
        
        for i, doc_id in enumerate(doc_ids):
            print(f"[{i+1}/{len(doc_ids)}] Generating chunk embeddings for document {doc_id}...", end=" ")
            
            try:
                # Generate chunk embeddings for this document
                document_service.generate_document_chunk_embeddings(doc_id)
                print("SUCCESS")
                success_count += 1
                    
            except Exception as e:
                print(f"FAILED ({str(e)[:50]})")
                failed_count += 1
                continue
        
        # Summary
        print()
        print("=" * 60)
        print("Summary:")
        print("=" * 60)
        print(f"Documents processed: {success_count + failed_count}")
        print(f"Successful:          {success_count}")
        print(f"Failed:              {failed_count}")
        
        if failed_count > 0:
            print(f"\nWARNING: {failed_count} documents failed.")
            return 1
        
        print("\nAll chunk embeddings generated successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
