#!/usr/bin/env python3
"""
Process Document Embeddings for INITIALIZED Documents

This script processes embeddings for documents that are still in INITIALIZED status.
It uses the Flask application context to properly initialize all dependencies.

Usage:
    python run_embeddings.py [--limit N]
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Process remaining document embeddings")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents to process")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Document Embedding Processor")
    print("=" * 60)
    print()
    
    # Create Flask app context to avoid circular imports
    from flask import Flask
    app = Flask(__name__)
    
    with app.app_context():
        # Now we can safely import
        from lpm_kernel.file_data.document_service import document_service
        from lpm_kernel.file_data.process_status import ProcessStatus
        
        # Get unembedded documents
        print("Finding documents with INITIALIZED status...")
        docs = document_service._repository.find_unembedding()
        print(f"Found {len(docs)} documents needing embeddings")
        
        if not docs:
            print("No documents need processing. All done!")
            return 0
        
        if args.limit:
            docs = docs[:args.limit]
            print(f"Processing first {args.limit} documents only")
        
        print()
        
        # Process each document
        success_count = 0
        failed_count = 0
        
        for i, doc in enumerate(docs):
            doc_id = doc.id
            print(f"[{i+1}/{len(docs)}] Processing document {doc_id}...", end=" ")
            
            try:
                # Generate document embedding
                embedding = document_service.process_document_embedding(doc_id)
                
                if embedding is not None:
                    # Generate chunk embeddings
                    document_service.generate_document_chunk_embeddings(doc_id)
                    print("SUCCESS")
                    success_count += 1
                else:
                    print("FAILED (no embedding)")
                    failed_count += 1
                    
            except Exception as e:
                print(f"FAILED ({str(e)[:50]})")
                failed_count += 1
                continue
        
        # Summary
        print()
        print("=" * 60)
        print("Summary:")
        print("=" * 60)
        print(f"Total processed: {success_count + failed_count}")
        print(f"Successful:      {success_count}")
        print(f"Failed:          {failed_count}")
        
        if failed_count > 0:
            print(f"\nWARNING: {failed_count} documents failed.")
            return 1
        
        print("\nAll documents processed successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
