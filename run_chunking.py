#!/usr/bin/env python3
"""
Create chunks for documents that don't have any chunks.

This script:
1. Finds all documents without chunks (using direct SQL query)
2. Runs the chunking/splitting process on each
3. Saves the chunks to the database

Usage:
    python run_chunking.py [--limit N]
"""
import sys
import os
import argparse
import sqlite3

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def get_docs_without_chunks():
    """Get document IDs that have no chunks using direct SQL"""
    conn = sqlite3.connect('./data/sqlite/lpm.db')
    cursor = conn.cursor()
    
    # Find documents that don't have any chunks
    cursor.execute("""
        SELECT d.id, d.name 
        FROM document d
        LEFT JOIN chunk c ON d.id = c.document_id
        WHERE c.id IS NULL
        ORDER BY d.id
    """)
    docs = cursor.fetchall()
    conn.close()
    return docs


def main():
    parser = argparse.ArgumentParser(description="Create chunks for documents without chunks")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents to process")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Document Chunking Processor")
    print("=" * 60)
    print()
    
    # Find documents without chunks using direct SQL
    print("Finding documents without chunks (direct SQL query)...")
    docs_without_chunks = get_docs_without_chunks()
    print(f"Found {len(docs_without_chunks)} documents without chunks")
    
    if not docs_without_chunks:
        print("All documents have chunks. Nothing to do!")
        return 0
    
    if args.limit:
        docs_without_chunks = docs_without_chunks[:args.limit]
        print(f"Processing first {args.limit} documents only")
    
    # Create Flask app context
    from flask import Flask
    app = Flask(__name__)
    
    with app.app_context():
        from lpm_kernel.file_data.document_service import document_service
        from lpm_kernel.kernel.chunk_service import ChunkService
        from lpm_kernel.L1.bio import Chunk
        from lpm_kernel.utils import TokenParagraphSplitter
        from lpm_kernel.configs.config import Config
        
        config = Config.from_env()
        
        # Use consistent chunk size with original documents (512 tokens, 50 overlap)
        chunk_size = 512
        chunk_overlap = 50
        print(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
        
        chunker = TokenParagraphSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunk_service = ChunkService()
        
        print()
        
        # Process each document
        success_count = 0
        failed_count = 0
        total_chunks = 0
        
        for i, (doc_id, doc_name) in enumerate(docs_without_chunks):
            print(f"[{i+1}/{len(docs_without_chunks)}] Chunking document {doc_id}...", end=" ")
            
            try:
                # Get document from service
                doc = document_service._repository.find_one(doc_id)
                
                if not doc or not doc.raw_content:
                    print("SKIPPED (no content)")
                    failed_count += 1
                    continue
                
                # Split into text chunks using split_text method
                text_chunks = chunker.split_text(doc.raw_content)
                
                # Create Chunk objects and save each one
                for text in text_chunks:
                    chunk = Chunk(
                        id=0,  # Dummy ID, database will assign real one
                        document_id=doc_id,
                        content=text,
                        embedding=None,
                        tags=None,
                        topic=None
                    )
                    chunk_service.save_chunk(chunk)
                
                total_chunks += len(text_chunks)
                print(f"SUCCESS ({len(text_chunks)} chunks)")
                success_count += 1
                    
            except Exception as e:
                print(f"FAILED ({str(e)[:50]})")
                import traceback
                traceback.print_exc()
                failed_count += 1
                continue
        
        # Summary
        print()
        print("=" * 60)
        print("Summary:")
        print("=" * 60)
        print(f"Documents processed: {success_count + failed_count}")
        print(f"Successful:          {success_count}")
        print(f"Failed/Skipped:      {failed_count}")
        print(f"Total chunks created: {total_chunks}")
        
        if failed_count > 0:
            print(f"\nNote: {failed_count} documents failed or were skipped (likely no content).")
        
        print("\nDone!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
