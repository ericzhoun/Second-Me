#!/usr/bin/env python3
"""
Repair Chunk Embeddings Database

This script syncs the chunk has_embedding status from ChromaDB to the database.
Use this to fix the issue where 80%+ of chunks show as "missing embeddings"
even though they actually exist in ChromaDB.

Usage:
    python repair_chunk_embeddings.py
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Direct imports to avoid circular dependency through api package
import chromadb
from sqlalchemy import create_engine, Column, BigInteger, Boolean, Text, String, DateTime, JSON, ForeignKey, Integer
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

def main():
    """Run the chunk embedding sync repair"""
    try:
        print("=" * 60)
        print("Chunk Embedding Database Repair Tool")
        print("=" * 60)
        print()
        print("This will sync the database has_embedding field with ChromaDB.")
        print("Checking all chunks...")
        print()
        
        # Initialize ChromaDB client
        chroma_path = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
        chroma_client = chromadb.PersistentClient(path=chroma_path)
        
        try:
            chunk_collection = chroma_client.get_collection(name="document_chunks")
            print(f"Found ChromaDB collection 'document_chunks' with {chunk_collection.count()} entries")
        except Exception as e:
            print(f"Error accessing ChromaDB collection: {e}")
            print("Make sure the ChromaDB database exists at:", chroma_path)
            return 1
        
        # Initialize SQLite database connection
        db_path = os.getenv("DATABASE_URL", "sqlite:///./data/sqlite/lpm.db")
        if db_path.startswith("sqlite:///"):
            # Convert relative path to absolute
            db_file = db_path.replace("sqlite:///", "")
            if not os.path.isabs(db_file):
                db_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), db_file)
            db_path = f"sqlite:///{db_file}"
        
        print(f"Connecting to database: {db_path}")
        engine = create_engine(db_path)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Query all chunks from database
        from sqlalchemy import text
        result = session.execute(text("SELECT id, has_embedding FROM chunk"))
        chunks = result.fetchall()
        
        print(f"Found {len(chunks)} chunks in database")
        print()
        
        stats = {
            "total_chunks": len(chunks),
            "synced_to_true": 0,
            "synced_to_false": 0,
            "already_correct": 0,
            "errors": 0
        }
        
        for chunk_id, db_has_embedding in chunks:
            try:
                # Check if embedding exists in ChromaDB
                result = chunk_collection.get(
                    ids=[str(chunk_id)], 
                    include=["embeddings"]
                )
                
                # Check if embeddings exist
                embeddings = result.get("embeddings") if result else None
                has_embedding_in_chromadb = (
                    embeddings is not None and 
                    len(embeddings) > 0
                )
                
                # Convert db_has_embedding to bool for comparison
                db_has_embedding_bool = bool(db_has_embedding)
                
                # Compare with database status
                if db_has_embedding_bool != has_embedding_in_chromadb:
                    # Update database to match ChromaDB
                    session.execute(
                        text("UPDATE chunk SET has_embedding = :has_embedding WHERE id = :id"),
                        {"has_embedding": has_embedding_in_chromadb, "id": chunk_id}
                    )
                    
                    if has_embedding_in_chromadb:
                        stats["synced_to_true"] += 1
                        print(f"  Fixed chunk {chunk_id}: DB False -> ChromaDB True")
                    else:
                        stats["synced_to_false"] += 1
                        print(f"  Warning: chunk {chunk_id}: DB True -> ChromaDB False")
                else:
                    stats["already_correct"] += 1
                    
            except Exception as e:
                stats["errors"] += 1
                print(f"  Error syncing chunk {chunk_id}: {str(e)}")
                continue
        
        # Commit all changes
        session.commit()
        session.close()
        
        # Display results
        print()
        print("=" * 60)
        print("Sync Results:")
        print("=" * 60)
        print(f"Total chunks processed:     {stats['total_chunks']}")
        print(f"Synced to True (FIXED):     {stats['synced_to_true']}")
        print(f"Synced to False:            {stats['synced_to_false']}")
        print(f"Already correct:            {stats['already_correct']}")
        print(f"Errors:                     {stats['errors']}")
        print()
        
        if stats['synced_to_true'] > 0:
            print(f"SUCCESS: Fixed {stats['synced_to_true']} chunks!")
            print(f"   These chunks had embeddings in ChromaDB but were showing")
            print(f"   as 'missing' in the database.")
        else:
            print("No chunks needed repair.")
        
        if stats['errors'] > 0:
            print(f"\nWARNING: {stats['errors']} errors occurred during sync.")
            print("   Check the output above for details.")
        
        print()
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        import traceback
        print()
        print("ERROR: Repair failed!")
        print(f"   {str(e)}")
        traceback.print_exc()
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
