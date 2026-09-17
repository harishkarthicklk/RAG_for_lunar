import json
from pathlib import Path

import chromadb
import ollama


# ============================================================
# CONFIGURATION
# ============================================================

CHUNKS_DIR = Path("chunks")
CHROMA_DIR = Path("chroma_db")

EMBEDDING_MODEL = "nomic-embed-text:latest"
COLLECTION_NAME = "learning_assistant"


# ============================================================
# INITIALIZE CHROMADB
# ============================================================

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={
        "description": "RAG knowledge base for Learning Assistant"
    }
)


# ============================================================
# GENERATE EMBEDDING
# ============================================================

def generate_embedding(text):
    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=text
    )

    return response["embedding"]


# ============================================================
# PROCESS ONE CHUNK FILE
# ============================================================

def process_chunk_file(json_file):

    print(f"\nProcessing: {json_file.name}")

    with open(json_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Chunks found: {len(chunks)}")

    for index, chunk in enumerate(chunks):

        chunk_id = chunk["chunk_id"]
        text = chunk["text"]

        metadata = {
            "doc_id": str(chunk.get("doc_id", "")),
            "title": str(chunk.get("title", "")),
            "topic": str(chunk.get("topic", "")),
            "source": str(chunk.get("source", "")),
            "chunk_index": int(chunk.get("chunk_index", index)),
            "total_chunks": int(chunk.get("total_chunks", len(chunks)))
        }

        print(
            f"  Embedding chunk "
            f"{index + 1}/{len(chunks)}: {chunk_id}"
        )

        embedding = generate_embedding(text)

        collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata]
        )

    print(f"Completed: {json_file.name}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LEARNING ASSISTANT - EMBEDDING + CHROMADB")
    print("=" * 70)

    print(f"Embedding model : {EMBEDDING_MODEL}")
    print(f"ChromaDB path   : {CHROMA_DIR}")
    print(f"Collection      : {COLLECTION_NAME}")

    json_files = sorted(
        CHUNKS_DIR.glob("*_chunks.json")
    )

    if not json_files:
        print("\nERROR: No chunk files found.")
        print("Run chunk_documents.py first.")
        return

    print(f"\nChunk files found: {len(json_files)}")

    # --------------------------------------------------------
    # Process every chunk file
    # --------------------------------------------------------

    for json_file in json_files:
        process_chunk_file(json_file)

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EMBEDDING + CHROMADB STORAGE COMPLETED")
    print("=" * 70)

    print(f"Total documents in ChromaDB: {collection.count()}")
    print(f"Database location: {CHROMA_DIR.absolute()}")


if __name__ == "__main__":
    main()