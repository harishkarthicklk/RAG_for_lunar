import os
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENTS_DIR = Path("documents")
CHUNKS_DIR = Path("chunks")

# Chunk configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEXT SPLITTER
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        ", ",
        " ",
        ""
    ],
    length_function=len
)


# ============================================================
# PROCESS ONE JSON DOCUMENT
# ============================================================

def process_document(json_file):

    print(f"\nProcessing: {json_file.name}")

    # --------------------------------------------------------
    # Read JSON
    # --------------------------------------------------------

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --------------------------------------------------------
    # Extract metadata
    # --------------------------------------------------------

    doc_id = data.get("doc_id", json_file.stem)
    title = data.get("title", json_file.stem)
    topic = data.get("topic", "")
    source = data.get("source", json_file.name)

    transcript = data.get("transcript", "").strip()

    if not transcript:
        print(f"WARNING: No transcript found in {json_file.name}")
        return

    # --------------------------------------------------------
    # Split transcript into chunks
    # --------------------------------------------------------

    chunks = text_splitter.split_text(transcript)

    print(f"Original characters : {len(transcript)}")
    print(f"Number of chunks    : {len(chunks)}")

    # --------------------------------------------------------
    # Create chunk objects
    # --------------------------------------------------------

    chunk_documents = []

    for index, chunk_text in enumerate(chunks):

        chunk_id = f"{doc_id}_chunk_{index:04d}"

        chunk_document = {
            "chunk_id": chunk_id,

            "doc_id": doc_id,

            "title": title,

            "topic": topic,

            "source": source,

            "chunk_index": index,

            "total_chunks": len(chunks),

            "text": chunk_text
        }

        chunk_documents.append(chunk_document)

    # --------------------------------------------------------
    # Save chunks
    # --------------------------------------------------------

    output_file = CHUNKS_DIR / f"{json_file.stem}_chunks.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            chunk_documents,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"Saved to: {output_file}")


# ============================================================
# PROCESS ALL JSON DOCUMENTS
# ============================================================

def main():

    json_files = sorted(DOCUMENTS_DIR.glob("*.json"))

    if not json_files:
        print("No JSON documents found inside the documents folder.")
        return

    print("=" * 60)
    print("RAG DOCUMENT CHUNKING")
    print("=" * 60)

    print(f"Documents found: {len(json_files)}")

    for json_file in json_files:
        process_document(json_file)

    print("\n" + "=" * 60)
    print("CHUNKING COMPLETED")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()