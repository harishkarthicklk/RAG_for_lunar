import chromadb
import ollama

from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, START, END


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_DIR = "chroma_db"

COLLECTION_NAME = "learning_assistant"

EMBEDDING_MODEL = "nomic-embed-text:latest"

LLM_MODEL = "llama3"

TOP_K = 5

# ------------------------------------------------------------
# DISTANCE THRESHOLD
# ------------------------------------------------------------
# ChromaDB returns a distance value for each retrieved chunk.
#
# Based on the current testing of the knowledge base:
#
#     distance < 150 -> relevant context
#     distance >= 150 -> not sufficiently relevant
#
# The threshold can be tuned later after testing all documents.
# ------------------------------------------------------------

DISTANCE_THRESHOLD = 150


# ============================================================
# CONNECT TO CHROMADB
# ============================================================

client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


# ============================================================
# LANGGRAPH STATE
# ============================================================

class RAGState(TypedDict, total=False):

    # User's original question
    question: str

    # Query embedding
    query_embedding: List[float]

    # Top 5 search results
    search_results: Dict[str, Any]

    # Top 5 chunks
    top_chunks: List[Dict[str, Any]]

    # Best chunk selected from top 5
    best_chunk: Dict[str, Any]

    # Whether useful context exists
    context_found: bool

    # Final LLM answer
    answer: str


# ============================================================
# NODE 1
# EMBED USER QUESTION
# ============================================================

def embed_question_node(state: RAGState):

    question = state["question"]

    print("\n" + "=" * 70)
    print("STEP 1: EMBEDDING USER QUESTION")
    print("=" * 70)

    print(f"Question: {question}")

    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=question
    )

    embedding = response["embedding"]

    print("Embedding generated")
    print(f"Embedding dimensions: {len(embedding)}")

    return {
        "query_embedding": embedding
    }


# ============================================================
# NODE 2
# SEARCH CHROMADB
# ============================================================

def search_chromadb_node(state: RAGState):

    print("\n" + "=" * 70)
    print("STEP 2: VECTOR SEARCH IN CHROMADB")
    print("=" * 70)

    results = collection.query(
        query_embeddings=[state["query_embedding"]],
        n_results=TOP_K,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    top_chunks = []

    for i in range(len(documents)):

        chunk = {
            "id": ids[i],
            "text": documents[i],
            "metadata": metadatas[i],
            "distance": distances[i]
        }

        top_chunks.append(chunk)

    print(f"Top {len(top_chunks)} chunks retrieved")

    for i, chunk in enumerate(top_chunks):

        print("\n" + "-" * 60)
        print(f"Rank {i + 1}")
        print(f"Chunk ID : {chunk['id']}")
        print(f"Distance : {chunk['distance']}")
        print(
            f"Title    : "
            f"{chunk['metadata'].get('title', 'Unknown')}"
        )

    return {
        "search_results": results,
        "top_chunks": top_chunks
    }


# ============================================================
# NODE 3
# CHECK WHETHER USEFUL CONTEXT EXISTS
# ============================================================

def check_context_node(state: RAGState):

    print("\n" + "=" * 70)
    print("STEP 3: CHECKING CONTEXT")
    print("=" * 70)

    top_chunks = state.get("top_chunks", [])

    if not top_chunks:

        print("No chunks returned from ChromaDB")

        return {
            "context_found": False
        }

    # Best result is the first result because ChromaDB
    # returns results ordered by distance.
    best_distance = top_chunks[0]["distance"]

    print(f"Best distance: {best_distance}")
    print(f"Threshold    : {DISTANCE_THRESHOLD}")

    # --------------------------------------------------------
    # CONTEXT IS RELEVANT WHEN DISTANCE IS LESS THAN 70
    # --------------------------------------------------------

    if best_distance < DISTANCE_THRESHOLD:

        print("Useful context FOUND")

        return {
            "context_found": True
        }

    else:

        print("No sufficiently relevant context found")

        return {
            "context_found": False
        }


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

def route_after_context_check(state: RAGState):

    if state.get("context_found", False):

        return "select_best_chunk"

    return "direct_llm"


# ============================================================
# NODE 4
# SELECT BEST CHUNK
# ============================================================

def select_best_chunk_node(state: RAGState):

    print("\n" + "=" * 70)
    print("STEP 4: SELECTING BEST CHUNK")
    print("=" * 70)

    top_chunks = state["top_chunks"]

    # ChromaDB already orders results by distance.
    # Therefore the first result is the most similar.
    best_chunk = top_chunks[0]

    print(f"Selected chunk: {best_chunk['id']}")
    print(f"Distance      : {best_chunk['distance']}")
    print(
        f"Title         : "
        f"{best_chunk['metadata'].get('title', 'Unknown')}"
    )

    return {
        "best_chunk": best_chunk
    }


# ============================================================
# NODE 5A
# LLM WITH RETRIEVED CONTEXT
# ============================================================

def llm_with_context_node(state: RAGState):

    print("\n" + "=" * 70)
    print("STEP 5: SENDING QUESTION + BEST CHUNK TO LLAMA 3")
    print("=" * 70)

    question = state["question"]

    best_chunk = state["best_chunk"]

    context = best_chunk["text"]

    prompt = f"""
You are a Learning Assistant.

Answer the user's question using the provided context.

CONTEXT:
{context}

USER QUESTION:
{question}

Instructions:
- Use the context as the primary source.
- Give a clear and simple answer.
- Do not mention that you are using a vector database.
- Do not mention the RAG system.
- If the context does not completely answer the question,
  say that the provided context does not contain enough
  information rather than inventing details.
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    return {
        "answer": answer
    }


# ============================================================
# NODE 5B
# DIRECT LLM
# ============================================================

def direct_llm_node(state: RAGState):

    print("\n" + "=" * 70)
    print("STEP 5: NO RELEVANT CONTEXT")
    print("=" * 70)

    print("Sending question directly to Llama 3")

    question = state["question"]

    prompt = f"""
You are a helpful Learning Assistant.

Answer the following question clearly and accurately.

USER QUESTION:
{question}

There was no sufficiently relevant information found
in the Learning Assistant knowledge base.

Answer the question directly using your general knowledge.
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    return {
        "answer": answer
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

def build_graph():

    graph = StateGraph(RAGState)

    # --------------------------------------------------------
    # Add nodes
    # --------------------------------------------------------

    graph.add_node(
        "embed_question",
        embed_question_node
    )

    graph.add_node(
        "search_chromadb",
        search_chromadb_node
    )

    graph.add_node(
        "check_context",
        check_context_node
    )

    graph.add_node(
        "select_best_chunk",
        select_best_chunk_node
    )

    graph.add_node(
        "llm_with_context",
        llm_with_context_node
    )

    graph.add_node(
        "direct_llm",
        direct_llm_node
    )

    # --------------------------------------------------------
    # Normal flow
    # --------------------------------------------------------

    graph.add_edge(
        START,
        "embed_question"
    )

    graph.add_edge(
        "embed_question",
        "search_chromadb"
    )

    graph.add_edge(
        "search_chromadb",
        "check_context"
    )

    # --------------------------------------------------------
    # Conditional flow
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "check_context",
        route_after_context_check,
        {
            "select_best_chunk": "select_best_chunk",
            "direct_llm": "direct_llm"
        }
    )

    # --------------------------------------------------------
    # Context path
    # --------------------------------------------------------

    graph.add_edge(
        "select_best_chunk",
        "llm_with_context"
    )

    # --------------------------------------------------------
    # Both paths end here
    # --------------------------------------------------------

    graph.add_edge(
        "llm_with_context",
        END
    )

    graph.add_edge(
        "direct_llm",
        END
    )

    return graph.compile()


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("       LEARNING ASSISTANT - LANGGRAPH RAG")
    print("=" * 70)

    print(f"Embedding model : {EMBEDDING_MODEL}")
    print(f"LLM model       : {LLM_MODEL}")
    print(f"ChromaDB        : {COLLECTION_NAME}")
    print(f"Top K           : {TOP_K}")
    print(f"Threshold       : {DISTANCE_THRESHOLD}")

    print(f"\nKnowledge base contains {collection.count()} chunks.")

    # --------------------------------------------------------
    # Build LangGraph
    # --------------------------------------------------------

    app = build_graph()

    # --------------------------------------------------------
    # User question loop
    # --------------------------------------------------------

    while True:

        question = input(
            "\nYou > "
        ).strip()

        if not question:
            continue

        if question.lower() in {
            "exit",
            "quit",
            "q"
        }:

            print("\nExiting Learning Assistant...")
            break

        # ----------------------------------------------------
        # Initial state
        # ----------------------------------------------------

        initial_state = {
            "question": question
        }

        # ----------------------------------------------------
        # Run LangGraph
        # ----------------------------------------------------

        final_state = app.invoke(
            initial_state
        )

        # ----------------------------------------------------
        # Display answer
        # ----------------------------------------------------

        print("\n")
        print("=" * 70)
        print("FINAL ANSWER")
        print("=" * 70)

        print(final_state["answer"])

        print("=" * 70)


# ============================================================
# RUN
# ============================================================
def run_rag(question):
    app = build_graph()

    initial_state = {
        "question": question
    }

    final_state = app.invoke(initial_state)

    return final_state


if __name__ == "__main__":
    main()