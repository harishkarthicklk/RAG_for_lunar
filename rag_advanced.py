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
# QUESTION LENGTH CONFIGURATION
# ------------------------------------------------------------

SHORT_QUESTION_LIMIT = 200

EXPANDED_QUESTION_MIN = 300
EXPANDED_QUESTION_MAX = 350

# ------------------------------------------------------------
# CHROMADB DISTANCE THRESHOLD
# ------------------------------------------------------------

DISTANCE_THRESHOLD = 180


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

    # Original question entered by user
    original_question: str

    # Question used for embedding/vector search
    search_question: str

    # Query embedding
    query_embedding: List[float]

    # Top 5 ChromaDB results
    search_results: Dict[str, Any]

    # Top 5 chunks
    top_chunks: List[Dict[str, Any]]

    # Best chunk
    best_chunk: Dict[str, Any]

    # Whether original question was short
    is_short_question: bool

    # Final answer
    answer: str


# ============================================================
# NODE 1
# CHECK QUESTION LENGTH
# ============================================================

def check_question_length_node(state: RAGState):

    print("\n" + "=" * 70)
    print("STEP 1: CHECKING QUESTION LENGTH")
    print("=" * 70)

    question = state["original_question"]

    question_length = len(question)

    print(f"Original question : {question}")
    print(f"Question length   : {question_length} characters")
    print(f"Short limit       : {SHORT_QUESTION_LIMIT} characters")

    if question_length < SHORT_QUESTION_LIMIT:

        print("Question type     : SHORT QUESTION")
        print("LLM expansion     : REQUIRED")

        return {
            "is_short_question": True
        }

    else:

        print("Question type     : NORMAL/LONG QUESTION")
        print("LLM expansion     : NOT REQUIRED")

        return {
            "is_short_question": False,
            "search_question": question
        }


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

def route_after_question_check(state: RAGState):

    if state.get("is_short_question", False):

        return "expand_question"

    return "embed_question"


# ============================================================
# NODE 2
# EXPAND SHORT QUESTION USING LLM
# ============================================================

def expand_question_node(state: RAGState):

    print("\n" + "=" * 70)
    print("STEP 2: EXPANDING SHORT QUESTION USING LLAMA 3")
    print("=" * 70)

    original_question = state["original_question"]

    print(f"Original question:")
    print(original_question)

    prompt = f"""
You are a question expansion assistant for a Learning Assistant
RAG system.

The user has provided a short question.

Your task is to expand the question into a clear and detailed
search query containing between 300 and 350 characters.

IMPORTANT RULES:

1. Preserve the exact meaning of the original question.
2. Do not answer the question.
3. Do not change the topic.
4. Add useful context, concepts, relationships, and keywords
   that are directly related to the original question.
5. The expanded question will be used only for vector search.
6. Output ONLY the expanded question.
7. Do not add explanations.
8. Do not use labels such as "Expanded question:".
9. Keep the final output between 300 and 350 characters.

ORIGINAL QUESTION:
{original_question}
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

    expanded_question = response["message"]["content"].strip()

    print("\nExpanded question:")
    print(expanded_question)

    print(
        f"\nExpanded question length: "
        f"{len(expanded_question)} characters"
    )

    return {
        "search_question": expanded_question
    }


# ============================================================
# NODE 3
# EMBED QUESTION
# ============================================================

def embed_question_node(state: RAGState):

    print("\n" + "=" * 70)
    print("STEP 3: EMBEDDING SEARCH QUESTION")
    print("=" * 70)

    search_question = state["search_question"]

    print(f"Question used for embedding:")
    print(search_question)

    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=search_question
    )

    embedding = response["embedding"]

    print("\nEmbedding generated")
    print(f"Embedding dimensions: {len(embedding)}")

    return {
        "query_embedding": embedding
    }


# ============================================================
# NODE 4
# VECTOR SEARCH IN CHROMADB
# ============================================================

def search_chromadb_node(state: RAGState):

    print("\n" + "=" * 70)
    print("STEP 4: VECTOR SEARCH IN CHROMADB")
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
# NODE 5
# CHECK RETRIEVED CONTEXT
# ============================================================

def check_context_node(state: RAGState):

    print("\n" + "=" * 70)
    print("STEP 5: CHECKING RETRIEVED CONTEXT")
    print("=" * 70)

    top_chunks = state.get("top_chunks", [])

    if not top_chunks:

        print("No chunks returned from ChromaDB")

        return {
            "best_chunk": {}
        }

    best_distance = top_chunks[0]["distance"]

    print(f"Best distance : {best_distance}")

    print(f"Threshold     : {DISTANCE_THRESHOLD}")

    if best_distance < DISTANCE_THRESHOLD:

        print("Useful context FOUND")

        return {
            "best_chunk": top_chunks[0]
        }

    else:

        print("No sufficiently relevant context found")

        return {
            "best_chunk": {}
        }


# ============================================================
# NODE 6
# SELECT BEST CHUNK
# ============================================================

def select_best_chunk_node(state: RAGState):

    print("\n" + "=" * 70)
    print("STEP 6: SELECTING BEST CHUNK")
    print("=" * 70)

    top_chunks = state.get("top_chunks", [])

    if not top_chunks:

        return {
            "best_chunk": {}
        }

    # ChromaDB orders results by distance.
    # Rank 1 is therefore the best retrieved chunk.

    best_chunk = top_chunks[0]

    print(f"Selected chunk : {best_chunk['id']}")

    print(f"Distance       : {best_chunk['distance']}")

    print(
        f"Title          : "
        f"{best_chunk['metadata'].get('title', 'Unknown')}"
    )

    return {
        "best_chunk": best_chunk
    }


# ============================================================
# CONDITIONAL ROUTING AFTER CONTEXT CHECK
# ============================================================

def route_after_context_check(state: RAGState):

    best_chunk = state.get("best_chunk", {})

    if best_chunk:

        return "llm_with_context"

    return "direct_llm"


# ============================================================
# NODE 7A
# FINAL LLM WITH RETRIEVED CONTEXT
# ============================================================

def llm_with_context_node(state: RAGState):

    print("\n" + "=" * 70)
    print("STEP 7: FINAL LLM CALL")
    print("=" * 70)

    print("Sending:")
    print("1. Original user question")
    print("2. Best retrieved chunk")
    print("to Llama 3")

    original_question = state["original_question"]

    best_chunk = state["best_chunk"]

    context = best_chunk["text"]

    prompt = f"""
You are a Learning Assistant.

Answer the user's ORIGINAL question using the provided
retrieved context.

IMPORTANT:

The expanded question was used only for vector search.

You must answer the ORIGINAL QUESTION.

CONTEXT:
{context}

ORIGINAL USER QUESTION:
{original_question}

Instructions:
- Use the retrieved context as the primary source.
- Answer the original question directly.
- Give a clear and simple explanation.
- Do not mention the vector database.
- Do not mention embeddings.
- Do not mention LangGraph.
- Do not mention the RAG pipeline.
- Do not mention that the question was expanded.
- Do not blindly copy the context.
- If the context does not completely answer the question,
  clearly state that the provided context does not contain
  enough information instead of inventing details.
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
# NODE 7B
# DIRECT LLM
# ============================================================

def direct_llm_node(state: RAGState):

    print("\n" + "=" * 70)
    print("STEP 7: NO SUFFICIENTLY RELEVANT CONTEXT")
    print("=" * 70)

    print("Sending original question directly to Llama 3")

    original_question = state["original_question"]

    prompt = f"""
You are a helpful Learning Assistant.

Answer the following question clearly and accurately.

ORIGINAL USER QUESTION:
{original_question}

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
        "check_question_length",
        check_question_length_node
    )

    graph.add_node(
        "expand_question",
        expand_question_node
    )

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
    # START
    # --------------------------------------------------------

    graph.add_edge(
        START,
        "check_question_length"
    )

    # --------------------------------------------------------
    # Question length routing
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "check_question_length",
        route_after_question_check,
        {
            "expand_question": "expand_question",
            "embed_question": "embed_question"
        }
    )

    # --------------------------------------------------------
    # Short question path
    # --------------------------------------------------------

    graph.add_edge(
        "expand_question",
        "embed_question"
    )

    # --------------------------------------------------------
    # Embedding
    # --------------------------------------------------------

    graph.add_edge(
        "embed_question",
        "search_chromadb"
    )

    # --------------------------------------------------------
    # Vector search
    # --------------------------------------------------------

    graph.add_edge(
        "search_chromadb",
        "check_context"
    )

    # --------------------------------------------------------
    # Context checking
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "check_context",
        route_after_context_check,
        {
            "llm_with_context": "select_best_chunk",
            "direct_llm": "direct_llm"
        }
    )

    # --------------------------------------------------------
    # Best chunk
    # --------------------------------------------------------

    graph.add_edge(
        "select_best_chunk",
        "llm_with_context"
    )

    # --------------------------------------------------------
    # END
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

    print("       LEARNING ASSISTANT - ADVANCED LANGGRAPH RAG")

    print("=" * 70)

    print(f"Embedding model : {EMBEDDING_MODEL}")

    print(f"LLM model       : {LLM_MODEL}")

    print(f"ChromaDB        : {COLLECTION_NAME}")

    print(f"Top K           : {TOP_K}")

    print(f"Distance limit  : < {DISTANCE_THRESHOLD}")

    print(
        f"Short question  : < {SHORT_QUESTION_LIMIT} characters"
    )

    print(
        f"Expansion range : "
        f"{EXPANDED_QUESTION_MIN}-"
        f"{EXPANDED_QUESTION_MAX} characters"
    )

    print(
        f"\nKnowledge base contains "
        f"{collection.count()} chunks."
    )

    # --------------------------------------------------------
    # Build graph
    # --------------------------------------------------------

    app = build_graph()

    # --------------------------------------------------------
    # Interactive question loop
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
        # Initial LangGraph state
        # ----------------------------------------------------

        initial_state = {
            "original_question": question
        }

        # ----------------------------------------------------
        # Run LangGraph
        # ----------------------------------------------------

        final_state = app.invoke(
            initial_state
        )

        # ----------------------------------------------------
        # Display final answer
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
        "original_question": question
    }

    final_state = app.invoke(initial_state)

    return final_state


if __name__ == "__main__":
    main()