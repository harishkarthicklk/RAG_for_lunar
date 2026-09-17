import json
import numpy as np
import chromadb
import ollama

# Import LangGraph application builders from both files
from rag_pipeline import build_graph as build_pipeline_graph
from rag_advanced import build_graph as build_advanced_graph

# ============================================================
# CONFIGURATION
# ============================================================
LLM_MODEL = "llama3"
EMBEDDING_MODEL = "nomic-embed-text:latest"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "learning_assistant"

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(name=COLLECTION_NAME)

TEST_QUESTIONS = [
    "What is the definition of a Circuit Breaker pattern in microservices?",
    "What are non-transient failures, and how do they impact system availability?",
    "How does the Circuit Breaker pattern prevent cascading failures across microservices?",
    "What are the three primary states of a Circuit Breaker state machine?",
    "How does a Circuit Breaker transition from the Open state to the Half-Open state?",
    "What role does the failure counter threshold play in tripping the circuit to an Open state?",
    "Why is manual control over Circuit Breaker state transitions necessary in production?",
    "What design considerations must be addressed when setting timeouts between Open and Half-Open states?",
    "Why should developers avoid applying the Circuit Breaker pattern to transient network failures?",
    "How should logging and monitoring be configured when implementing a Circuit Breaker?"
]

# ============================================================
# HELPER FUNCTIONS & EMBEDDINGS
# ============================================================
def get_embedding(text: str):
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]

def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def query_llm(prompt: str) -> str:
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"].strip()

# ============================================================
# EVALUATION METRICS METRIC CALCULATORS
# ============================================================
def evaluate_faithfulness(answer: str, context: str) -> float:
    """Calculates supported claims / total claims."""
    if not context or not answer:
        return 0.0

    prompt_extract = f"""
    Extract all distinct factual statements/claims from the following text as a JSON list of strings.
    TEXT: {answer}
    Output ONLY valid JSON: ["statement 1", "statement 2", ...]
    """
    try:
        statements_raw = query_llm(prompt_extract)
        statements = json.loads(statements_raw)
    except Exception:
        statements = [line.strip('- ') for line in answer.split('.') if len(line.strip()) > 10]

    if not statements:
        return 1.0

    supported_count = 0
    for stmt in statements:
        prompt_check = f"""
        Determine if the following statement is supported by the context.
        CONTEXT: {context}
        STATEMENT: {stmt}
        Answer ONLY 'YES' or 'NO'.
        """
        verdict = query_llm(prompt_check).upper()
        if "YES" in verdict:
            supported_count += 1

    return supported_count / len(statements)

def evaluate_answer_relevance(original_question: str, answer: str) -> float:
    """Calculates similarity between user question and question reconstructed from answer."""
    if not answer:
        return 0.0

    prompt = f"""
    Generate a concise question that is directly answered by the following text.
    TEXT: {answer}
    Output ONLY the question.
    """
    generated_question = query_llm(prompt)
    
    emb_orig = get_embedding(original_question)
    emb_gen = get_embedding(generated_question)
    
    return max(0.0, cosine_similarity(emb_orig, emb_gen))

def evaluate_context_precision(question: str, top_chunks: list) -> float:
    """Calculates Precision@k weighted cumulative score for retrieved context."""
    if not top_chunks:
        return 0.0

    relevance_flags = []
    for chunk in top_chunks[:5]:
        prompt = f"""
        Is the following chunk relevant to answering the question?
        QUESTION: {question}
        CHUNK: {chunk.get('text', '')}
        Answer ONLY 'YES' or 'NO'.
        """
        verdict = query_llm(prompt).upper()
        relevance_flags.append(1 if "YES" in verdict else 0)

    total_relevant = sum(relevance_flags)
    if total_relevant == 0:
        return 0.0

    precision_at_k = []
    relevant_so_far = 0
    for k, rel in enumerate(relevance_flags, start=1):
        if rel == 1:
            relevant_so_far += 1
            precision_at_k.append(relevant_so_far / k)

    return sum(precision_at_k) / total_relevant

# ============================================================
# EVALUATION RUNNER
# ============================================================
def evaluate_system(app, is_advanced: bool = False):
    results = []

    for idx, question in enumerate(TEST_QUESTIONS, start=1):
        print(f"Evaluating Question {idx}/{len(TEST_QUESTIONS)}...")
        
        # Invoke LangGraph Application
        if is_advanced:
            initial_state = {"original_question": question}
        else:
            initial_state = {"question": question}
            
        final_state = app.invoke(initial_state)
        
        # Extract Answer and Chunks
        answer = final_state.get("answer", "")
        best_chunk = final_state.get("best_chunk", {})
        context = best_chunk.get("text", "")
        top_chunks = final_state.get("top_chunks", [best_chunk] if best_chunk else [])

        # Compute Metrics
        faithfulness = evaluate_faithfulness(answer, context)
        relevance = evaluate_answer_relevance(question, answer)
        precision = evaluate_context_precision(question, top_chunks)
        overall = (faithfulness + relevance + precision) / 3.0

        results.append({
            "id": f"Q{idx}",
            "faithfulness": faithfulness,
            "relevance": relevance,
            "precision": precision,
            "overall": overall
        })

    return results

# ============================================================
# MAIN EXECUTION & PRINT COMPARISON TABLE
# ============================================================
def main():
    print("\n=======================================================")
    print("STARTING EVALUATION: RAG PIPELINE VS ADVANCED RAG")
    print("=======================================================\n")

    # 1. Compile Graphs
    pipeline_app = build_pipeline_graph()
    advanced_app = build_advanced_graph()

    # 2. Run Evaluations
    print("--- Running Evaluation on rag_pipeline_2.py ---")
    pipeline_results = evaluate_system(pipeline_app, is_advanced=False)

    print("\n--- Running Evaluation on rag_advanced_2.py ---")
    advanced_results = evaluate_system(advanced_app, is_advanced=True)

    # 3. Compute Means
    p_faith = np.mean([r["faithfulness"] for r in pipeline_results])
    p_rel = np.mean([r["relevance"] for r in pipeline_results])
    p_prec = np.mean([r["precision"] for r in pipeline_results])
    p_over = np.mean([r["overall"] for r in pipeline_results])

    a_faith = np.mean([r["faithfulness"] for r in advanced_results])
    a_rel = np.mean([r["relevance"] for r in advanced_results])
    a_prec = np.mean([r["precision"] for r in advanced_results])
    a_over = np.mean([r["overall"] for r in advanced_results])

    # 4. Print Final Results Table
    print("\n" + "="*80)
    print(f"{'EVALUATION METRIC':<25} | {'RAG PIPELINE':<18} | {'RAG ADVANCED':<18} | {'DIFF':<10}")
    print("="*80)
    print(f"{'Faithfulness':<25} | {p_faith:<18.4f} | {a_faith:<18.4f} | {a_faith - p_faith:+.4f}")
    print(f"{'Answer Relevance':<25} | {p_rel:<18.4f} | {a_rel:<18.4f} | {a_rel - p_rel:+.4f}")
    print(f"{'Context Precision':<25} | {p_prec:<18.4f} | {a_prec:<18.4f} | {a_prec - p_prec:+.4f}")
    print("-"*80)
    print(f"{'OVERALL RAG SCORE':<25} | {p_over:<18.4f} | {a_over:<18.4f} | {a_over - p_over:+.4f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()