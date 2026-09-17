
import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Learning Assistant",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("🧠 Learning Assistant")

st.write(
    "RAG-based Learning Assistant using Llama 3, "
    "nomic-embed-text, ChromaDB and LangGraph."
)

st.divider()

# ============================================================
# SELECT RAG APPROACH
# ============================================================

st.subheader("Select RAG Approach")

approach = st.radio(
    "Choose the approach you want to use:",
    [
        "Normal RAG",
        "Advanced RAG"
    ],
    horizontal=True
)

# ============================================================
# APPROACH INFORMATION / CONSTRAINTS
# ============================================================

if approach == "Normal RAG":

    with st.expander("ℹ️ How Normal RAG Works", expanded=True):

        st.markdown("""
        ### Normal RAG Constraints

        **1️⃣ Question Length**

        - The question should be between **300 and 350 characters**.

        **2️⃣ Vector Search**

        - The question is converted into an embedding using
          `nomic-embed-text`.
        - Vector search is performed in **ChromaDB**.

        **3️⃣ Distance Threshold**

        - If the best retrieved chunk has a **distance < 150**:
          - Top **5 chunks** are retrieved.
          - The **best (Rank 1) chunk** is selected.
          - **Original question + best chunk** are sent to Llama 3
            for the final answer.

        - If the best retrieved chunk has a **distance ≥ 150**:
          - The retrieved context is considered insufficient.
          - **Only the original question** is sent to Llama 3.
        """)

elif approach == "Advanced RAG":

    with st.expander("ℹ️ How Advanced RAG Works", expanded=True):

        st.markdown("""
        ### Advanced RAG Flow

        **1️⃣ Question Length Check**

        - If the question is **too short (< 200 characters)**,
          it is first sent to Llama 3.

        **2️⃣ Question Expansion — 1st LLM Call**

        - Llama 3 expands the short question into a more detailed
          search question.
        - The expanded question is targeted to approximately
          **300–350 characters**.
        - The expanded question is used only for vector search.

        **3️⃣ Vector Search**

        - The expanded question is converted into an embedding
          using `nomic-embed-text`.
        - Vector search is performed in **ChromaDB**.
        - Top **5 chunks** are retrieved.
        - The **best (Rank 1) chunk** is selected.

        **4️⃣ Final Answer — 2nd LLM Call**

        - The **original user question + best retrieved chunk**
          are sent to Llama 3.
        - Llama 3 generates the final answer.

        **LLM Calls for a Short Question:**

        `1st LLM Call → Question Expansion`

        `2nd LLM Call → Final Answer`
        """)

st.divider()

# ============================================================
# QUESTION INPUT
# ============================================================

st.subheader("Ask Your Question")

question = st.text_area(
    "Enter your question:",
    height=120,
    placeholder="Example: How does Proof of Work achieve consensus?"
)

# ============================================================
# SHOW QUESTION LENGTH
# ============================================================

if question:

    question_length = len(question.strip())

    st.caption(
        f"Question length: **{question_length} characters**"
    )

    if approach == "Normal RAG":

        if 300 <= question_length <= 350:
            st.success(
                "✓ Question length is within the required "
                "300–350 character range."
            )
        else:
            st.warning(
                "⚠️ Normal RAG requires a question between "
                "300 and 350 characters."
            )

    else:

        if question_length < 200:
            st.info(
                "ℹ️ Short question detected. "
                "Llama 3 will expand the question before vector search."
            )
        else:
            st.success(
                "✓ Question is long enough. "
                "Question expansion will be skipped."
            )

# ============================================================
# ASK BUTTON
# ============================================================

if st.button("🚀 Ask Question", type="primary"):

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    # ========================================================
    # NORMAL RAG
    # ========================================================

    if approach == "Normal RAG":

        from rag_pipeline import run_rag

        with st.spinner("Running Normal RAG..."):

            try:
                result = run_rag(question.strip())

            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        st.success("Normal RAG completed.")

        # ----------------------------------------------------
        # FINAL ANSWER
        # ----------------------------------------------------

        st.subheader("🤖 Final Answer")

        st.write(
            result.get(
                "answer",
                "No answer generated."
            )
        )

        # ----------------------------------------------------
        # RETRIEVAL INFORMATION
        # ----------------------------------------------------

        if result.get("top_chunks"):

            st.divider()

            st.subheader("📚 Retrieved Context")

            for i, chunk in enumerate(
                result["top_chunks"]
            ):

                metadata = chunk.get(
                    "metadata",
                    {}
                )

                title = metadata.get(
                    "title",
                    "Unknown"
                )

                distance = chunk.get(
                    "distance",
                    "N/A"
                )

                with st.expander(
                    f"Rank {i + 1} — {title}"
                ):

                    st.write(
                        f"**Chunk ID:** "
                        f"{chunk.get('id', 'N/A')}"
                    )

                    st.write(
                        f"**Distance:** {distance}"
                    )

                    st.write(
                        chunk.get("text", "")
                    )

    # ========================================================
    # ADVANCED RAG
    # ========================================================

    else:

        from rag_advanced import run_rag

        with st.spinner("Running Advanced RAG..."):

            try:
                result = run_rag(question.strip())

            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        st.success("Advanced RAG completed.")

        # ----------------------------------------------------
        # QUERY PROCESSING
        # ----------------------------------------------------

        st.subheader("🔍 Query Processing")

        col1, col2 = st.columns(2)

        with col1:

            st.write("**Original Question**")

            st.info(
                result.get(
                    "original_question",
                    question
                )
            )

        with col2:

            st.write("**Search Question**")

            st.info(
                result.get(
                    "search_question",
                    question
                )
            )

        # ----------------------------------------------------
        # FINAL ANSWER
        # ----------------------------------------------------

        st.divider()

        st.subheader("🤖 Final Answer")

        st.write(
            result.get(
                "answer",
                "No answer generated."
            )
        )

        # ----------------------------------------------------
        # RETRIEVED CHUNKS
        # ----------------------------------------------------

        if result.get("top_chunks"):

            st.divider()

            st.subheader("📚 Top 5 Retrieved Chunks")

            for i, chunk in enumerate(
                result["top_chunks"]
            ):

                metadata = chunk.get(
                    "metadata",
                    {}
                )

                title = metadata.get(
                    "title",
                    "Unknown"
                )

                distance = chunk.get(
                    "distance",
                    "N/A"
                )

                with st.expander(
                    f"Rank {i + 1} — {title}"
                ):

                    st.write(
                        f"**Chunk ID:** "
                        f"{chunk.get('id', 'N/A')}"
                    )

                    st.write(
                        f"**Distance:** {distance}"
                    )

                    st.write(
                        chunk.get("text", "")
                    )

        # ----------------------------------------------------
        # BEST CHUNK
        # ----------------------------------------------------

        best_chunk = result.get(
            "best_chunk",
            {}
        )

        if best_chunk:

            st.divider()

            st.subheader("⭐ Best Retrieved Chunk")

            metadata = best_chunk.get(
                "metadata",
                {}
            )

            st.write(
                f"**Chunk ID:** "
                f"{best_chunk.get('id', 'N/A')}"
            )

            st.write(
                f"**Title:** "
                f"{metadata.get('title', 'Unknown')}"
            )

            st.write(
                f"**Distance:** "
                f"{best_chunk.get('distance', 'N/A')}"
            )

            st.write(
                best_chunk.get("text", "")
            )

