# RAG_for_lunar
Agentic RAG Learning Assistant using LangGraph, ChromaDB, Ollama, and Llama3 with Normal and Advanced RAG approaches for context-aware question answering.

# RAG-Based Learning Assistant

## Overview

This project is a **Retrieval-Augmented Generation (RAG) based Learning Assistant** developed to answer questions from educational content related to **Microservices and Distributed Systems**.

The knowledge base is created from YouTube videos. The video content is converted into text, stored as JSON documents, divided into smaller chunks, converted into vector embeddings, and stored in **ChromaDB**.

The project implements and compares two different RAG approaches:

- **Approach 1: Normal RAG**
- **Approach 2: Advanced RAG with Query Expansion**

Both approaches use the same knowledge base and are evaluated using:

- Faithfulness
- Answer Relevance
- Context Precision

The project also provides a **Streamlit web interface** for interacting with both RAG approaches.

---

# Project Objective

The main objective of this project is to build a learning assistant that can retrieve relevant information from educational video content and generate answers using an LLM.

The project also investigates whether **query expansion before vector retrieval** can improve the quality of retrieved context and the final generated answer.

---

# Overall Workflow

```text
YouTube Videos
      ↓
Text Extraction
      ↓
JSON Documents
      ↓
Text Chunking
      ↓
Embedding Generation
      ↓
ChromaDB
      ↓
┌───────────────────────────────┐
│                               │
│       RAG Approaches          │
│                               │
│  1. Normal RAG                │
│  2. Advanced RAG              │
│                               │
└───────────────┬───────────────┘
                ↓
             Llama3
                ↓
          Final Answer
                ↓
          Streamlit UI
