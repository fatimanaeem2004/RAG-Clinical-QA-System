import json
import os

def build_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

def create_notebooks():
    os.makedirs("notebooks", exist_ok=True)
    
    # ----------------------------------------------------
    # Notebook 1: Data Preparation
    # ----------------------------------------------------
    nb1_cells = [
        {
            "cell_type": "markdown",
            "source": [
                "# RAG Data Preparation and Segmenting\n",
                "This notebook details the loading, processing, and chunking experiments performed on our domain-specific medical Guidelines corpus.\n",
                "Specifically, we study how **Chunk Size** and **Chunk Overlap** affect context segmentation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "source": [
                "# Verify imports\n",
                "import os\n",
                "import sys\n",
                "sys.path.append('..')\n",
                "from src.document_processor import DocumentProcessor\n",
                "print(\"Processor library loaded successfully!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "source": [
                "### Load and Inspect the Corpus Documents\n",
                "We read the guidelines text files from the `../data/` directory and print their character lengths."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "source": [
                "processor = DocumentProcessor()\n",
                "data_dir = \"../data\"\n",
                "docs = processor.load_documents(data_dir)\n",
                "for name, text in docs.items():\n",
                "    print(f\"Document: {name} | Length: {len(text)} chars | Word Count: {len(text.split())} words\")"
            ]
        },
        {
            "cell_type": "markdown",
            "source": [
                "### Chunking Strategy Experimentation\n",
                "Here we test split dimensions: \n",
                "1. **Small Chunk** (150 tokens, 15 tokens overlap)\n",
                "2. **Medium Chunk** (300 tokens, 30 tokens overlap)\n",
                "3. **Large Chunk** (600 tokens, 60 tokens overlap)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "source": [
                "for size, overlap in [(150, 15), (300, 30), (600, 60)]:\n",
                "    chunks = processor.process_directory(data_dir, chunk_size=size, chunk_overlap=overlap)\n",
                "    print(f\"[Experiment] Size: {size:3d} | Overlap: {overlap:2d} | Generated Chunks: {len(chunks):2d}\")\n",
                "    if chunks:\n",
                "        avg_len = sum(c['token_count'] for c in chunks) / len(chunks)\n",
                "        print(f\"             -> Average actual chunk size: {avg_len:.1f} tokens\")"
            ]
        },
        {
            "cell_type": "markdown",
            "source": [
                "### Inspecting Chunk Contents\n",
                "Let's print the first chunk of a document to inspect text alignment and segmentation completeness."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "source": [
                "chunks = processor.process_directory(data_dir, chunk_size=300, chunk_overlap=30)\n",
                "first_chunk = chunks[0]\n",
                "print(f\"Chunk ID: {first_chunk['chunk_id']}\")\n",
                "print(f\"Token Count: {first_chunk['token_count']}\")\n",
                "print(\"Content Snippet:\\n----------------\")\n",
                "print(first_chunk['text'][:350] + \"...\")"
            ]
        }
    ]
    
    with open("notebooks/01_data_preparation.ipynb", "w", encoding="utf-8") as f:
        json.dump(build_notebook(nb1_cells), f, indent=1)
        
    # ----------------------------------------------------
    # Notebook 2: Retrieval Experimentation
    # ----------------------------------------------------
    nb2_cells = [
        {
            "cell_type": "markdown",
            "source": [
                "# RAG Retrieval Strategy Experimentation\n",
                "This notebook evaluates three retrieval approaches on the domain corpus:\n",
                "1. **Lexical Keyword matching (BM25)**\n",
                "2. **Semantic Dense Vector search (Cosine Similarity via all-MiniLM-L6-v2)**\n",
                "3. **Hybrid Search (BM25 + Semantic fused via Reciprocal Rank Fusion - RRF)**"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "source": [
                "import sys\n",
                "sys.path.append('..')\n",
                "from src.document_processor import DocumentProcessor\n",
                "from src.retriever import Retriever\n",
                "import numpy as np\n",
                "print(\"Retrieval classes loaded successfully!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "source": [
                "### Load and Index Chunks\n",
                "We use the Medium chunk size configuration (300 tokens) to process the corpus and build the Retriever indexes."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "source": [
                "processor = DocumentProcessor()\n",
                "chunks = processor.process_directory(\"../data\", chunk_size=300, chunk_overlap=30)\n",
                "retriever = Retriever(chunks)\n",
                "print(f\"Successfully indexed {len(chunks)} chunks in BM25 and Vector databases!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "source": [
                "### Compare Retrieval Top-K Results\n",
                "Let's test search queries to compare lexical matching vs conceptual similarity."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "source": [
                "query = \"What medication is first line for a patient with protein in their urine or CKD?\"\n",
                "\n",
                "print(\"=== BM25 KEYWORD SEARCH ===\")\n",
                "bm25_res = retriever.retrieve(query, method=\"bm25\", top_k=2)\n",
                "for c, s in bm25_res:\n",
                "    print(f\"Score: {s:.3f} | {c['chunk_id']} | Text: {c['text'][:150]}...\")\n",
                "    \n",
                "print(\"\\n=== SEMANTIC VECTOR SEARCH ===\")\n",
                "sem_res = retriever.retrieve(query, method=\"semantic\", top_k=2)\n",
                "for c, s in sem_res:\n",
                "    print(f\"Score: {s:.3f} | {c['chunk_id']} | Text: {c['text'][:150]}...\")\n",
                "    \n",
                "print(\"\\n=== HYBRID RRF SEARCH ===\")\n",
                "hyb_res = retriever.retrieve(query, method=\"hybrid\", top_k=2)\n",
                "for c, s in hyb_res:\n",
                "    print(f\"RRF Score: {s:.5f} | {c['chunk_id']} | Text: {c['text'][:150]}...\")"
            ]
        },
        {
            "cell_type": "markdown",
            "source": [
                "### Observations\n",
                "- **BM25** matches specific keywords like 'CKD', but might rank general sentences high if they happen to contain the words.\n",
                "- **Semantic search** catches concepts like 'protein in urine' (matching the clinical term 'albuminuria' in our text), showing strong semantic understanding.\n",
                "- **Hybrid search (RRF)** effectively bubbles up chunks that contain both precise terminology and conceptual proximity, offering the best overall results."
            ]
        }
    ]
    
    with open("notebooks/02_retrieval_experimentation.ipynb", "w", encoding="utf-8") as f:
        json.dump(build_notebook(nb2_cells), f, indent=1)

    # ----------------------------------------------------
    # Notebook 3: RAG Pipeline Evaluation
    # ----------------------------------------------------
    nb3_cells = [
        {
            "cell_type": "markdown",
            "source": [
                "# RAG Pipeline Execution and Performance Evaluation\n",
                "This notebook compiles the complete RAG pipeline, runs benchmark evaluations using our custom LLM-as-a-judge evaluators, "
                "compares latencies, and plots results."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "source": [
                "import sys\n",
                "import pandas as pd\n",
                "sys.path.append('..')\n",
                "from src.document_processor import DocumentProcessor\n",
                "from src.retriever import Retriever\n",
                "from src.rag_engine import RAGEngine\n",
                "from src.evaluator import RAGEvaluator\n",
                "print(\"All pipeline classes loaded successfully!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "source": [
                "### Load the LLM and Evaluator System\n",
                "We initialize the engine with the lightweight `Qwen/Qwen2.5-0.5B-Instruct` model."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "source": [
                "model_name = \"Qwen/Qwen2.5-0.5B-Instruct\"\n",
                "engine = RAGEngine(model_name=model_name)\n",
                "evaluator = RAGEvaluator(engine.model, engine.tokenizer)\n",
                "print(\"LLM loaded and ready for inference and evaluation.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "source": [
                "### Running a Single RAG Query and Evaluation\n",
                "Let's run a test query, retrieve chunks using the Hybrid method, generate an answer with reordering, and evaluate its quality."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "source": [
                "processor = DocumentProcessor()\n",
                "chunks = processor.process_directory(\"../data\", chunk_size=300, chunk_overlap=30)\n",
                "retriever = Retriever(chunks)\n",
                "\n",
                "query = \"What is the proper patient sitting posture before measuring blood pressure?\"\n",
                "retrieved = retriever.retrieve(query, method=\"hybrid\", top_k=3)\n",
                "\n",
                "# Generate\n",
                "answer, context, inf_time = engine.generate_answer(query, retrieved, post_processing=\"reorder\")\n",
                "print(f\"--- Generated Answer (Inference Time: {inf_time:.2f}s) ---\\n\", answer)\n",
                "\n",
                "# Evaluate\n",
                "faith_score, faith_audit = evaluator.evaluate_faithfulness(context, answer)\n",
                "rel_score, rel_audit = evaluator.evaluate_relevance(query, answer)\n",
                "print(f\"\\nEvaluation Metrics:\\n- Faithfulness: {faith_score:.2f}\\n- Relevance: {rel_score:.2f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "source": [
                "### Loading benchmark results and plotting comparison charts\n",
                "We read the experiment results CSV and plot Faithfulness vs Relevance and Latencies."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "source": [
                "import matplotlib.pyplot as plt\n",
                "import os\n",
                "\n",
                "if os.path.exists(\"../data/results.csv\"):\n",
                "    df = pd.read_csv(\"../data/results.csv\")\n",
                "    print(df.to_string())\n",
                "else:\n",
                "    print(\"Run `run_experiments.py` first to output results CSV!\")"
            ]
        }
    ]
    
    with open("notebooks/03_rag_pipeline_evaluation.ipynb", "w", encoding="utf-8") as f:
        json.dump(build_notebook(nb3_cells), f, indent=1)

    print("Notebooks successfully created in the notebooks/ folder!")

if __name__ == "__main__":
    create_notebooks()
