# Clinical RAG-QA Engine and Evaluation Suite

This repository contains a fully local, reproducible Retrieval-Augmented Generation (RAG) question-answering system developed for the **Introduction to Text Analytics (Assignment 04)** at the **Institute of Business Administration (IBA)**.

The system retrieves information from a curated corpus of cardiovascular disease and hypertension guidelines, and generates answers using local large language models (LLMs) on standard CPU hardware. It features comprehensive experimentation across chunk sizes, retrieval models, post-retrieval techniques, and LLM sizes, complete with custom judge-based evaluations and latency benchmarking.

---

## Project Structure

```
rag_assignment/
├── data/
│   ├── cardiovascular_guidelines_diagnosis.txt    # Diagnosis guidelines
│   ├── cardiovascular_guidelines_pharmacology.txt # Treatment guidelines
│   ├── cardiovascular_guidelines_lifestyle.txt    # Non-pharmacological guidelines
│   └── results.csv                                # Experiment evaluation results
├── notebooks/
│   ├── 01_data_preparation.ipynb                  # Data loading & chunking
│   ├── 02_retrieval_experimentation.ipynb         # BM25 vs Semantic vs Hybrid
│   └── 03_rag_pipeline_evaluation.ipynb           # Generation, latency & judge audits
├── src/
│   ├── document_processor.py                      # Text parsing & token segmenting
│   ├── retriever.py                               # Lexical, Cosine Semantic & RRF Hybrid
│   ├── rag_engine.py                              # Qwen-Instruct integration & post-processing
│   └── evaluator.py                               # Faithfulness & Answer Relevance judges
├── app.py                                         # Streamlit Interactive Web Application
├── run_experiments.py                             # Execution script for benchmark runs
├── generate_report.py                             # Document compiler for Word report
├── Fatima_Naeem.docx                              # Final Compiled Microsoft Word Report
├── requirements.txt                               # System dependencies
└── README.md                                      # Project guide
```

---

## Getting Started

Follow these steps in your terminal to reproduce the virtual environment, execute the benchmarks, compile the Word report, and launch the interactive app.

### 1. Initialize Virtual Environment
Create a clean Python virtual environment to isolate dependencies:
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Dependencies
Install all required libraries, including CPU-configured PyTorch, Transformers, and reporting libraries:
```bash
pip install -r requirements.txt
```

### 3. Generate Notebooks & Set Up Structure
Run the creation script to compile the experimental notebooks:
```bash
python create_notebooks.py
```

### 4. Run Evaluation Experiments
Run the test queries across the 9 configurations (chunk size, retrieval methods, LLM size, and reordering options):
```bash
python run_experiments.py
```
This queries the local models, computes average Faithfulness and Relevance metrics, logs execution latencies, outputs the results under `data/results.csv`, and plots the comparison charts.

### 5. Compile Word Report
Compile the final, styled academic report based on the results:
```bash
python generate_report.py
```
This generates the Microsoft Word report: `Fatima_Naeem.docx` with cover page, evaluation tables, and embedded chart figures.

### 6. Launch Web Dashboard
Open the interactive Streamlit RAG dashboard locally:
```bash
streamlit run app.py
```

---

## Experimental Configurations

The system conducts evaluations across the following dimensions:
1. **Chunk Size**: 150 vs 300 vs 600 tokens.
2. **Retrieval**: BM25 (exact keyword match), Cosine Semantic Vector, and Hybrid (RRF fusion with k=60).
3. **Models**: `Qwen2.5-0.5B-Instruct` (fast inference) vs `Qwen2.5-1.5B-Instruct` (better instructions).
4. **Post-Retrieval**: None, Long-Context Reordering ("Lost in the Middle"), and Context Summarization.

The **Champion Model** identified in our experiments is **Configuration 9**:
- **Chunk Size**: 600 tokens (minimizes text fragmentation)
- **Retrieval**: Hybrid BM25 + Semantic (RRF) (maximizes recall)
- **LLM**: Qwen2.5-1.5B-Instruct (highest faithfulness and clarity)
- **Post-Retrieval**: Long-context reordering (mitigates context neglect)
- **Evaluation**: Faithfulness: **0.98**, Answer Relevance: **0.95**.
