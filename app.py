import os
import time
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from src.document_processor import DocumentProcessor
from src.retriever import Retriever
from src.rag_engine import RAGEngine
from src.evaluator import RAGEvaluator

st.set_page_config(
    page_title="Clinical QA System (RAG)",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    .main-title {
        color: #1a73e8;
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0rem;
    }
    .sub-title {
        color: #5f6368;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1a73e8;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #202124;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #5f6368;
    }
    .source-card {
        background-color: #ffffff;
        border: 1px solid #dadce0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .source-header {
        font-weight: bold;
        color: #1a73e8;
        font-size: 0.9rem;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Cache managers to avoid reloading models on every re-run
@st.cache_resource
def get_rag_engine(model_name):
    return RAGEngine(model_name=model_name)

@st.cache_resource
def get_processed_data(chunk_size, overlap):
    doc_proc = DocumentProcessor()
    chunks = doc_proc.process_directory("data", chunk_size=chunk_size, chunk_overlap=overlap)
    retriever = Retriever(chunks)
    return chunks, retriever

# Title Banner
st.markdown("<div class='main-title'>🏥 Clinical RAG-QA Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Interactive evaluation platform for Cardiovascular Disease and Hypertension guidelines</div>", unsafe_allow_html=True)

# Sidebar Configurations
st.sidebar.header("⚙️ System Configurations")

model_choice = st.sidebar.selectbox(
    "1. Choose LLM Model",
    ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"],
    index=0,
    help="Select the local LLM. 1.5B is smarter, 0.5B is faster."
)

chunk_size = st.sidebar.slider(
    "2. Chunk Size (Tokens)",
    min_value=100,
    max_value=800,
    value=300,
    step=50,
    help="Size of the text blocks sent to the retriever."
)

chunk_overlap = st.sidebar.slider(
    "3. Chunk Overlap (Tokens)",
    min_value=10,
    max_value=100,
    value=30,
    step=10,
    help="Overlapping window size between adjacent chunks."
)

retrieval_method = st.sidebar.selectbox(
    "4. Retrieval Strategy",
    ["BM25", "Semantic (Vector)", "Hybrid (BM25 + Semantic)"],
    index=2,
    help="BM25 matches terms exactly. Semantic matches concepts. Hybrid combines both via RRF."
)

post_processing = st.sidebar.selectbox(
    "5. Post-Retrieval Processing",
    ["None", "Long-Context Reordering", "Summarization"],
    index=1,
    help="Lost-in-the-Middle reorders key documents. Summarization distills context first."
)

temperature = st.sidebar.slider(
    "6. Generation Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.1,
    help="Lower values make the model more deterministic and clinical."
)

# Convert labels to code values
ret_method_val = {"BM25": "bm25", "Semantic (Vector)": "semantic", "Hybrid (BM25 + Semantic)": "hybrid"}[retrieval_method]
post_proc_val = {"None": "none", "Long-Context Reordering": "reorder", "Summarization": "summarize"}[post_processing]

# Setup Tabs
tab_qa, tab_eval, tab_corpus, tab_guide = st.tabs(["💬 Ask Questions", "📊 Performance Suite", "📁 Document Corpus", "📋 Reproducibility"])

with tab_qa:
    st.subheader("Query the Clinical Guidelines")
    
    # Ready prompt input
    query = st.text_input("Enter your clinical question:", "What sits at the heart level and is the sitting protocol before measuring blood pressure?")
    
    if st.button("Generate Clinical Response", type="primary"):
        with st.spinner("Executing RAG Pipeline (processing text, retrieving chunks, generating response)..."):
            try:
                # 1. Process data and get retriever
                chunks, retriever = get_processed_data(chunk_size, chunk_overlap)
                
                # 2. Retrieve context
                t_ret_start = time.time()
                retrieved = retriever.retrieve(query, method=ret_method_val, top_k=3)
                ret_time = time.time() - t_ret_start
                
                # 3. Load LLM engine
                rag_engine = get_rag_engine(model_choice)
                
                # 4. Generate answer
                answer, context_used, inf_time = rag_engine.generate_answer(
                    query, retrieved, post_processing=post_proc_val, temperature=temperature
                )
                
                total_time = ret_time + inf_time
                
                # 5. Evaluate response using LLM-as-a-judge
                evaluator = RAGEvaluator(rag_engine.model, rag_engine.tokenizer)
                f_score, f_analysis = evaluator.evaluate_faithfulness(context_used, answer)
                r_score, r_analysis = evaluator.evaluate_relevance(query, answer)
                
                # Layout results
                col_ans, col_metrics = st.columns([2, 1])
                
                with col_ans:
                    st.markdown("### 🩺 Clinical Answer")
                    st.info(answer)
                    
                    st.markdown("### 📄 Retrieved Context Chunks Used")
                    for idx, item in enumerate(retrieved):
                        # item is (chunk, score)
                        chunk = item[0]
                        score = item[1]
                        st.markdown(f"""
                        <div class='source-card'>
                            <div class='source-header'>Source: {chunk['doc_name']} (Score/RRF: {score:.4f})</div>
                            <div style='font-size:0.95rem;'>{chunk['text']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_metrics:
                    st.markdown("### 📊 Performance Analytics")
                    
                    # Latencies
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{total_time:.3f}s</div>
                        <div class='metric-label'>Total System Response Time</div>
                    </div>
                    <div class='metric-card' style='border-left-color: #ea4335;'>
                        <div class='metric-value'>{ret_time * 1000:.1f} ms</div>
                        <div class='metric-label'>Retrieval Time</div>
                    </div>
                    <div class='metric-card' style='border-left-color: #fbbc05;'>
                        <div class='metric-value'>{inf_time:.3f}s</div>
                        <div class='metric-label'>LLM Inference Time</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # QA scores
                    st.markdown("### 🎯 Quality Evaluation")
                    st.markdown(f"""
                    <div class='metric-card' style='border-left-color: #34a853;'>
                        <div class='metric-value'>{f_score:.2f} / 1.0</div>
                        <div class='metric-label'>Faithfulness Score (Grounding)</div>
                    </div>
                    <div class='metric-card' style='border-left-color: #9334e8;'>
                        <div class='metric-value'>{r_score:.2f} / 1.0</div>
                        <div class='metric-label'>Answer Relevance Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("Show Detailed Judge Reasoning"):
                        st.markdown("**Faithfulness Audit:**")
                        st.code(f_analysis)
                        st.markdown("**Relevance Audit:**")
                        st.code(r_analysis)
                        
            except Exception as e:
                st.error(f"Error executing RAG Pipeline: {e}")
                st.exception(e)

with tab_eval:
    st.subheader("Benchmark Comparison Suite")
    st.write(
        "A full performance comparison of the 9 configurations tested in the report is shown below. "
        "These include variations in chunk sizes, retrieval methods, models, and reordering techniques."
    )
    
    # Load results
    if os.path.exists("data/results.csv"):
        results_df = pd.read_csv("data/results.csv")
        st.dataframe(results_df, use_container_width=True)
    else:
        st.warning("Experiment results have not been generated yet. Showing baseline evaluation data.")
        st.dataframe(pd.DataFrame(BASELINE_RESULTS), use_container_width=True)

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        if os.path.exists("data/metrics_comparison.png"):
            st.image("data/metrics_comparison.png", caption="Grounding Quality comparison", use_container_width=True)
        else:
            st.info("Metrics plot not generated. Run `python run_experiments.py` to plot actual metrics.")
            
    with col_img2:
        if os.path.exists("data/latency_comparison.png"):
            st.image("data/latency_comparison.png", caption="System latencies comparison (seconds)", use_container_width=True)
        else:
            st.info("Latency plot not generated. Run `python run_experiments.py` to plot actual latencies.")

with tab_corpus:
    st.subheader("Clinical Domain Corpus files")
    st.write("Below are the reference guidelines text documents processed by the document processor:")
    
    files = ["cardiovascular_guidelines_diagnosis.txt", "cardiovascular_guidelines_pharmacology.txt", "cardiovascular_guidelines_lifestyle.txt"]
    for fname in files:
        fpath = os.path.join("data", fname)
        if os.path.exists(fpath):
            with st.expander(f"📄 {fname} ({os.path.getsize(fpath)} bytes)"):
                with open(fpath, "r", encoding="utf-8") as f:
                    st.code(f.read())
        else:
            st.info(f"File {fname} is not initialized.")

with tab_guide:
    st.subheader("How to Reproduce this System")
    st.markdown("""
    To replicate this entire system (including code, evaluation benchmarks, and the final word document report):
    
    1. **Install Virtual Environment:**
       ```bash
       python -m venv venv
       .\\venv\\Scripts\\activate
       ```
       
    2. **Install Required Libraries:**
       ```bash
       pip install -r requirements.txt
       ```
       
    3. **Run Experiments & Benchmarks:**
       ```bash
       python run_experiments.py
       ```
       This will query the LLM to run baseline benchmarks, generate results under `data/results.csv` and output comparison plots.
       
    4. **Generate Microsoft Word Report:**
       ```bash
       python generate_report.py
       ```
       This builds the final, formatted Word report: `Fatima_Naeem.docx`.
       
    5. **Launch interactive Streamlit UI:**
       ```bash
       streamlit run app.py
       ```
    """)
