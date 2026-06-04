import numpy as np
import torch
from rank_bm25 import BM25Okapi

class Retriever:
    def __init__(self, chunks, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.chunks = chunks
        self.embedding_model_name = embedding_model_name
        self.tokenizer = None
        self.model = None
        self.embeddings = None
        
        # Tokenize corpus for BM25
        self.tokenized_corpus = [chunk["text"].lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Initialize embeddings
        self._init_embedding_model()
        self._build_vector_index()

    def _init_embedding_model(self):
        """Initializes the embedding model using HuggingFace Transformers directly for reliability."""
        from transformers import AutoTokenizer, AutoModel
        print(f"Loading embedding model: {self.embedding_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        self.model = AutoModel.from_pretrained(self.embedding_model_name)
        self.model.eval()

    def _get_embedding(self, text):
        """Generates embedding vector for a single text using mean pooling."""
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean Pooling
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask
        # Normalize
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.numpy()[0]

    def _build_vector_index(self):
        """Generates embeddings for all chunks in the corpus."""
        print(f"Generating embeddings for {len(self.chunks)} chunks...")
        embeddings_list = []
        for chunk in self.chunks:
            emb = self._get_embedding(chunk["text"])
            embeddings_list.append(emb)
        self.embeddings = np.array(embeddings_list)

    def retrieve_bm25(self, query, top_k=3):
        """Retrieves top_k chunks using BM25 keyword search."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0: # Only return matching items
                results.append((self.chunks[idx], float(scores[idx])))
        return results

    def retrieve_semantic(self, query, top_k=3):
        """Retrieves top_k chunks using Vector similarity (cosine similarity)."""
        query_emb = self._get_embedding(query)
        # Cosine similarity since both vectors are normalized
        similarities = np.dot(self.embeddings, query_emb)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(similarities[idx])))
        return results

    def retrieve_hybrid(self, query, top_k=3, k_rrf=60):
        """
        Retrieves top_k chunks using Reciprocal Rank Fusion (RRF)
        combining BM25 and Semantic search results.
        """
        # Retrieve a larger pool from both methods to fuse them
        pool_size = max(top_k * 3, 10)
        bm25_res = self.retrieve_bm25(query, top_k=pool_size)
        semantic_res = self.retrieve_semantic(query, top_k=pool_size)
        
        # Calculate RRF scores
        rrf_scores = {}
        
        # Helper to index by chunk_id
        chunk_map = {chunk["chunk_id"]: chunk for chunk in self.chunks}
        
        # Process BM25 ranks
        for rank, (chunk, _) in enumerate(bm25_res):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_rrf + (rank + 1)))
            
        # Process Semantic ranks
        for rank, (chunk, _) in enumerate(semantic_res):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_rrf + (rank + 1)))
            
        # Sort by RRF score
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for cid, rrf_score in sorted_rrf:
            results.append((chunk_map[cid], float(rrf_score)))
            
        return results

    def retrieve(self, query, method="hybrid", top_k=3):
        """Generic retrieve method wrapper."""
        if method == "bm25":
            return self.retrieve_bm25(query, top_k)
        elif method == "semantic":
            return self.retrieve_semantic(query, top_k)
        elif method == "hybrid":
            return self.retrieve_hybrid(query, top_k)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
