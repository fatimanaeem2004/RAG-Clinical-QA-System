import os
import re

class DocumentProcessor:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def load_documents(self, directory_path):
        """Loads all text files from a directory."""
        documents = {}
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory {directory_path} does not exist.")
        
        for filename in os.listdir(directory_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(directory_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    documents[filename] = f.read()
        return documents

    def count_tokens(self, text):
        """Counts tokens in a text snippet. Falls back to word counting if no tokenizer is present."""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        # Fallback: estimate 1.3 tokens per word on average
        words = len(re.findall(r'\w+', text))
        return int(words * 1.3)

    def split_into_chunks(self, text, doc_name, chunk_size=300, chunk_overlap=30):
        """
        Splits text into chunks of roughly `chunk_size` tokens with `chunk_overlap` tokens overlap.
        Uses overlapping word windows to avoid tiktoken dependencies during raw processing,
        but validates sizes using count_tokens.
        """
        words = text.split()
        chunks = []
        
        # Approximate words per chunk
        # Assuming 1 word ≈ 1.3 tokens, words_per_chunk = chunk_size / 1.3
        words_per_chunk = int(chunk_size / 1.3)
        words_overlap = int(chunk_overlap / 1.3)
        
        if words_per_chunk <= 0:
            words_per_chunk = 50
        if words_overlap >= words_per_chunk:
            words_overlap = words_per_chunk // 2

        idx = 0
        chunk_id = 0
        while idx < len(words):
            end_idx = min(idx + words_per_chunk, len(words))
            chunk_words = words[idx:end_idx]
            chunk_text = " ".join(chunk_words)
            
            # Recalculate actual tokens
            token_count = self.count_tokens(chunk_text)
            
            chunks.append({
                "chunk_id": f"{doc_name}_chunk_{chunk_id}",
                "doc_name": doc_name,
                "text": chunk_text,
                "token_count": token_count
            })
            
            chunk_id += 1
            if end_idx == len(words):
                break
            idx += (words_per_chunk - words_overlap)
            
        return chunks

    def process_directory(self, directory_path, chunk_size=300, chunk_overlap=30):
        """Loads and chunks all documents in a directory."""
        documents = self.load_documents(directory_path)
        all_chunks = []
        for doc_name, content in documents.items():
            chunks = self.split_into_chunks(content, doc_name, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
        return all_chunks
