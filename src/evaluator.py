import re
import time
import torch

class RAGEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def _call_evaluator_llm(self, system_prompt, user_prompt):
        """Helper to invoke the loaded LLM with a low temperature for evaluation."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,  # Deterministic execution
                do_sample=False
            )
        
        response_ids = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    def evaluate_faithfulness(self, context, answer):
        """
        Evaluates the faithfulness of the generated answer relative to the context.
        Returns a score between 0.0 and 1.0 and the raw analysis.
        """
        system_prompt = (
            "You are a strict clinical quality assurance auditor. Your job is to verify if a generated "
            "clinical answer is completely supported by the provided clinical context. You must not allow "
            "any hallucination or extrapolation."
        )
        
        user_prompt = (
            f"CLINICAL CONTEXT:\n{context}\n\n"
            f"GENERATED CLINICAL ANSWER:\n{answer}\n\n"
            f"TASK:\n"
            f"1. Break down the generated answer into individual factual statements or claims.\n"
            f"2. For each claim, determine if it is directly and explicitly supported by the clinical context. "
            f"Mark each as [SUPPORTED] or [NOT SUPPORTED].\n"
            f"3. Calculate the Faithfulness Score as the ratio of supported claims to total claims.\n"
            f"4. Output your response in the exact format shown below:\n\n"
            f"ANALYSIS:\n"
            f"- Claim 1: [text] - [SUPPORTED/NOT SUPPORTED]\n"
            f"- Claim 2: [text] - [SUPPORTED/NOT SUPPORTED]\n"
            f"...\n"
            f"FAITHFULNESS SCORE: [a decimal value between 0.0 and 1.0]"
        )
        
        raw_output = self._call_evaluator_llm(system_prompt, user_prompt)
        
        # Extract score
        match = re.search(r"FAITHFULNESS SCORE:\s*([0-9.]+)", raw_output, re.IGNORECASE)
        score = 1.0
        if match:
            try:
                score = float(match.group(1))
                score = min(max(score, 0.0), 1.0) # bound check
            except ValueError:
                pass
        else:
            # Fallback check if it has [SUPPORTED] vs [NOT SUPPORTED]
            total_claims = len(re.findall(r"Claim \d+", raw_output))
            not_supported = len(re.findall(r"NOT SUPPORTED", raw_output))
            if total_claims > 0:
                score = (total_claims - not_supported) / total_claims
                
        return score, raw_output

    def evaluate_relevance(self, query, answer):
        """
        Evaluates the relevance of the generated answer relative to the user's query.
        Returns a score between 0.0 and 1.0 and the raw analysis.
        """
        system_prompt = (
            "You are an expert medical educator. Your job is to evaluate if a generated clinical answer "
            "directly, fully, and relevance-wise addresses the user's question, without containing "
            "unnecessary details or failing to answer the core query."
        )
        
        user_prompt = (
            f"USER QUESTION:\n{query}\n\n"
            f"GENERATED CLINICAL ANSWER:\n{answer}\n\n"
            f"TASK:\n"
            f"1. Analyze if the generated answer directly addresses the core aspects of the user question.\n"
            f"2. Check if the answer contains irrelevant, redundant, or confusing information.\n"
            f"3. Grade the answer on a scale from 0.0 (completely irrelevant) to 1.0 (perfectly relevant and direct).\n"
            f"4. Output your response in the exact format shown below:\n\n"
            f"EVALUATION:\n"
            f"[Provide brief analysis of relevance]\n"
            f"RELEVANCE SCORE: [a decimal value between 0.0 and 1.0]"
        )
        
        raw_output = self._call_evaluator_llm(system_prompt, user_prompt)
        
        # Extract score
        match = re.search(r"RELEVANCE SCORE:\s*([0-9.]+)", raw_output, re.IGNORECASE)
        score = 1.0
        if match:
            try:
                score = float(match.group(1))
                score = min(max(score, 0.0), 1.0)
            except ValueError:
                pass
                
        return score, raw_output
