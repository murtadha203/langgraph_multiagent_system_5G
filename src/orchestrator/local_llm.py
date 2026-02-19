# src/orchestrator/local_llm.py

import os
# Force Transformers to use PyTorch ONLY and ignore (broken) TensorFlow
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class LocalTeleLLMEngine:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", device="auto"):
        """
        Loads the specialized Tele-LLM (Llama 3.2 3B) from LOCAL FOLDER.
        """
        print(f"Loading Tele-LLM from: {model_name}...")
        
        # Determine device (Force CUDA if available, else CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running on: {self.device.upper()}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model in half-precision (float16) to save RAM (6GB vs 12GB)
            # Modern PyTorch supports float16 on CPU (slow but works)
            dtype = torch.float16
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                offload_folder="offload",
                low_cpu_mem_usage=True,
            )
            
            # Create a text-generation pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=20, # We only need a short output (Mode Name)
                temperature=0.1,   # Low temp = Deterministic/Safety
                do_sample=True,
            )
            
            print(">>> Tele-LLM 3B Loaded Successfully. Ready for 6G Control.")
            
        except Exception as e:
            print(f"CRITICAL ERROR loading Tele-LLM: {e}")
            raise e

    def predict(self, system_prompt, user_prompt):
        """
        Runs the LLM pipeline.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        outputs = self.pipe(
            messages,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return outputs[0]["generated_text"][-1]["content"]

    def predict_weights(self, system_prompt, user_prompt):
        """
        Returns normalized weights (alpha, beta, gamma).
        """
        # Get raw text response (e.g., "8, 2, 5")
        response_text = self.predict(system_prompt, user_prompt)
        
        try:
            import re
            # 1. Regex Parsing (The Bulletproof Vest)
            # Find any sequence of 3 numbers separated by anything (comma, space, words)
            # Looks for: digit...digit...digit
            matches = re.findall(r'\d+', response_text)
            
            if len(matches) >= 3:
                # Take the first 3 numbers found
                parts = [float(x) for x in matches[:3]]
            else:
                print(f"LLM Regex Fail: Found {matches}. Defaulting.")
                return 0.33, 0.33, 0.34
            
            # 3. Clip to 0-10 range just in case
            l = min(max(parts[0], 0), 10)
            e = min(max(parts[1], 0), 10)
            r = min(max(parts[2], 0), 10)
            
            # 4. Normalize (Ensure sum is 1.0)
            total = l + e + r
            if total == 0: return 0.33, 0.33, 0.34
            
            alpha = l / total  # Latency Weight
            beta = e / total   # Energy Weight
            gamma = r / total  # Reliability Weight
            
            return alpha, beta, gamma

        except Exception as e:
            print(f"LLM Critical Fail: {e}")
            return 0.33, 0.33, 0.34


# Quick Test
if __name__ == "__main__":
    ai = LocalTeleLLMEngine()
    print("Test Response:", ai.predict("You are a network controller.", "Traffic is congested."))