import os
# Force Transformers to use PyTorch ONLY and ignore (broken) TensorFlow
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time

# The model you validated (Targeting LOCAL FOLDER)
# Switch to Qwen 2.5 1.5B (Edge Optimized)
# This fits easily in RAM (4GB) and is faster/smarter for logic.
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

def test_model():
    print(f"\n--- 6G AI Test: Loading {MODEL_NAME} ---\n")
    
    # 1. Hardware Check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware Detected: {device.upper()}")
    
    if device == "cpu":
        print("WARNING: Running on CPU. This will be slow (Expect 10-30s).")
        # Try float16 to save memory (6GB vs 12GB). Might be slower/unsupported on old CPUs.
        dtype = torch.float16 
    else:
        print("Running on GPU. This should be fast (Expect <2s).")
        dtype = torch.float16 

    try:
        # 2. Load Tokenizer
        print(" [1/3] Loading Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # 3. Load Model with DISK OFFLOAD (Prevents OOM)
        print(" [2/3] Loading Model (Approx 6GB)...")
        if device == "cpu":
            print("       Enabled Disk Offloading (Save RAM)...")
        
        load_start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            device_map="auto", # Smart dispatch
            offload_folder="offload", # Spill to disk if RAM is full
            low_cpu_mem_usage=True,
        )
        load_time = time.time() - load_start
        print(f"       Model Loaded in {load_time:.2f} seconds.")
        
        # 4. Create Pipeline
        print(" [3/3] Creating Inference Pipeline...")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=50,
            temperature=0.1,
        )
        print(">>> READY. Starting Inference Test... <<<\n")
        
        # 5. Run Inference
        # Simulation: Battery Critical, Traffic Normal
        system_msg = """You are a 6G Network Controller. 
Output Format: ONLY 3 numbers (Latency, Energy, Reliability) 0-10.
Examples:
User: Battery Critical. -> AI: 2, 10, 5
User: High Traffic. -> AI: 9, 2, 4"""
        
        user_msg = "Current Status: Battery 5% (Critical), Traffic Low. Assign priorities."
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        
        print(f"PROMPT: {user_msg}")
        print("Thinking...")
        
        # --- TIMER START ---
        inference_start = time.time()
        
        outputs = pipe(
            messages,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        # --- TIMER END ---
        inference_end = time.time()
        duration = inference_end - inference_start
        
        # 6. Show Result
        response = outputs[0]["generated_text"][-1]["content"]
        
        print("-" * 40)
        print(f"[AI Response]: {response}")
        print("-" * 40)
        
        # Regex Check
        import re
        matches = re.findall(r'\d+', response)
        if len(matches) >= 3:
            print(f"[Parsed Weights]: Lat={matches[0]}, Eng={matches[1]}, Rel={matches[2]}")
        else:
            print("[Parsed Weights]: FAILED to find 3 numbers.")
            
        print(f"[Inference Time]: {duration:.4f} seconds")
        print("-" * 40)
        
        # Verdict logic
        if duration > 5.0:
            print("[NOTE]: Response is slow (CPU). But logic works.")
        else:
            print("[NOTE]: Response speed is excellent.")

    except Exception as e:
        print(f"\n[CRITICAL FAIL]: {e}")
        print("Tip: If you ran out of memory, try closing Chrome/VS Code and run again.")

if __name__ == "__main__":
    test_model()