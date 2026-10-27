"""
Regenerates responses in train_pilot.json and dev_pilot.json using the base Qwen3-4B-Instruct model
to ensure they are natural language paragraphs instead of bullet lists.
"""

import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def main():
    base_model_id = "Qwen/Qwen3-4B-Instruct-2507"
    print(f"Loading base model {base_model_id}...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        quantization_config=quantization_config
    )
    
    system_prompt = """You are an expert football tactical analyst.
Your task is to analyze matches using only the provided statistical evidence.

Rules:
1. ONLY reference facts from the provided evidence. Do not hallucinate or use outside knowledge.
2. Cite specific numbers and statistics when making claims.
3. If the provided evidence is insufficient to answer the question, explicitly acknowledge this limitation.
4. Use appropriate football tactical terminology correctly.
5. Structure your analysis with clear reasoning and logic.
6. Write your analysis in continuous natural language paragraphs. DO NOT repeat the evidence back as a bulleted list. You must provide a cohesive, analytical narrative."""

    files_to_process = [
        ("data/evaluation/train_pilot.json", "response"),
        ("data/evaluation/dev_pilot.json", "reference_answer")
    ]
    
    for file_path, response_field in files_to_process:
        path = Path(file_path)
        if not path.exists():
            print(f"File {file_path} not found.")
            continue
            
        print(f"Processing {file_path}...")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for i, example in enumerate(data):
            print(f"Generating for {file_path} example {i+1}/{len(data)}...")
            
            if "question" in example:
                question = example["question"]
            else:
                instruction = example["instruction"]
                q_start = instruction.rfind("Question: ") + len("Question: ")
                q_end = instruction.rfind("\n\nAnswer:")
                question = instruction[q_start:q_end]
                
            evidence = example["evidence"]
            
            if not evidence:
                ans = "There is no statistical evidence available to answer this question."
            elif example["category"] == "limitation_aware":
                ans = "Based on the provided statistical evidence, there is insufficient data to answer this question. The data only includes tactical metrics (passes, shots, defensive actions, etc.) and does not contain information regarding formations, player intent, crowd atmosphere, or managerial instructions."
            else:
                context = example.get("context", {})
                context_str = (
                    f"Match Context:\n"
                    f"Competition: {context.get('competition', 'Unknown')}\n"
                    f"Season: {context.get('season', 'Unknown')}\n"
                    f"Date: {context.get('match_date', 'Unknown')}\n"
                    f"Match: {context.get('home_team', 'Unknown')} vs {context.get('away_team', 'Unknown')}\n"
                    f"Score: {context.get('score', 'Unknown')}"
                )
                evidence_str = "Evidence:\n" + "\n".join(f"- {e}" for e in evidence)
                
                base_prompt = f"{context_str}\n\n{evidence_str}\n\nQuestion: {question}\n\nAnswer:"
                full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{base_prompt}<|im_end|>\n<|im_start|>assistant\n"
                
                inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=400,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                input_length = inputs.input_ids.shape[1]
                response_tokens = outputs[0][input_length:]
                ans = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            
            example[response_field] = ans
            if "instruction" in example:
                example["instruction"] = f"{system_prompt}\n\n{base_prompt}"
            
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
        print(f"Finished updating {file_path}")

if __name__ == "__main__":
    main()
