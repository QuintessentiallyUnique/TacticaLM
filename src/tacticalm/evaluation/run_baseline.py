"""Run base model inference on evaluation examples."""

import time
import torch
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from tacticalm.dataset.schemas import EvaluationExample


def load_base_model(model_name: str, load_in_4bit: bool = True) -> Tuple[Any, Any]:
    """Load the base model and tokenizer with BitsAndBytes optimization."""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    ) if load_in_4bit else None
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )
    return model, tokenizer


def run_inference(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 400, temperature: float = 0.1) -> str:
    """Run inference on the given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    do_sample = temperature > 0.0
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature if do_sample else None,
        do_sample=do_sample
    )
    
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text.strip()


def run_baseline_evaluation(
    eval_examples: List[EvaluationExample],
    model: Any,
    tokenizer: Any,
    prompt_formatter: Callable[[EvaluationExample], str],
    output_dir: Path
) -> List[Dict[str, Any]]:
    """Run inference on all examples, saves outputs with metadata."""
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for example in eval_examples:
        prompt = prompt_formatter(example)
        
        start_time = time.time()
        generated_answer = run_inference(model, tokenizer, prompt)
        generation_time = time.time() - start_time
        
        result = {
            "example_id": example.example_id,
            "question": example.question,
            "generated_answer": generated_answer,
            "reference_answer": example.reference_answer,
            "evidence": example.evidence,
            "category": example.category.value,
            "generation_time_seconds": generation_time
        }
        results.append(result)
        
    output_file = output_dir / "baseline_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
        
    return results
