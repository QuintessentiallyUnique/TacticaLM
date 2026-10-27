"""
Model loading utilities for inference.
"""

import logging

logger = logging.getLogger(__name__)

def load_model_for_inference(base_model: str, adapter_path: str = None, load_in_4bit: bool = True) -> tuple:
    """
    Loads a language model and tokenizer for inference.
    
    Args:
        base_model (str): Path or identifier of the base model.
        adapter_path (str, optional): Path to the LoRA adapter. Defaults to None.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit precision. Defaults to True.
        
    Returns:
        tuple: A tuple containing the loaded (model, tokenizer).
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
    except ImportError as e:
        logger.error(f"Failed to import required libraries for model loading: {e}")
        raise

    logger.info(f"Loading base model: {base_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    if adapter_path:
        logger.info(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        
    model.eval()
    return model, tokenizer
