"""Training script entrypoint for TacticaLM.

Usage:
    py -3.12 scripts/run_training.py --data data/evaluation/train_pilot.json --output models/adapters/tacticalm-v1
"""

import argparse
import sys
import logging
from pathlib import Path

from tacticalm.training.train import run_training
from tacticalm.training.tracking import setup_mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run QLoRA fine-tuning for TacticaLM")
    parser.add_argument("--data", type=str, required=True, help="Path to training dataset JSON")
    parser.add_argument("--output", type=str, required=True, help="Path to save the adapter")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="Base model name")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Per device batch size")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Training data not found at {data_path}")
        return 1
        
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_mlflow("tacticalm-qlora")
    
    try:
        run_training(
            train_data_path=data_path,
            output_dir=output_dir,
            model_name=args.model,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr
        )
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
