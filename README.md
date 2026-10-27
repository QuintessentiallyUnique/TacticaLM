# TacticaLM: A Fine-Tuned Football Tactical Analysis Model

## Research Question
How can we effectively fine-tune a Large Language Model to perform high-quality, evidence-based football tactical analysis from raw event data?

## Architecture Overview
TacticaLM leverages QLoRA to fine-tune the Qwen/Qwen3-4B-Instruct-2507 model on StatsBomb open data. The system extracts tactical features (e.g., progressive passes, pressing actions) and generates structured instructions to train the model to act as an expert tactical analyst.

## Status
Under Development

## Tech Stack
- **Language**: Python 3.11
- **ML Frameworks**: PyTorch, HuggingFace Transformers, PEFT, TRL
- **Data Processing**: Pandas, NumPy, Statsbombpy
- **Serving**: FastAPI, Streamlit
- **Experiment Tracking**: MLflow

## License
Apache 2.0 License. See [LICENSE](LICENSE) for details.
