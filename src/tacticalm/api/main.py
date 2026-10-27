"""
Main FastAPI application setup.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tacticalm.api import routes
from tacticalm.api.routes import router
from tacticalm.utils.config import get_inference_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_inference_model_config(
    config: dict[str, Any] | None = None,
) -> tuple[str, str | None, bool]:
    """Return base model, adapter path, and quantization setting from inference config."""
    config_data = config if config is not None else get_inference_config()
    inference_config = config_data.get("inference", {})
    model_config = inference_config.get("model", {})

    base_model_path = model_config.get("base", "Qwen/Qwen3-4B-Instruct-2507")
    adapter_path = model_config.get("adapter") or None
    load_in_4bit = bool(model_config.get("load_in_4bit", True))

    return base_model_path, adapter_path, load_in_4bit


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for the application.
    Loads the model during startup.
    """
    logger.info("Starting up FastAPI application...")
    try:
        from tacticalm.inference.generate import TacticalGenerator
        from tacticalm.inference.model_loader import load_model_for_inference

        base_model_path, adapter_path, load_in_4bit = get_inference_model_config()

        logger.info("Loading model and tokenizer...")
        model, tokenizer = load_model_for_inference(
            base_model_path,
            adapter_path,
            load_in_4bit=load_in_4bit,
        )
        routes._generator = TacticalGenerator(model, tokenizer)
        logger.info("Model loaded successfully.")

    except ImportError as e:
        logger.warning(f"Could not import inference dependencies: {e}. API will start in degraded mode.")
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error during startup: {e}")

    yield

    logger.info("Shutting down FastAPI application...")

app = FastAPI(
    title="TacticaLM API",
    description="API for football tactical analysis using TacticaLM.",
    version="0.1.0",
    lifespan=lifespan
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
