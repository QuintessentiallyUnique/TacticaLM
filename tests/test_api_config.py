"""Regression tests for API inference configuration."""

from tacticalm.api.main import get_inference_model_config


def test_get_inference_model_config_reads_model_settings():
    config = {
        "inference": {
            "model": {
                "base": "local/base",
                "adapter": "models/adapters/test",
                "load_in_4bit": False,
            }
        }
    }

    base_model, adapter_path, load_in_4bit = get_inference_model_config(config)

    assert base_model == "local/base"
    assert adapter_path == "models/adapters/test"
    assert load_in_4bit is False
