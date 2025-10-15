import os
import logging
from typing import Dict
from pydantic import BaseModel, Field, field_validator
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Configuration class for Gemini model settings."""

    web_search_model: str = Field(
        default="gemini-flash-latest",
        description="Model to use for web search functionality",
    )
    default_model: str = Field(
        default="gemini-flash-lite-latest", description="Default model for general use"
    )
    advanced_model: str = Field(
        default="gemini-2.5-pro",
        description="Advanced model for complex tasks",
    )

    @field_validator("web_search_model")
    @classmethod
    def validate_web_search_model(cls, v):
        """Validate web search model name."""
        if not v.startswith("gemini-"):
            logger.warning(f"Invalid web search model format: {v}")
            raise ValueError(f"Invalid model format: {v}. Must start with 'gemini-'")
        return v

    @field_validator("default_model")
    @classmethod
    def validate_default_model(cls, v):
        """Validate default model name."""
        if not v.startswith("gemini-"):
            logger.warning(f"Invalid default model format: {v}")
            raise ValueError(f"Invalid model format: {v}. Must start with 'gemini-'")
        return v

    @field_validator("advanced_model")
    @classmethod
    def validate_advanced_model(cls, v):
        """Validate advanced model name."""
        if not v.startswith("gemini-"):
            logger.warning(f"Invalid advanced model format: {v}")
            raise ValueError(f"Invalid model format: {v}. Must start with 'gemini-'")
        return v


@lru_cache(maxsize=1)
def get_config() -> ModelConfig:
    """Load and cache configuration from environment variables."""
    try:
        config = ModelConfig(
            web_search_model=_get_env_with_default(
                "GEMINI_WEB_SEARCH_MODEL", "gemini-flash-latest"
            ),
            default_model=_get_env_with_default(
                "GEMINI_DEFAULT_MODEL", "gemini-flash-lite-latest"
            ),
            advanced_model=_get_env_with_default(
                "GEMINI_ADVANCED_MODEL", "gemini-2.5-pro"
            ),
        )
        logger.info("Configuration loaded successfully")
        return config
    except Exception as e:
        logger.warning(f"Error loading configuration: {e}. Using defaults.")
        return ModelConfig()


def clear_config_cache():
    """Clear the configuration cache to force reload."""
    get_config.cache_clear()


def _get_env_with_default(env_var: str, default: str) -> str:
    """Get environment variable with default value."""
    value = os.getenv(env_var, default)
    if value != default:
        logger.info(f"Using custom {env_var}: {value}")
    return value


def get_model_for_web_search() -> str:
    """Get the configured model for web search."""
    config = get_config()
    return config.web_search_model


def get_default_model() -> str:
    """Get the configured default model."""
    config = get_config()
    return config.default_model


def get_advanced_model() -> str:
    """Get the configured advanced model."""
    config = get_config()
    return config.advanced_model


def get_all_models() -> Dict[str, str]:
    """Get all configured models as a dictionary."""
    config = get_config()
    return {
        "web_search": config.web_search_model,
        "default": config.default_model,
        "advanced": config.advanced_model,
    }
