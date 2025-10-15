import pytest
import os
from unittest.mock import AsyncMock, patch
from gemini_mcp.tools import web_search, use_gemini
from gemini_mcp.config import (
    get_config,
    get_model_for_web_search,
    get_default_model,
    get_advanced_model,
)


class TestNewModelConfiguration:
    """Test cases for new model configuration functionality."""

    def setup_method(self):
        """Clear cache before each test."""
        get_config.cache_clear()

    def teardown_method(self):
        """Clear cache after each test."""
        get_config.cache_clear()

    def test_default_model_values(self):
        """Test that default model values match original hardcoded values."""
        # Clear any environment variables that might be set
        for key in [
            "GEMINI_WEB_SEARCH_MODEL",
            "GEMINI_DEFAULT_MODEL",
            "GEMINI_ADVANCED_MODEL",
        ]:
            if key in os.environ:
                del os.environ[key]

        get_config.cache_clear()

        config = get_config()

        # Verify default values match the original hardcoded values
        assert config.web_search_model == "gemini-flash-latest"
        assert config.default_model == "gemini-flash-lite-latest"
        assert config.advanced_model == "gemini-2.5-pro"

    @patch.dict(os.environ, {"GEMINI_WEB_SEARCH_MODEL": "gemini-2.5-pro"})
    def test_custom_web_search_model(self):
        """Test custom web search model configuration."""
        get_config.cache_clear()
        model = get_model_for_web_search()
        assert model == "gemini-2.5-pro"

    @patch.dict(os.environ, {"GEMINI_DEFAULT_MODEL": "gemini-flash-lite-latest"})
    def test_custom_default_model(self):
        """Test custom default model configuration."""
        get_config.cache_clear()
        model = get_default_model()
        assert model == "gemini-flash-lite-latest"

    @patch.dict(os.environ, {"GEMINI_ADVANCED_MODEL": "gemini-2.5-pro"})
    def test_custom_advanced_model(self):
        """Test custom advanced model configuration."""
        get_config.cache_clear()
        model = get_advanced_model()
        assert model == "gemini-2.5-pro"

    @patch.dict(
        os.environ,
        {
            "GEMINI_WEB_SEARCH_MODEL": "gemini-2.5-pro",
            "GEMINI_DEFAULT_MODEL": "gemini-flash-lite-latest",
            "GEMINI_ADVANCED_MODEL": "gemini-2.5-pro",
        },
    )
    def test_all_custom_models(self):
        """Test all custom model configurations together."""
        get_config.cache_clear()
        config = get_config()

        assert config.web_search_model == "gemini-2.5-pro"
        assert config.default_model == "gemini-flash-lite-latest"
        assert config.advanced_model == "gemini-2.5-pro"

    @pytest.mark.asyncio
    @patch("gemini_mcp.tools.get_gemini_client")
    async def test_web_search_with_custom_model_integration(
        self, mock_get_gemini_client
    ):
        """Test web_search integration with custom model configuration."""
        # Set environment variable and clear cache
        os.environ["GEMINI_WEB_SEARCH_MODEL"] = "gemini-2.5-pro"
        get_config.cache_clear()

        # Setup mocks
        mock_client = AsyncMock()
        mock_get_gemini_client.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.text = "Custom model search result"
        mock_response.candidates = [AsyncMock()]
        mock_response.candidates[0].grounding_metadata = None
        mock_client.aio.models.generate_content.return_value = mock_response

        # Call the function
        await web_search(query="test query")

        # Verify the custom model was used
        mock_client.aio.models.generate_content.assert_called_once()
        call_args = mock_client.aio.models.generate_content.call_args
        assert call_args[1]["model"] == "gemini-2.5-pro"

    @pytest.mark.asyncio
    @patch("gemini_mcp.tools.get_gemini_client")
    async def test_use_gemini_with_custom_default_model_integration(
        self, mock_get_gemini_client
    ):
        """Test use_gemini integration with custom default model configuration."""
        # Set environment variable and clear cache
        os.environ["GEMINI_DEFAULT_MODEL"] = "gemini-flash-lite-latest"
        get_config.cache_clear()

        # Setup mocks
        mock_client = AsyncMock()
        mock_get_gemini_client.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.text = "Custom default model response"
        mock_client.aio.models.generate_content.return_value = mock_response

        # Call the function without specifying model
        await use_gemini(prompt="test prompt")

        # Verify the custom default model was used
        mock_client.aio.models.generate_content.assert_called_once()
        call_args = mock_client.aio.models.generate_content.call_args
        assert call_args[1]["model"] == "gemini-flash-lite-latest"

    @pytest.mark.asyncio
    @patch("gemini_mcp.tools.get_gemini_client")
    async def test_use_gemini_with_custom_advanced_model_integration(
        self, mock_get_gemini_client
    ):
        """Test use_gemini integration with custom advanced model configuration."""
        # Set environment variable and clear cache
        os.environ["GEMINI_ADVANCED_MODEL"] = "gemini-2.5-pro"
        get_config.cache_clear()

        # Setup mocks
        mock_client = AsyncMock()
        mock_get_gemini_client.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.text = "Custom advanced model response"
        mock_client.aio.models.generate_content.return_value = mock_response

        # Call the function with pro model
        await use_gemini(prompt="test prompt", model="gemini-2.5-pro")

        # Verify the custom advanced model was used
        mock_client.aio.models.generate_content.assert_called_once()
        call_args = mock_client.aio.models.generate_content.call_args
        assert call_args[1]["model"] == "gemini-2.5-pro"

    def test_backward_compatibility_no_env_vars(self):
        """Test that backward compatibility is maintained when no environment variables are set."""
        # Ensure no env vars are set
        for key in [
            "GEMINI_WEB_SEARCH_MODEL",
            "GEMINI_DEFAULT_MODEL",
            "GEMINI_ADVANCED_MODEL",
        ]:
            if key in os.environ:
                del os.environ[key]

        get_config.cache_clear()

        # Verify default values are used
        config = get_config()
        assert config.web_search_model == "gemini-flash-latest"
        assert config.default_model == "gemini-flash-lite-latest"
        assert config.advanced_model == "gemini-2.5-pro"

    def test_invalid_environment_variables_fallback(self):
        """Test that invalid environment variables fall back to defaults."""
        # Set invalid environment variables
        os.environ["GEMINI_WEB_SEARCH_MODEL"] = "invalid-model"
        os.environ["GEMINI_DEFAULT_MODEL"] = "another-invalid-model"
        os.environ["GEMINI_ADVANCED_MODEL"] = "yet-another-invalid-model"

        get_config.cache_clear()

        # Verify defaults are used
        config = get_config()
        assert config.web_search_model == "gemini-flash-latest"
        assert config.default_model == "gemini-flash-lite-latest"
        assert config.advanced_model == "gemini-2.5-pro"
