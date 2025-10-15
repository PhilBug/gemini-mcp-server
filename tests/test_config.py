import pytest
import os
from unittest.mock import patch
from gemini_mcp.config import (
    ModelConfig,
    get_config,
    get_model_for_web_search,
    get_default_model,
    get_advanced_model,
    get_all_models,
)


class TestModelConfig:
    """Test cases for ModelConfig validation logic."""

    def test_valid_model_names(self):
        """Test that valid model names are accepted."""
        valid_models = [
            "gemini-flash-latest",
            "gemini-flash-lite-latest",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ]

        for model in valid_models:
            config = ModelConfig(
                web_search_model=model, default_model=model, advanced_model=model
            )
            assert config.web_search_model == model
            assert config.default_model == model
            assert config.advanced_model == model

    def test_invalid_model_names(self):
        """Test that invalid model names raise ValueError."""
        invalid_models = [
            "2.0-flash",
            "gpt-4",
            "claude-3",
            "invalid-model",
            "",
            "gemini",
            "Gemini-flash-latest",  # Capital G
        ]

        for model in invalid_models:
            with pytest.raises(
                ValueError,
                match=f"Invalid model format: {model}. Must start with 'gemini-'",
            ):
                ModelConfig(
                    web_search_model=model, default_model=model, advanced_model=model
                )

    def test_mixed_valid_invalid_models(self):
        """Test configuration with mixed valid and invalid models."""
        # Valid web_search and default, but invalid advanced
        with pytest.raises(
            ValueError,
            match="Invalid model format: invalid-model. Must start with 'gemini-'",
        ):
            ModelConfig(
                web_search_model="gemini-flash-latest",
                default_model="gemini-2.5-pro",
                advanced_model="invalid-model",
            )

    def test_partial_validation(self):
        """Test that validation stops at first error."""
        # This should fail on web_search_model validation
        with pytest.raises(
            ValueError,
            match="Invalid model format: bad-model. Must start with 'gemini-'",
        ):
            ModelConfig(
                web_search_model="bad-model",
                default_model="gemini-flash-latest",
                advanced_model="gemini-2.5-pro",
            )

    def test_default_values(self):
        """Test that default values are applied when not specified."""
        config = ModelConfig()
        assert config.web_search_model == "gemini-flash-latest"
        assert config.default_model == "gemini-flash-lite-latest"
        assert config.advanced_model == "gemini-2.5-pro"

    def test_partial_specification(self):
        """Test that partial specification uses defaults for unspecified fields."""
        config = ModelConfig(web_search_model="gemini-2.5-pro")
        assert config.web_search_model == "gemini-2.5-pro"
        assert config.default_model == "gemini-flash-lite-latest"  # Default
        assert config.advanced_model == "gemini-2.5-pro"  # Default


class TestEnvironmentVariableParsing:
    """Test cases for environment variable parsing."""

    def setup_method(self):
        """Save original environment variables."""
        self.original_env = os.environ.copy()

    def teardown_method(self):
        """Restore original environment variables."""
        os.environ.clear()
        os.environ.update(self.original_env)
        # Clear the cache after each test
        get_config.cache_clear()

    @patch("gemini_mcp.config.logger")
    def test_valid_environment_variables(self, mock_logger):
        """Test parsing with valid environment variables."""
        os.environ["GEMINI_WEB_SEARCH_MODEL"] = "gemini-2.5-pro"
        os.environ["GEMINI_DEFAULT_MODEL"] = "gemini-1.5-pro"  # Different from default
        os.environ["GEMINI_ADVANCED_MODEL"] = "gemini-2.5-pro"

        config = get_config()

        assert config.web_search_model == "gemini-2.5-pro"
        assert config.default_model == "gemini-1.5-pro"
        assert config.advanced_model == "gemini-2.5-pro"

        # Check that logger was called for custom values
        custom_calls = [
            call
            for call in mock_logger.info.call_args_list
            if "Using custom" in str(call)
        ]

        # Should have 2 custom calls (web_search and default are different from defaults)
        assert len(custom_calls) == 2

        # Check each custom call
        custom_values = [call[0][0] for call in custom_calls]
        assert any(
            "Using custom GEMINI_WEB_SEARCH_MODEL: gemini-2.5-pro" in call
            for call in custom_values
        )
        assert any(
            "Using custom GEMINI_DEFAULT_MODEL: gemini-1.5-pro" in call
            for call in custom_values
        )

    @patch("gemini_mcp.config.logger")
    def test_partial_environment_variables(self, mock_logger):
        """Test parsing with partial environment variables."""
        os.environ["GEMINI_WEB_SEARCH_MODEL"] = "gemini-2.5-pro"
        # Don't set default and advanced - should use defaults

        config = get_config()

        assert config.web_search_model == "gemini-2.5-pro"
        assert config.default_model == "gemini-flash-lite-latest"  # Default
        assert config.advanced_model == "gemini-2.5-pro"  # Default

        # Check that logger was called only for the custom value
        custom_calls = [
            call
            for call in mock_logger.info.call_args_list
            if "Using custom" in str(call)
        ]
        assert len(custom_calls) == 1
        assert (
            custom_calls[0][0][0]
            == "Using custom GEMINI_WEB_SEARCH_MODEL: gemini-2.5-pro"
        )

    @patch("gemini_mcp.config.logger")
    def test_no_environment_variables(self, mock_logger):
        """Test parsing with no environment variables (all defaults)."""
        # Ensure no env vars are set
        for key in [
            "GEMINI_WEB_SEARCH_MODEL",
            "GEMINI_DEFAULT_MODEL",
            "GEMINI_ADVANCED_MODEL",
        ]:
            if key in os.environ:
                del os.environ[key]

        config = get_config()

        assert config.web_search_model == "gemini-flash-latest"
        assert config.default_model == "gemini-flash-lite-latest"
        assert config.advanced_model == "gemini-2.5-pro"

        # Check that logger was not called for custom values
        custom_calls = [
            call
            for call in mock_logger.info.call_args_list
            if "Using custom" in str(call)
        ]
        assert len(custom_calls) == 0

    @patch("gemini_mcp.config.logger")
    def test_invalid_environment_variables(self, mock_logger):
        """Test parsing with invalid environment variables."""
        os.environ["GEMINI_WEB_SEARCH_MODEL"] = "invalid-model"
        os.environ["GEMINI_DEFAULT_MODEL"] = "gemini-flash-latest"
        os.environ["GEMINI_ADVANCED_MODEL"] = "gemini-2.5-pro"

        config = get_config()

        # Should fall back to defaults when invalid values are provided
        assert config.web_search_model == "gemini-flash-latest"  # Default
        assert config.default_model == "gemini-flash-lite-latest"
        assert config.advanced_model == "gemini-2.5-pro"

        # Check warning was logged
        mock_logger.warning.assert_called()
        warning_message = mock_logger.warning.call_args[0][0]
        assert "Error loading configuration" in warning_message
        assert "Invalid model format: invalid-model" in warning_message


class TestDefaultFallbackBehavior:
    """Test cases for default fallback behavior."""

    def setup_method(self):
        """Save original environment variables."""
        self.original_env = os.environ.copy()

    def teardown_method(self):
        """Restore original environment variables."""
        os.environ.clear()
        os.environ.update(self.original_env)
        # Clear the cache after each test
        get_config.cache_clear()

    @patch("gemini_mcp.config.logger")
    def test_fallback_to_defaults_on_error(self, mock_logger):
        """Test that defaults are used when environment variables cause errors."""
        os.environ["GEMINI_WEB_SEARCH_MODEL"] = "invalid-model"

        config = get_config()

        # Should use defaults
        assert config.web_search_model == "gemini-flash-latest"
        assert config.default_model == "gemini-flash-lite-latest"
        assert config.advanced_model == "gemini-2.5-pro"

        # Check warning was logged
        mock_logger.warning.assert_called()
        warning_message = mock_logger.warning.call_args[0][0]
        assert "Error loading configuration" in warning_message
        assert "Invalid model format: invalid-model" in warning_message

    @patch("gemini_mcp.config.logger")
    def test_fallback_with_multiple_errors(self, mock_logger):
        """Test fallback when multiple environment variables are invalid."""
        os.environ["GEMINI_WEB_SEARCH_MODEL"] = "invalid-web"
        os.environ["GEMINI_DEFAULT_MODEL"] = "invalid-default"
        os.environ["GEMINI_ADVANCED_MODEL"] = "invalid-advanced"

        config = get_config()

        # Should use all defaults
        assert config.web_search_model == "gemini-flash-latest"
        assert config.default_model == "gemini-flash-lite-latest"
        assert config.advanced_model == "gemini-2.5-pro"

        # Check warning was logged
        mock_logger.warning.assert_called()
        warning_message = mock_logger.warning.call_args[0][0]
        assert "Error loading configuration" in warning_message
        assert "Invalid model format: invalid-web" in warning_message


class TestErrorHandlingAndWarningLogging:
    """Test cases for error handling and warning logging."""

    @patch("gemini_mcp.config.logger")
    def test_warning_for_invalid_model_format(self, mock_logger):
        """Test that warnings are logged for invalid model formats."""
        with pytest.raises(
            ValueError,
            match="Invalid model format: bad-model. Must start with 'gemini-'",
        ):
            ModelConfig(
                web_search_model="bad-model",
                default_model="gemini-flash-latest",
                advanced_model="gemini-2.5-pro",
            )

        # Check warning was logged
        mock_logger.warning.assert_called_with(
            "Invalid web search model format: bad-model"
        )

    @patch("gemini_mcp.config.logger")
    def test_warning_for_each_invalid_model(self, mock_logger):
        """Test that warnings are logged for each invalid model field."""
        with pytest.raises(ValueError):
            ModelConfig(
                web_search_model="bad-web",
                default_model="bad-default",
                advanced_model="bad-advanced",
            )

        # Check warnings were logged for each invalid model
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        assert any(
            "Invalid web search model format: bad-web" in call for call in warning_calls
        )
        assert any(
            "Invalid default model format: bad-default" in call
            for call in warning_calls
        )
        assert any(
            "Invalid advanced model format: bad-advanced" in call
            for call in warning_calls
        )


class TestConfigurationCaching:
    """Test cases for configuration caching mechanism."""

    def setup_method(self):
        """Clear cache before each test."""
        # Clear the LRU cache
        get_config.cache_clear()

    def test_caching_behavior(self):
        """Test that configuration is cached properly."""
        # First call should create and cache the config
        config1 = get_config()

        # Second call should return the cached config
        config2 = get_config()

        # Should be the same object (cached)
        assert config1 is config2

    def test_cache_with_environment_variables(self):
        """Test that cache respects environment variables."""
        # Set environment variable
        os.environ["GEMINI_WEB_SEARCH_MODEL"] = "gemini-2.5-pro"

        # First call should cache the config with the env var value
        config1 = get_config()
        assert config1.web_search_model == "gemini-2.5-pro"

        # Change environment variable
        os.environ["GEMINI_WEB_SEARCH_MODEL"] = "gemini-flash-latest"

        # Second call should still return cached config (old value)
        config2 = get_config()
        assert config2.web_search_model == "gemini-2.5-pro"
        assert config1 is config2

    def test_cache_clear(self):
        """Test that cache can be cleared."""
        # Get initial config
        config1 = get_config()

        # Clear cache
        get_config.cache_clear()

        # Get config again - should be a new object
        config2 = get_config()
        assert config1 is not config2


class TestConfigurationReload:
    """Test cases for configuration reload functionality."""

    def setup_method(self):
        """Clear cache and save original environment."""
        get_config.cache_clear()
        self.original_env = os.environ.copy()

    def teardown_method(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self.original_env)
        get_config.cache_clear()

    def test_reload_with_changed_environment(self):
        """Test reloading configuration with changed environment variables."""
        # Set initial environment
        os.environ["GEMINI_WEB_SEARCH_MODEL"] = "gemini-2.5-pro"

        # Get initial config
        config1 = get_config()
        assert config1.web_search_model == "gemini-2.5-pro"

        # Change environment
        os.environ["GEMINI_WEB_SEARCH_MODEL"] = "gemini-flash-latest"

        # Clear cache and get new config
        get_config.cache_clear()
        config2 = get_config()

        # Should have new value
        assert config2.web_search_model == "gemini-flash-latest"
        assert config1 is not config2

    def test_reload_with_invalid_environment(self):
        """Test reloading configuration with invalid environment variables."""
        # Set valid initial environment
        os.environ["GEMINI_WEB_SEARCH_MODEL"] = "gemini-2.5-pro"

        # Get initial config
        config1 = get_config()
        assert config1.web_search_model == "gemini-2.5-pro"

        # Set invalid environment
        os.environ["GEMINI_WEB_SEARCH_MODEL"] = "invalid-model"

        # Clear cache and get new config - should fall back to defaults
        get_config.cache_clear()
        config2 = get_config()

        # Should have default value
        assert config2.web_search_model == "gemini-flash-latest"
        assert config1 is not config2


class TestUtilityFunctions:
    """Test cases for utility functions that use the configuration."""

    def setup_method(self):
        """Clear cache before each test."""
        get_config.cache_clear()

    def test_get_model_for_web_search(self):
        """Test get_model_for_web_search function."""
        # Test with default config
        model = get_model_for_web_search()
        assert model == "gemini-flash-latest"

        # Test with custom config via environment
        os.environ["GEMINI_WEB_SEARCH_MODEL"] = "gemini-2.5-pro"
        get_config.cache_clear()

        model = get_model_for_web_search()
        assert model == "gemini-2.5-pro"

    def test_get_default_model(self):
        """Test get_default_model function."""
        # Test with default config
        model = get_default_model()
        assert model == "gemini-flash-lite-latest"

        # Test with custom config via environment
        os.environ["GEMINI_DEFAULT_MODEL"] = "gemini-2.5-pro"
        get_config.cache_clear()

        model = get_default_model()
        assert model == "gemini-2.5-pro"

    def test_get_advanced_model(self):
        """Test get_advanced_model function."""
        # Test with default config
        model = get_advanced_model()
        assert model == "gemini-2.5-pro"

        # Test with custom config via environment
        os.environ["GEMINI_ADVANCED_MODEL"] = "gemini-flash-latest"
        get_config.cache_clear()

        model = get_advanced_model()
        assert model == "gemini-flash-latest"

    def test_get_all_models(self):
        """Test get_all_models function."""
        # Clear any existing environment variables
        for key in [
            "GEMINI_WEB_SEARCH_MODEL",
            "GEMINI_DEFAULT_MODEL",
            "GEMINI_ADVANCED_MODEL",
        ]:
            if key in os.environ:
                del os.environ[key]

        # Clear cache to ensure we get fresh config
        get_config.cache_clear()

        models = get_all_models()

        assert isinstance(models, dict)
        assert "web_search" in models
        assert "default" in models
        assert "advanced" in models

        # Check default values
        assert models["web_search"] == "gemini-flash-latest"
        assert models["default"] == "gemini-flash-lite-latest"
        assert models["advanced"] == "gemini-2.5-pro"

        # Test with custom environment
        os.environ["GEMINI_WEB_SEARCH_MODEL"] = "gemini-2.5-pro"
        os.environ["GEMINI_DEFAULT_MODEL"] = "gemini-flash-lite-latest"
        os.environ["GEMINI_ADVANCED_MODEL"] = "gemini-2.5-pro"
        get_config.cache_clear()

        models = get_all_models()
        assert models["web_search"] == "gemini-2.5-pro"
        assert models["default"] == "gemini-flash-lite-latest"
        assert models["advanced"] == "gemini-2.5-pro"
