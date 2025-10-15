import pytest
import os
from unittest.mock import AsyncMock, patch
from gemini_mcp.tools import web_search, use_gemini
from gemini_mcp.config import get_config as get_config_cached


# Existing tests remain unchanged
@pytest.mark.asyncio
@patch("gemini_mcp.tools.get_gemini_client")
async def test_web_search_without_citations(mock_get_gemini_client):
    # Mock the async client and its methods
    mock_client = AsyncMock()
    mock_get_gemini_client.return_value = mock_client
    mock_response = AsyncMock()
    mock_response.text = "Test search result"
    mock_response.candidates = [AsyncMock()]
    mock_response.candidates[0].grounding_metadata = None
    mock_client.aio.models.generate_content.return_value = mock_response

    # Call the function
    result = await web_search(query="test query")

    # Assertions
    assert result == {"text": "Test search result"}
    mock_client.aio.models.generate_content.assert_called_once()


@pytest.mark.asyncio
@patch("gemini_mcp.tools.get_gemini_client")
async def test_web_search_with_citations(mock_get_gemini_client):
    # Mock the async client and its methods
    mock_client = AsyncMock()
    mock_get_gemini_client.return_value = mock_client
    mock_response = AsyncMock()
    mock_response.text = "Test search result with citations"

    # Mock grounding metadata
    mock_metadata = AsyncMock()
    mock_metadata.web_search_queries = ["query1", "query2"]
    mock_metadata.grounding_attributions = [
        {"url": "http://example.com/1", "title": "Source 1"},
        {"url": "http://example.com/2", "title": "Source 2"},
    ]

    # Mock candidates
    mock_candidate = AsyncMock()
    mock_candidate.grounding_metadata = mock_metadata
    mock_response.candidates = [mock_candidate]

    # Mock process_grounding_to_structured_citations
    with patch(
        "gemini_mcp.tools.process_grounding_to_structured_citations"
    ) as mock_process:
        mock_process.return_value = [
            {"url": "http://example.com/1", "title": "Source 1"},
            {"url": "http://example.com/2", "title": "Source 2"},
        ]

        mock_client.aio.models.generate_content.return_value = mock_response

        # Call the function
        result = await web_search(query="test query", include_citations=True)

        # Assertions
        assert "text" in result
        assert result["text"] == "Test search result with citations"
        assert "web_search_queries" in result
        assert result["web_search_queries"] == ["query1", "query2"]
        assert "citations" in result
        assert len(result["citations"]) == 2
        mock_client.aio.models.generate_content.assert_called_once()
        mock_process.assert_called_once_with(mock_metadata)


@pytest.mark.asyncio
@patch("gemini_mcp.tools.get_gemini_client")
async def test_use_gemini(mock_get_gemini_client):
    # Clear cache to ensure we get the default model
    get_config_cached.cache_clear()

    # Mock the async client and its methods
    mock_client = AsyncMock()
    mock_get_gemini_client.return_value = mock_client
    mock_response = AsyncMock()
    mock_response.text = "Gemini test response"
    mock_client.aio.models.generate_content.return_value = mock_response

    # Call the function
    result = await use_gemini(prompt="test prompt")

    # Assertions
    assert result == {"text": "Gemini test response"}
    mock_client.aio.models.generate_content.assert_called_once_with(
        model="gemini-flash-lite-latest",
        contents="test prompt",
    )


@pytest.mark.asyncio
@patch("gemini_mcp.tools.get_gemini_client")
async def test_use_gemini_pro_model(mock_get_gemini_client):
    # Mock the async client and its methods
    mock_client = AsyncMock()
    mock_get_gemini_client.return_value = mock_client
    mock_response = AsyncMock()
    mock_response.text = "Gemini Pro test response"
    mock_client.aio.models.generate_content.return_value = mock_response

    # Call the function
    result = await use_gemini(
        prompt="test prompt", model="gemini-2.5-pro"
    )

    # Assertions
    assert result == {"text": "Gemini Pro test response"}
    mock_client.aio.models.generate_content.assert_called_once_with(
        model="gemini-2.5-pro",
        contents="test prompt",
    )


# New tests for web_search with different configurations
@pytest.mark.asyncio
@patch("gemini_mcp.tools.get_gemini_client")
@patch("gemini_mcp.tools.get_current_date")
@patch("gemini_mcp.tools.get_model_for_web_search")
async def test_web_search_with_custom_model(
    mock_get_model, mock_get_date, mock_get_gemini_client
):
    """Test web_search with a custom configured model."""
    # Clear cache to ensure we get the mocked model
    get_config_cached.cache_clear()

    # Setup mocks
    mock_get_model.return_value = "gemini-2.5-pro"
    mock_get_date.return_value = "2023-01-01"

    mock_client = AsyncMock()
    mock_get_gemini_client.return_value = mock_client
    mock_response = AsyncMock()
    mock_response.text = "Custom model search result"
    mock_response.candidates = [AsyncMock()]
    mock_response.candidates[0].grounding_metadata = None
    mock_client.aio.models.generate_content.return_value = mock_response

    # Call the function
    result = await web_search(query="test query")

    # Assertions
    assert result == {"text": "Custom model search result"}
    mock_client.aio.models.generate_content.assert_called_once_with(
        model="gemini-2.5-pro",
        contents='Conduct targeted Google Searches to gather the most recent, credible information on "test query" and synthesize it into a verifiable text artifact.\n\nInstructions:\n- Query should ensure that the most current information is gathered. The current date is 2023-01-01.\n- Conduct multiple, diverse searches to gather comprehensive information.\n- Consolidate key findings while meticulously tracking the source(s) for each specific piece of information.\n- The output should be a well-written summary or report based on your search findings. \n- Only include the information found in the search results, don\'t make up any information.\n\nResearch Topic:\ntest query\n',
        config={
            "temperature": 0.0,
            "tools": [{"google_search": {}}],
        },
    )


@pytest.mark.asyncio
@patch("gemini_mcp.tools.get_gemini_client")
@patch("gemini_mcp.tools.get_current_date")
@patch("gemini_mcp.tools.get_model_for_web_search")
async def test_web_search_with_citations_custom_model(
    mock_get_model, mock_get_date, mock_get_gemini_client
):
    """Test web_search with citations using a custom configured model."""
    # Clear cache to ensure we get the mocked model
    get_config_cached.cache_clear()

    # Setup mocks
    mock_get_model.return_value = "gemini-2.5-pro"
    mock_get_date.return_value = "2023-01-01"

    mock_client = AsyncMock()
    mock_get_gemini_client.return_value = mock_client
    mock_response = AsyncMock()
    mock_response.text = "Custom model search result with citations"

    # Mock grounding metadata
    mock_metadata = AsyncMock()
    mock_metadata.web_search_queries = ["custom query1", "custom query2"]
    mock_metadata.grounding_attributions = [
        {"url": "http://example.com/1", "title": "Source 1"},
        {"url": "http://example.com/2", "title": "Source 2"},
    ]

    # Mock candidates
    mock_candidate = AsyncMock()
    mock_candidate.grounding_metadata = mock_metadata
    mock_response.candidates = [mock_candidate]

    # Mock process_grounding_to_structured_citations
    with patch(
        "gemini_mcp.tools.process_grounding_to_structured_citations"
    ) as mock_process:
        mock_process.return_value = [
            {"url": "http://example.com/1", "title": "Source 1"},
            {"url": "http://example.com/2", "title": "Source 2"},
        ]

        mock_client.aio.models.generate_content.return_value = mock_response

        # Call the function
        result = await web_search(query="test query", include_citations=True)

        # Assertions
        assert "text" in result
        assert result["text"] == "Custom model search result with citations"
        assert "web_search_queries" in result
        assert result["web_search_queries"] == ["custom query1", "custom query2"]
        assert "citations" in result
        assert len(result["citations"]) == 2
        mock_client.aio.models.generate_content.assert_called_once()
        mock_process.assert_called_once_with(mock_metadata)


# New tests for use_gemini with different configurations
@pytest.mark.asyncio
@patch("gemini_mcp.tools.get_gemini_client")
@patch("gemini_mcp.config.get_default_model")
async def test_use_gemini_with_default_config(
    mock_get_default_model, mock_get_gemini_client
):
    """Test use_gemini with default configured model."""
    # Clear cache to ensure we get the mocked model
    get_config_cached.cache_clear()

    # Setup mocks
    mock_get_default_model.return_value = "gemini-flash-lite-latest"

    mock_client = AsyncMock()
    mock_get_gemini_client.return_value = mock_client
    mock_response = AsyncMock()
    mock_response.text = "Default model response"
    mock_client.aio.models.generate_content.return_value = mock_response

    # Call the function without specifying model
    result = await use_gemini(prompt="test prompt")

    # Assertions
    assert result == {"text": "Default model response"}
    mock_client.aio.models.generate_content.assert_called_once_with(
        model="gemini-flash-lite-latest",
        contents="test prompt",
    )


@pytest.mark.asyncio
@patch("gemini_mcp.tools.get_gemini_client")
@patch("gemini_mcp.config.get_default_model")
@patch("gemini_mcp.config.get_advanced_model")
async def test_use_gemini_with_advanced_config(
    mock_get_advanced_model, mock_get_default_model, mock_get_gemini_client
):
    """Test use_gemini with advanced configured model."""
    # Clear cache to ensure we get the mocked models
    get_config_cached.cache_clear()

    # Setup mocks
    mock_get_default_model.return_value = "gemini-flash-lite-latest"
    mock_get_advanced_model.return_value = "gemini-2.5-pro"

    mock_client = AsyncMock()
    mock_get_gemini_client.return_value = mock_client
    mock_response = AsyncMock()
    mock_response.text = "Advanced model response"
    mock_client.aio.models.generate_content.return_value = mock_response

    # Call the function with pro model
    result = await use_gemini(prompt="test prompt", model="gemini-2.5-pro")

    # Assertions
    assert result == {"text": "Advanced model response"}
    mock_client.aio.models.generate_content.assert_called_once_with(
        model="gemini-2.5-pro",
        contents="test prompt",
    )


# Tests for backward compatibility
@pytest.mark.asyncio
@patch("gemini_mcp.tools.get_gemini_client")
@patch("gemini_mcp.config.get_default_model")
async def test_use_gemini_backward_compatibility_flash(
    mock_get_default_model, mock_get_gemini_client
):
    """Test backward compatibility for gemini-flash-latest model."""
    # Clear cache to ensure we get the mocked model
    get_config_cached.cache_clear()

    # Setup mocks
    mock_get_default_model.return_value = "gemini-flash-lite-latest"

    mock_client = AsyncMock()
    mock_get_gemini_client.return_value = mock_client
    mock_response = AsyncMock()
    mock_response.text = "Backward compatibility response"
    mock_client.aio.models.generate_content.return_value = mock_response

    # Call the function with the old flash model name
    result = await use_gemini(
        prompt="test prompt", model="gemini-flash-latest"
    )

    # Should use the default model (backward compatibility)
    assert result == {"text": "Backward compatibility response"}
    mock_client.aio.models.generate_content.assert_called_once_with(
        model="gemini-flash-lite-latest",  # Uses default model
        contents="test prompt",
    )


@pytest.mark.asyncio
@patch("gemini_mcp.tools.get_gemini_client")
@patch("gemini_mcp.config.get_default_model")
@patch("gemini_mcp.config.get_advanced_model")
async def test_use_gemini_backward_compatibility_pro(
    mock_get_advanced_model, mock_get_default_model, mock_get_gemini_client
):
    """Test backward compatibility for gemini-2.5-pro model."""
    # Clear cache to ensure we get the mocked models
    get_config_cached.cache_clear()

    # Setup mocks
    mock_get_default_model.return_value = "gemini-flash-lite-latest"
    mock_get_advanced_model.return_value = (
        "gemini-2.5-pro"  # Updated to match the actual model name
    )

    mock_client = AsyncMock()
    mock_get_gemini_client.return_value = mock_client
    mock_response = AsyncMock()
    mock_response.text = "Pro backward compatibility response"
    mock_client.aio.models.generate_content.return_value = mock_response

    # Call the function with the old pro model name
    result = await use_gemini(
        prompt="test prompt", model="gemini-2.5-pro"
    )

    # Should use the advanced model
    assert result == {"text": "Pro backward compatibility response"}
    mock_client.aio.models.generate_content.assert_called_once_with(
        model="gemini-2.5-pro",  # Uses advanced model
        contents="test prompt",
    )


# Tests for model selection logic with different configurations
@pytest.mark.asyncio
@patch("gemini_mcp.tools.get_gemini_client")
@patch("gemini_mcp.config.get_default_model")
@patch("gemini_mcp.config.get_advanced_model")
async def test_use_gemini_model_selection_explicit_default(
    mock_get_advanced_model, mock_get_default_model, mock_get_gemini_client
):
    """Test model selection when explicit default model is provided."""
    # Clear cache to ensure we get the mocked models
    get_config_cached.cache_clear()

    # Setup mocks
    mock_get_default_model.return_value = "gemini-flash-lite-latest"
    mock_get_advanced_model.return_value = "gemini-2.5-pro"

    mock_client = AsyncMock()
    mock_get_gemini_client.return_value = mock_client
    mock_response = AsyncMock()
    mock_response.text = "Explicit default response"
    mock_client.aio.models.generate_content.return_value = mock_response

    # Call the function with explicit default model
    result = await use_gemini(prompt="test prompt", model="gemini-flash-lite-latest")

    # Should use the specified model
    assert result == {"text": "Explicit default response"}
    mock_client.aio.models.generate_content.assert_called_once_with(
        model="gemini-flash-lite-latest",
        contents="test prompt",
    )


@pytest.mark.asyncio
@patch("gemini_mcp.tools.get_gemini_client")
@patch("gemini_mcp.config.get_default_model")
@patch("gemini_mcp.config.get_advanced_model")
async def test_use_gemini_model_selection_no_model_specified(
    mock_get_advanced_model, mock_get_default_model, mock_get_gemini_client
):
    """Test model selection when no model is specified."""
    # Clear cache to ensure we get the mocked models
    get_config_cached.cache_clear()

    # Setup mocks
    mock_get_default_model.return_value = "gemini-flash-lite-latest"
    mock_get_advanced_model.return_value = "gemini-2.5-pro"

    mock_client = AsyncMock()
    mock_get_gemini_client.return_value = mock_client
    mock_response = AsyncMock()
    mock_response.text = "No model specified response"
    mock_client.aio.models.generate_content.return_value = mock_response

    # Call the function without specifying model
    result = await use_gemini(prompt="test prompt")

    # Should use the default model
    assert result == {"text": "No model specified response"}
    mock_client.aio.models.generate_content.assert_called_once_with(
        model="gemini-flash-lite-latest",
        contents="test prompt",
    )


# Tests for environment variable configuration integration
@pytest.mark.asyncio
@patch("gemini_mcp.tools.get_gemini_client")
@patch("gemini_mcp.tools.get_model_for_web_search")
@patch.dict(os.environ, {"GEMINI_WEB_SEARCH_MODEL": "gemini-2.5-pro"})
async def test_web_search_with_env_config(mock_get_model, mock_get_gemini_client):
    """Test web_search with environment variable configuration."""
    # Clear cache to ensure we get the mocked model
    get_config_cached.cache_clear()

    # Setup mocks
    mock_get_model.return_value = "gemini-2.5-pro"

    mock_client = AsyncMock()
    mock_get_gemini_client.return_value = mock_client
    mock_response = AsyncMock()
    mock_response.text = "Env config search result"
    mock_response.candidates = [AsyncMock()]
    mock_response.candidates[0].grounding_metadata = None
    mock_client.aio.models.generate_content.return_value = mock_response

    # Call the function
    result = await web_search(query="test query")

    # Assertions
    assert result == {"text": "Env config search result"}
    mock_client.aio.models.generate_content.assert_called_once_with(
        model="gemini-2.5-pro",
        contents='Conduct targeted Google Searches to gather the most recent, credible information on "test query" and synthesize it into a verifiable text artifact.\n\nInstructions:\n- Query should ensure that the most current information is gathered. The current date is 2025-10-15.\n- Conduct multiple, diverse searches to gather comprehensive information.\n- Consolidate key findings while meticulously tracking the source(s) for each specific piece of information.\n- The output should be a well-written summary or report based on your search findings. \n- Only include the information found in the search results, don\'t make up any information.\n\nResearch Topic:\ntest query\n',
        config={
            "temperature": 0.0,
            "tools": [{"google_search": {}}],
        },
    )


@pytest.mark.asyncio
@patch("gemini_mcp.tools.get_gemini_client")
@patch("gemini_mcp.config.get_default_model")
@patch.dict(os.environ, {"GEMINI_DEFAULT_MODEL": "gemini-flash-lite-latest"})
async def test_use_gemini_with_env_config(
    mock_get_default_model, mock_get_gemini_client
):
    """Test use_gemini with environment variable configuration."""
    # Clear cache to ensure we get the mocked model
    get_config_cached.cache_clear()

    # Setup mocks
    mock_get_default_model.return_value = "gemini-flash-lite-latest"

    mock_client = AsyncMock()
    mock_get_gemini_client.return_value = mock_client
    mock_response = AsyncMock()
    mock_response.text = "Env config response"
    mock_client.aio.models.generate_content.return_value = mock_response

    # Call the function without specifying model
    result = await use_gemini(prompt="test prompt")

    # Assertions
    assert result == {"text": "Env config response"}
    mock_client.aio.models.generate_content.assert_called_once_with(
        model="gemini-flash-lite-latest",
        contents="test prompt",
    )


# Test for process_grounding_to_structured_citations function
def test_process_grounding_to_structured_citations_with_none():
    """Test process_grounding_to_structured_citations with None input."""
    from gemini_mcp.utils import process_grounding_to_structured_citations

    # Test with None input
    result = process_grounding_to_structured_citations(None)
    assert result == []


def test_process_grounding_to_structured_citations_with_no_grounding_supports():
    """Test process_grounding_to_structured_citations with metadata lacking grounding_supports."""
    from gemini_mcp.utils import process_grounding_to_structured_citations

    # Create mock metadata without grounding_supports
    class MockMetadata:
        pass

    mock_metadata = MockMetadata()
    result = process_grounding_to_structured_citations(mock_metadata)
    assert result == []


def test_process_grounding_to_structured_citations_with_empty_grounding_supports():
    """Test process_grounding_to_structured_citations with empty grounding_supports."""
    from gemini_mcp.utils import process_grounding_to_structured_citations

    # Create mock metadata with empty grounding_supports
    class MockMetadata:
        def __init__(self):
            self.grounding_supports = []

    mock_metadata = MockMetadata()
    result = process_grounding_to_structured_citations(mock_metadata)
    assert result == []
