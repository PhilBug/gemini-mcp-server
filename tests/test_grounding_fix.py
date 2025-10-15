import pytest
from unittest.mock import AsyncMock, patch
from gemini_mcp.tools import web_search
from gemini_mcp.utils import process_grounding_to_structured_citations


@pytest.mark.asyncio
@patch("gemini_mcp.tools.get_gemini_client")
async def test_web_search_with_none_grounding_metadata(mock_get_gemini_client):
    """Test web_search when grounding_metadata is None to verify our fix."""
    # Mock the async client and its methods
    mock_client = AsyncMock()
    mock_get_gemini_client.return_value = mock_client
    mock_response = AsyncMock()
    mock_response.text = "Test search result"

    # Mock candidates with None grounding_metadata
    mock_candidate = AsyncMock()
    mock_candidate.grounding_metadata = None
    mock_response.candidates = [mock_candidate]

    mock_client.aio.models.generate_content.return_value = mock_response

    # Call the function with include_citations=True
    result = await web_search(query="test query", include_citations=True)

    # Assertions
    assert result["citations"] == []
    assert result["web_search_queries"] == []
    assert result["text"] == "Test search result"
    mock_client.aio.models.generate_content.assert_called_once()


def test_process_grounding_to_structured_citations_none_fix():
    """Test that process_grounding_to_structured_citations handles None input correctly."""
    # This should not raise an exception
    result = process_grounding_to_structured_citations(None)
    assert result == []


def test_process_grounding_to_structured_citations_no_grounding_supports():
    """Test that process_grounding_to_structured_citations handles missing grounding_supports correctly."""

    class MockMetadata:
        pass

    # This should not raise an exception
    result = process_grounding_to_structured_citations(MockMetadata())
    assert result == []


def test_process_grounding_to_structured_citations_empty_grounding_supports():
    """Test that process_grounding_to_structured_citations handles empty grounding_supports correctly."""

    class MockMetadata:
        def __init__(self):
            self.grounding_supports = []

    # This should not raise an exception
    result = process_grounding_to_structured_citations(MockMetadata())
    assert result == []
