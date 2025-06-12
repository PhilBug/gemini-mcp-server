import pytest
from unittest.mock import AsyncMock, patch
from gemini_mcp.tools import web_search, use_gemini


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
        model="gemini-2.5-flash-preview-05-20",
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
        prompt="test prompt", model="gemini-2.5-pro-preview-05-06"
    )

    # Assertions
    assert result == {"text": "Gemini Pro test response"}
    mock_client.aio.models.generate_content.assert_called_once_with(
        model="gemini-2.5-pro-preview-05-06",
        contents="test prompt",
    )
