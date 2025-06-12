from pydantic import BaseModel
from typing import List, Any
import datetime
import os
from fastmcp.server.dependencies import get_http_request
from starlette.requests import Request


class Source(BaseModel):
    title: str
    uri: str


class CitationEntry(BaseModel):
    text: str
    start_index: int
    end_index: int
    sources: List[Source]


class WebSearchToolOutput(BaseModel):
    text: str
    web_search_queries: List[str]
    citations: List[CitationEntry]


class TextToolOutput(BaseModel):
    text: str


def get_current_date() -> str:
    """Returns the current date as a string in YYYY-MM-DD format."""
    return datetime.datetime.now().strftime("%Y-%m-%d")


def process_grounding_to_structured_citations(
    grounding_metadata: Any,
):
    """
    Processes grounding metadata from the Gemini API response to produce a list
    of structured CitationEntry objects.
    Based on the user's provided example script.
    """
    citations = []
    for support in grounding_metadata.grounding_supports:
        obj = {
            "text": support.segment.text,
            "start_index": (
                support.segment.start_index if support.segment.start_index else 0
            ),
            "end_index": support.segment.end_index,
            "sources": [
                {
                    "title": grounding_metadata.grounding_chunks[idx].web.title,
                    "uri": grounding_metadata.grounding_chunks[idx].web.uri,
                }
                for idx in support.grounding_chunk_indices
            ],
        }
        citations.append(obj)
    return citations


async def get_gemini_client():
    """
    Handles authentication for Gemini client based on transport mode.

    Returns:
        tuple: (Client, api_key_to_use) - The authenticated Gemini client and the API key used

    Raises:
        ValueError: If authentication fails or transport mode is invalid
        ImportError: If google.genai library is not found
    """
    transport_mode = os.getenv(
        "MCP_TRANSPORT_MODE", "streamable-http"
    )  # Default to streamable-http if not set
    api_key_to_use = None

    if transport_mode == "stdio":
        api_key_to_use = os.getenv("GEMINI_API_KEY")
        if not api_key_to_use:

            raise ValueError(
                "Authentication failed. GEMINI_API_KEY not found for stdio mode."
            )

    elif transport_mode == "streamable-http":
        try:
            request_starlette: Request = get_http_request()
        except RuntimeError:

            raise ValueError(
                "Tool must be called via an HTTP request for streamable-http mode."
            )

        bearer_token = getattr(request_starlette.state, "bearer_token", None)
        if not bearer_token:

            raise ValueError(
                "Authentication failed in streamable-http mode. Bearer token not found."
            )

        api_key_to_use = bearer_token

    else:
        raise ValueError(f"Invalid MCP_TRANSPORT_MODE: {transport_mode}")

    # Ensure api_key_to_use is set by one of the branches
    if not api_key_to_use:
        raise ValueError("Critical: API key for Gemini client is missing.")

    try:
        from google.genai import Client
    except ImportError:
        raise ImportError("google-genai library not found. Please install it.")

    genai_client = Client(api_key=api_key_to_use)
    return genai_client
