from fastmcp import Context
from pydantic import Field
from starlette.responses import JSONResponse
from typing import Annotated, List
from .utils import (
    process_grounding_to_structured_citations,
    get_current_date,
    get_gemini_client,
)


web_search_prompt = """Conduct targeted Google Searches to gather the most recent, credible information on "{query}" and synthesize it into a verifiable text artifact.

Instructions:
- Query should ensure that the most current information is gathered. The current date is {current_date_str}.
- Conduct multiple, diverse searches to gather comprehensive information.
- Consolidate key findings while meticulously tracking the source(s) for each specific piece of information.
- The output should be a well-written summary or report based on your search findings. 
- Only include the information found in the search results, don't make up any information.

Research Topic:
{query}
"""


async def search_web(
    query: Annotated[
        str,
        Field(description="The query to search."),
    ],
) -> JSONResponse:  # Return type is now StructuredToolOutput
    """Tool to search the web using Google Search API - performs real-time web searches."""

    genai_client = await get_gemini_client()
    current_date_str = get_current_date()

    print(f"Searching web for query: {query}")
    response = await genai_client.aio.models.generate_content(
        model="gemini-2.0-flash",
        contents=web_search_prompt.format(
            query=query, current_date_str=current_date_str
        ),
        config={
            "temperature": 0.0,
            "tools": [{"google_search": {}}],
        },
    )

    structured_citations = []
    web_search_queries_used = []

    if response.candidates[0].grounding_metadata:
        structured_citations = process_grounding_to_structured_citations(
            response.candidates[0].grounding_metadata
        )

        # Extract web search queries if available
        if response.candidates[0].grounding_metadata.web_search_queries:
            web_search_queries_used = list(
                response.candidates[0].grounding_metadata.web_search_queries
            )
    return {
        "text": response.text,
        "web_search_queries": web_search_queries_used,
        "citations": structured_citations,
    }
