# Gemini MCP Server

A Model Context Protocol server that provides web search capabilities powered by Google's Gemini API. This server enables LLMs to perform intelligent web searches and return synthesized responses with citations.

**Available Tools:**
- **search_web** - Performs a web search using Gemini and returns synthesized results with citations
  - `query` (string, required): The search query to execute

**Example Response:**

```json
{
  "text": "Recent advancements in AI include breakthrough developments in large language models, computer vision, and autonomous systems...",
  "web_search_queries": ["latest AI developments 2024", "AI breakthroughs"],
  "citations": [
    {
      "url": "https://example.com/ai-news",
      "title": "Latest AI Developments 2024",
      "text_content": "Summary of recent AI advances..."
    },
    ...
  ]
}
```

## Installation

```bash
pip install git+https://github.com/philschmid/gemini-mcp-example.git
```

## Authentication

- **STDIO mode**: Uses `GEMINI_API_KEY` environment variable
- **HTTP mode**: Requires Bearer token in Authorization header

### Running the Server

#### STDIO Mode (Local/Direct Integration)

```bash
GEMINI_API_KEY="your_gemini_api_key_here" gemini-mcp --transport stdio
```

#### HTTP Mode (Network Access)

```bash
gemini-mcp --transport streamable-http
```

The server will start on `http://0.0.0.0:8000/mcp`

## Usage Examples

Add to your `mcpServers` configuration:

**STDIO Mode:**
```json
{
  "mcpServers": {
    "gemini-search": {
      "command": "gemini-mcp",
      "args": ["--transport", "stdio"],
      "env": {
        "GEMINI_API_KEY": "your_gemini_api_key_here"
      }
    }
  }
}
```

### With MCP Inspector

Start the server and test your server using the MCP inspector:

```bash
npx @modelcontextprotocol/inspector
```


## License

This project is licensed under the MIT License.