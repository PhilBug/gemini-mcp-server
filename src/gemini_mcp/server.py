from fastmcp import FastMCP
from starlette.middleware import Middleware
import inspect
from gemini_mcp import tools
import argparse
import os
from gemini_mcp.auth import BearerTokenAuthMiddleware
from fastmcp.tools import FunctionTool
from starlette.responses import HTMLResponse
from starlette.requests import Request
from html import escape


mcp = FastMCP(
    name="gemini-mcp",
    instructions="This server is uses Gemini API models and features to help you build AI Agents.",
)

# Dynamically add all async functions from the tools module
for name, func in inspect.getmembers(tools):
    if inspect.isasyncgenfunction(func) or inspect.iscoroutinefunction(func):
        if hasattr(func, "__module__") and func.__module__ == tools.__name__:
            mcp.add_tool(tool=FunctionTool.from_function(func, name=name))


@mcp.custom_route("/", methods=["GET"])
async def landing_page(request: Request):
    """
    Serves a landing page with ASCII art and connection details.
    """
    scheme = request.url.scheme
    if "x-forwarded-proto" in request.headers:
        scheme = request.headers["x-forwarded-proto"]
    host = request.url.netloc
    connect_url = f"{scheme}://{host}/mcp/"

    gemini_art = """         ###
       #######
      #########
     ###########
   ### ###########
  ##### ##########
 ###### #########
 ###### ########
  ##### ######
   ###  ####
   =*#  ##+=
  ==*#  #+==
  ===*  +===
   ==    ==
"""
    gemini_art_lines = gemini_art.split("\n")
    art_width = max(len(line) for line in gemini_art_lines) if gemini_art_lines else 0

    text_lines = [
        "",
        "",
        "",
        "",
        "",
        "Remote MCP Server for Google Gemini 2.5 and Google Search (Tool)",
        "",
        f"Connect to: `{connect_url}`",
        "",
        "Powered through <a href='https://ai.studio' target='_blank'>https://ai.studio</a> and Google Gemini",
    ]

    response_lines = []
    num_lines = max(len(gemini_art_lines), len(text_lines))

    for i in range(num_lines):
        art_line = gemini_art_lines[i] if i < len(gemini_art_lines) else ""
        text_line = text_lines[i] if i < len(text_lines) else ""

        padding = " " * ((art_width + 4) - len(art_line))
        art_part = escape(art_line)
        # For the text part, we avoid escaping since it contains an HTML anchor tag.
        # This is safe because we construct the string and URL ourselves.
        response_lines.append(
            f'<span style="color: #6495ED;">{art_part}</span>{padding}{text_line}'
        )

    response_text = "\n".join(response_lines)

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Gemini MCP Server</title>
    <style>
        body {{
            background-color: #0d1117;
            color: #c9d1d9;
            font-family: monospace;
        }}
        pre {{
            font-size: 14px;
        }}
        a {{
            color: #58a6ff;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
<pre>{response_text}</pre>
</body>
</html>
"""
    return HTMLResponse(content=html_content.strip())


def main():
    parser = argparse.ArgumentParser(description="Run Gemini MCP Server.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="streamable-http",
        help="Transport method to use (default: streamable-http)",
    )
    args = parser.parse_args()

    run_kwargs = {"transport": args.transport}
    custom_middleware = []

    if args.transport == "streamable-http":
        os.environ["MCP_TRANSPORT_MODE"] = "streamable-http"
        custom_middleware = [Middleware(BearerTokenAuthMiddleware)]
        run_kwargs["host"] = "0.0.0.0"
        run_kwargs["port"] = 8000
        run_kwargs["path"] = "/mcp"
        run_kwargs["middleware"] = custom_middleware
    elif args.transport == "stdio":
        os.environ["MCP_TRANSPORT_MODE"] = "stdio"

    mcp.run(**run_kwargs)


if __name__ == "__main__":
    main()
