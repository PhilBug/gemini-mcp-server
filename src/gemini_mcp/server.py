from fastmcp import FastMCP
from starlette.middleware import Middleware
import inspect
from gemini_mcp import tools
import argparse
import os
from gemini_mcp.auth import BearerTokenAuthMiddleware

mcp = FastMCP(
    name="Gemini Web Search",
)

# Dynamically add all async functions from the tools module
for name, func in inspect.getmembers(tools):
    if inspect.isasyncgenfunction(func) or inspect.iscoroutinefunction(func):
        if hasattr(func, "__module__") and func.__module__ == tools.__name__:
            mcp.add_tool(fn=func, name=name)


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
        # run_kwargs["log_level"] = "info" # uvicorn's log_level can be configured here if needed
    elif args.transport == "stdio":
        os.environ["MCP_TRANSPORT_MODE"] = "stdio"
        # For stdio, host, port, path, and middleware are not applicable

    mcp.run(**run_kwargs)


if __name__ == "__main__":
    main()
