FROM python:3.13-slim

ENV PYTHONUNBUFFERED True

# Install uv
RUN pip install uv

# Install dependencies
WORKDIR /app
COPY pyproject.toml .
COPY src ./src
RUN uv pip install --system .

# Set the entrypoint
EXPOSE 8000
CMD ["gemini-mcp", "--transport", "streamable-http"] 