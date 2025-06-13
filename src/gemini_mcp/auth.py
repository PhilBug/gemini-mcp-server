from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse


class BearerTokenAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:
        # Exclude the root path from authentication
        if request.url.path == "/":
            return await call_next(request)
        # Check if the header is missing
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return JSONResponse(
                status_code=401, content={"detail": "Authorization header missing"}
            )

        # Checks if the header is in the correct format
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Invalid Authorization header. Must be 'Bearer <token>'"
                },
            )

        # Checks if the token is a AI Studio token (does not check if it is valid)
        if not parts[1].startswith("AI"):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid token. Must be a AI Studio token"},
            )

        token = parts[1]
        request.state.bearer_token = token

        response = await call_next(request)
        return response
