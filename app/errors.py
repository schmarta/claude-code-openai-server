"""OpenAI-shaped error envelopes.

OpenAI clients (including the OpenAI Python SDK that hermes uses) expect errors
in the shape ``{"error": {"message", "type", "param", "code"}}``. We mirror that
both for non-streaming JSON responses and for mid-stream SSE error chunks.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from fastapi.responses import JSONResponse


class OpenAIError(Exception):
    """An error that should be rendered as an OpenAI error envelope."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        type: str = "invalid_request_error",
        param: Optional[str] = None,
        code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.type = type
        self.param = param
        self.code = code

    def envelope(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.type,
                "param": self.param,
                "code": self.code,
            }
        }

    def json_response(self) -> JSONResponse:
        return JSONResponse(status_code=self.status_code, content=self.envelope())

    def sse_data(self) -> str:
        """The body of a single SSE ``data:`` line carrying this error."""
        return json.dumps(self.envelope())


def error_envelope(
    message: str,
    *,
    type: str = "server_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> dict[str, Any]:
    return {"error": {"message": message, "type": type, "param": param, "code": code}}
