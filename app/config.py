"""Runtime configuration (env-driven, prefix ``CCI_``) and model-alias resolution.

All knobs are read from the environment with a ``CCI_`` prefix (e.g.
``CCI_PORT=9000``). Defaults are chosen so the server runs out of the box against
a local ``claude login`` session with no API key.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Model aliases the `claude` CLI accepts directly on `--model`. Anything in this
# set, or any id beginning with "claude", is passed through verbatim; everything
# else falls back to `default_model`. The CLI itself resolves an alias like
# "opus" to the current concrete id, so we never hard-code dated ids here.
KNOWN_MODEL_ALIASES = {
    "opus",
    "sonnet",
    "haiku",
    "fable",
    "opusplan",
    "default",
}


class Settings(BaseSettings):
    """Server settings. Override any field via a ``CCI_<FIELD>`` environment var."""

    model_config = SettingsConfigDict(env_prefix="CCI_", extra="ignore")

    # ── HTTP server ──────────────────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 8787

    # ── Claude CLI ───────────────────────────────────────────────────────────
    claude_bin: str = "claude"
    default_model: str = "claude-opus-4-8"
    default_effort: str | None = None
    permission_mode: str = "bypassPermissions"
    # Force the CLI to inject tool schemas directly rather than behind a
    # tool-search indirection, so hermes' functions are always visible.
    enable_tool_search: bool = False

    # ── Workspace ──────────────────────────────────────────────────────────—
    default_workdir: Path = Path("~/cci-workspace").expanduser()
    # Per-request `workdir` overrides must resolve under one of these roots.
    # Empty => only `default_workdir` is allowed.
    allowed_workdir_roots: list[str] = []

    # ── MCP bridge ─────────────────────────────────────────────────────────—
    mcp_path_prefix: str = "/mcp"

    # ── Lifecycle / timeouts (seconds) ─────────────────────────────────────—
    request_timeout_s: int = 600
    suspended_ttl_s: int = 300
    idle_session_ttl_s: int = 900
    gc_interval_s: int = 30

    # ── Logging ────────────────────────────────────────────────────────────—
    log_level: str = "INFO"

    @field_validator("default_workdir", mode="before")
    @classmethod
    def _expand_workdir(cls, v: object) -> object:
        if isinstance(v, str):
            return Path(v).expanduser()
        return v

    def resolved_workdir_roots(self) -> list[Path]:
        """The set of allowed workspace roots as resolved absolute paths."""
        roots = [self.default_workdir]
        roots.extend(Path(r).expanduser() for r in self.allowed_workdir_roots)
        return [p.resolve() for p in roots]

    def resolve_workdir(self, requested: str | None) -> Path:
        """Resolve and validate a per-request workdir override.

        Returns ``default_workdir`` when no override is given. Raises
        ``ValueError`` if the override escapes every allowed root (traversal
        guard).
        """
        if not requested:
            return self.default_workdir
        candidate = Path(requested).expanduser().resolve()
        for root in self.resolved_workdir_roots():
            if candidate == root or root in candidate.parents:
                return candidate
        raise ValueError(
            f"workdir {requested!r} is not under an allowed root "
            f"({[str(r) for r in self.resolved_workdir_roots()]})"
        )


def resolve_model(requested: str | None, settings: Settings) -> str:
    """Map an OpenAI ``model`` field to a value the `claude` CLI accepts.

    Known aliases (opus/sonnet/haiku/…) and any ``claude*`` id pass through;
    anything else (including ``None`` or empty) falls back to the default model.
    """
    if not requested:
        return settings.default_model
    r = requested.strip()
    if r in KNOWN_MODEL_ALIASES or r.lower().startswith("claude"):
        return r
    return settings.default_model


@lru_cache
def get_settings() -> Settings:
    """Process-wide cached settings instance."""
    return Settings()


def ensure_workdir(path: Path) -> Path:
    """Create the workspace dir if missing; return it. Best-effort."""
    os.makedirs(path, exist_ok=True)
    return path
