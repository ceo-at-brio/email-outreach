"""
Central config — reads all secrets from the .env file in this directory.
Never hardcode API keys in source files; edit .env instead (it is gitignored).
Copy .env.example → .env and fill in your values before running the pipeline.
"""
import os
import pathlib

_ENV_FILE = pathlib.Path(__file__).parent / ".env"


def _load_env(path: pathlib.Path) -> None:
    """Parse KEY=VALUE lines from .env into os.environ.

    Values in .env always win over shell-level env vars so that
    project-specific keys are never shadowed by stale shell exports.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            os.environ[key] = val  # always override shell env vars


_load_env(_ENV_FILE)


def _require(key: str) -> str:
    val = os.environ.get(key, "").strip()
    if not val:
        raise EnvironmentError(
            f"\n[config] Missing required key: {key}\n"
            f"  → Add it to your .env file:  {_ENV_FILE}\n"
            f"  → See .env.example for the full list of keys.\n"
        )
    return val


# ── Gemini ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = _require("GEMINI_API_KEY")

# ── Apollo ────────────────────────────────────────────────────────────────────
APOLLO_API_KEY = _require("APOLLO_API_KEY")

# The exact email address of the Apollo inbox to send from.
SENDER_EMAIL = _require("SENDER_EMAIL")

# ── SerpAPI (optional — only used by google-leads.py) ────────────────────────
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")
