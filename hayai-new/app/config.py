import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the hayai-new root (or current working directory as fallback)
_ENV_CANDIDATES = [
    Path(__file__).resolve().parent.parent / ".env",
    Path(os.getcwd()) / ".env",
]
_ENV_PATH = None
for _env_path in _ENV_CANDIDATES:
    if _env_path.exists():
        load_dotenv(_env_path)
        _ENV_PATH = _env_path
        break


def _parse_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def _env_value_in_file(key: str) -> str | None:
    """Read a KEY=value line directly from the .env file (no env caching)."""
    if _ENV_PATH is None or not _ENV_PATH.exists():
        return None
    for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key}="):
            return stripped.split("=", 1)[1].strip()
    return None


def get_news_llm_enabled() -> bool:
    """Current value of the NEWS_LLM_ENABLED flag.

    Checks the process environment first (e.g. systemd EnvironmentFile), then
    falls back to reading the .env file directly so API toggles take effect
    without a process restart.
    """
    return _parse_bool(os.getenv("NEWS_LLM_ENABLED", _env_value_in_file("NEWS_LLM_ENABLED") or "true"))


def set_news_llm_enabled(value: bool) -> bool:
    """Persist the NEWS_LLM_ENABLED flag in the .env file (idempotent).

    Also updates the process environment so subsequent reads in the same
    process reflect the new value immediately.
    """
    global _ENV_PATH
    value = bool(value)
    new_val = "true" if value else "false"
    os.environ["NEWS_LLM_ENABLED"] = new_val

    if _ENV_PATH is None:
        _ENV_PATH = Path(os.getcwd()) / ".env"
    if not _ENV_PATH.exists():
        _ENV_PATH.write_text(f"NEWS_LLM_ENABLED={new_val}\n", encoding="utf-8")
        return value

    content = _ENV_PATH.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in content else "\n"
    lines = content.splitlines()
    key = "NEWS_LLM_ENABLED"
    replaced = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(f"{key}="):
            indent = line[: len(line) - len(line.lstrip())]
            lines[i] = f"{indent}{key}={new_val}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{key}={new_val}")

    _ENV_PATH.write_text(newline.join(lines) + newline, encoding="utf-8")
    return value


class Settings:
    DB_HOST: str = os.getenv("DB_HOST", "127.0.0.1")
    DB_PORT: int = int(os.getenv("DB_PORT", "3306"))
    DB_NAME: str = os.getenv("DB_NAME", "hayai")
    DB_USER: str = os.getenv("DB_USER", "dinogen")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "abc123")

    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_API_BASE_URL: str = os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com/v1")

    # When False, the 'sentiment' batch job skips DeepSeek LLM analysis
    # (news are still downloaded by the 'news' job, and 'signal' keeps using
    # already-computed news_sentiment rows). Useful to avoid token consumption
    # while away (e.g. holidays).
    NEWS_LLM_ENABLED: bool = _parse_bool(os.getenv("NEWS_LLM_ENABLED", "true"))

    # Auth (single user, cookie-based session)
    AUTH_USERNAME: str = os.getenv("AUTH_USERNAME", "")
    AUTH_PASSWORD: str = os.getenv("AUTH_PASSWORD", "")
    AUTH_SESSION_SECRET: str = os.getenv("AUTH_SESSION_SECRET", "")
    AUTH_SESSION_MAX_AGE: int = int(os.getenv("AUTH_SESSION_MAX_AGE", "43200"))

    # Filesystem paths support: /opt/hayai on Linux, local relative folder on Windows
    @property
    def HAYAI_ROOT(self) -> Path:
        env_root = os.getenv("HAYAI_ROOT")
        if env_root:
            return Path(env_root)
        # Default: if on Windows use local hayai-new folder, if Linux use /opt/hayai
        if os.name == "nt":
            return Path(__file__).resolve().parent.parent
        return Path("/opt/hayai")

    @property
    def MODELS_DIR(self) -> Path:
        p = self.HAYAI_ROOT / "models"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def DATA_DIR(self) -> Path:
        p = self.HAYAI_ROOT / "data"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def LOGS_DIR(self) -> Path:
        p = self.HAYAI_ROOT / "logs"
        p.mkdir(parents=True, exist_ok=True)
        return p

settings = Settings()
