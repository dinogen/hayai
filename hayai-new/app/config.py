import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the hayai-new root (or current working directory as fallback)
_ENV_CANDIDATES = [
    Path(__file__).resolve().parent.parent / ".env",
    Path(os.getcwd()) / ".env",
]
for _env_path in _ENV_CANDIDATES:
    if _env_path.exists():
        load_dotenv(_env_path)
        break

class Settings:
    DB_HOST: str = os.getenv("DB_HOST", "127.0.0.1")
    DB_PORT: int = int(os.getenv("DB_PORT", "3306"))
    DB_NAME: str = os.getenv("DB_NAME", "hayai")
    DB_USER: str = os.getenv("DB_USER", "dinogen")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "abc123")

    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_API_BASE_URL: str = os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com/v1")

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
