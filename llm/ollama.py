import subprocess
import time

from langchain_ollama import ChatOllama
from langchain_ollama._utils import validate_model
from loguru import logger


class OllamaManager:
    """Validate that a model exists in the local Ollama instance."""

    def __init__(self, model_name: str, temperature: float = 0.0) -> None:
        self._model_name = model_name
        self._chat = ChatOllama(model=self._model_name, temperature=temperature)
        self._is_running = False

    def is_running(self) -> bool:
        try:
            validate_model(client=self._chat._client, model_name=self._model_name)
            return True
        except ConnectionError as exc:
            logger.error(f"Connection to Ollama failed {exc}")
            return False

    def get_chat(self) -> ChatOllama:
        if not self.is_running():
            self.start()
        return self._chat

    def start(self, timeout: float = 30) -> subprocess.Popen | None:
        proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_running():
                return proc
            time.sleep(0.5)
        proc.terminate()
        raise RuntimeError("Ollama did not start in time")
