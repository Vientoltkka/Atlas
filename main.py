"""Atlas application entry point with Ollama model discovery."""

from collections.abc import Mapping
from typing import Protocol, cast


class _OllamaListClient(Protocol):
    """Protocol for the Ollama client behavior used by Atlas."""

    def list(self) -> object:
        """Return the raw Ollama model list response."""


class _OllamaModule(Protocol):
    """Protocol for the official Ollama Python module surface used by Atlas."""

    def Client(self) -> _OllamaListClient:
        """Create an Ollama client."""


class OllamaClient:
    """Client for listing locally installed Ollama models."""

    def __init__(self) -> None:
        """Initialize the Ollama client.

        Raises:
            RuntimeError: If the official Ollama Python package is unavailable.
        """
        self._client = self._load_client()

    def list_models(self) -> list[str]:
        """Return the names of locally installed Ollama models.

        Returns:
            A list containing only the installed model names.

        Raises:
            RuntimeError: If Ollama is unavailable or returns an invalid response.
        """
        try:
            response = self._client.list()
        except Exception as exc:
            raise RuntimeError("Ollama no esta disponible o no responde.") from exc

        raw_models = self._extract_models(response)
        return [name for model in raw_models if (name := self._extract_model_name(model))]

    def _load_client(self) -> _OllamaListClient:
        """Create the official Ollama Python client.

        Returns:
            An official Ollama client instance.

        Raises:
            RuntimeError: If the official Ollama Python package is unavailable.
        """
        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError("la libreria oficial 'ollama' no esta instalada.") from exc

        return cast(_OllamaModule, ollama).Client()

    def _extract_models(self, response: object) -> list[object]:
        """Extract model records from an Ollama list response.

        Args:
            response: Response returned by the official Ollama Python client.

        Returns:
            The raw model records from the response.

        Raises:
            RuntimeError: If the response shape is not recognized.
        """
        if isinstance(response, Mapping):
            models = response.get("models", [])
        else:
            models = getattr(response, "models", [])

        if not isinstance(models, list):
            raise RuntimeError("Ollama devolvio una respuesta inesperada.")

        return models

    def _extract_model_name(self, model: object) -> str | None:
        """Extract a model name from one Ollama model record.

        Args:
            model: A raw model record from Ollama.

        Returns:
            The model name, or None if the record has no usable name.
        """
        if isinstance(model, Mapping):
            value = model.get("model") or model.get("name")
        else:
            value = getattr(model, "model", None) or getattr(model, "name", None)

        return value if isinstance(value, str) else None


def main() -> None:
    """Run Atlas and print the available Ollama models."""
    print("Atlas OS v0.1.0")
    print()

    try:
        client = OllamaClient()
        models = client.list_models()
    except RuntimeError as exc:
        print(f"No se pudo conectar con Ollama: {exc}")
        return

    print("Modelos encontrados:")
    print()

    for model in models:
        print(f"• {model}")


if __name__ == "__main__":
    main()
"""Atlas entry point."""

from core.atlas import Atlas


def main() -> None:
    """Start Atlas."""
    atlas = Atlas()
    atlas.start()


if __name__ == "__main__":
    main()
    