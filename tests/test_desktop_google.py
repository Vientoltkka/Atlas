from tools.tool_context import ToolContext
from use_cases.desktop_interaction import DesktopInteractionUseCase


class RecordingExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ToolContext]] = []

    def execute(self, tool_name: str, context: ToolContext) -> str:
        self.calls.append((tool_name, context))
        return "ok"


def test_google_uses_only_the_fixed_url() -> None:
    executor = RecordingExecutor()
    use_case = DesktopInteractionUseCase(executor)

    assert use_case.execute("abre Google") == "✓ Abriendo Google."
    assert executor.calls[0][0] == "desktop.open_url"
    assert executor.calls[0][1].parameters == {"url": "https://www.google.com"}


def test_google_accepts_normalized_voice_text_without_arbitrary_url() -> None:
    executor = RecordingExecutor()
    use_case = DesktopInteractionUseCase(executor)

    assert use_case.execute("abre google\n\nResponde en español") == "✓ Abriendo Google."
    assert executor.calls[0][1].parameters == {"url": "https://www.google.com"}


def test_google_rejects_an_application_suffix() -> None:
    executor = RecordingExecutor()
    use_case = DesktopInteractionUseCase(executor)

    assert use_case.execute("abre Google con Chrome") == (
        "Error: Petición ambigua: usa exactamente 'abre Google'."
    )
    assert executor.calls == []
