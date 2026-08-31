from tools.desktop.windows_controller import WindowsDesktopController


class _User32:
    def __init__(self) -> None:
        self._titles = {1: "Bloc de notas", 2: "Oculta", 3: ""}
        self._visible = {1: True, 2: False, 3: True}
        self._pids = {1: 101, 2: 202, 3: 303}

    def EnumWindows(self, callback, _value):
        for handle in (1, 2, 3):
            callback(handle, 0)
        return True

    def IsWindow(self, _handle):
        return True

    def IsWindowVisible(self, handle):
        return self._visible[handle]

    def GetWindowTextLengthW(self, handle):
        return len(self._titles[handle])

    def GetWindowTextW(self, handle, buffer, _size):
        buffer.value = self._titles[handle]
        return len(buffer.value)

    def GetWindowThreadProcessId(self, handle, process_id):
        process_id._obj.value = self._pids[handle]
        return 1


def _controller() -> WindowsDesktopController:
    controller = WindowsDesktopController()
    controller._USER32 = _User32()
    return controller


def test_list_windows_includes_visible_window_with_title() -> None:
    assert _controller().list_windows() == [
        {"handle": 1, "title": "Bloc de notas", "process_id": 101}
    ]


def test_list_windows_excludes_invisible_window() -> None:
    windows = _controller().list_windows()

    assert all(window["handle"] != 2 for window in windows)


def test_list_windows_excludes_empty_title() -> None:
    windows = _controller().list_windows()

    assert all(window["handle"] != 3 for window in windows)