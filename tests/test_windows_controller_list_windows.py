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


class _Dwmapi:
    def __init__(self, cloaked: dict[int, int]) -> None:
        self._cloaked = cloaked

    def DwmGetWindowAttribute(self, handle, _attribute, value, _size):
        value._obj.value = self._cloaked.get(int(handle), 0)
        return 0


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


def test_list_windows_excludes_cloaked_window() -> None:
    controller = _controller()
    controller._USER32._visible = {1: True, 2: True, 3: True}
    controller._USER32._titles = {1: "Calculadora", 2: "Calculadora", 3: ""}
    controller._DWMAPI = _Dwmapi({2: 2})

    windows = controller.list_windows()

    assert [window["handle"] for window in windows] == [1]


def test_list_windows_includes_uncloaked_window_when_dwm_reports_zero() -> None:
    controller = _controller()
    controller._DWMAPI = _Dwmapi({})

    windows = controller.list_windows()

    assert [window["handle"] for window in windows] == [1]