# -*- coding: utf-8 -*-
"""Lanzador aislado de la Stage V2.

Uso:  python -m ui.stage_v2 [1..5]
"""
import sys

from PySide6.QtWidgets import QApplication

from ui.stage_v2.renderer import AtlasStageV2


def main():
    QApplication.setStyle("Fusion")
    app = QApplication(sys.argv)
    w = AtlasStageV2()
    for a in sys.argv[1:]:
        if a in ("1", "2", "3", "4", "5"):
            w._set_state(int(a) - 1)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
