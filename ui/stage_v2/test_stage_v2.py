# -*- coding: utf-8 -*-
"""Test de Stage V2 (solo sintaxis/import/creacion). NO full suite.

Ejecutar:  python -m ui.stage_v2.test_stage_v2
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def main():
    import ui.stage_v2.renderer as r  # importa (sintaxis + deps)

    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = r.AtlasStageV2()  # creacion (build_assets incluido)
    assert w.A and "core" in w.A and "glyph_fill" in w.A
    assert len(w.menu_rects) == 9
    print("OK Stage V2: import + creacion, assets=%d" % len(w.A))
    return 0


if __name__ == "__main__":
    sys.exit(main())
