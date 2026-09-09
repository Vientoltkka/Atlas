"""Controlled Computer Worker fixture: the single safe demo target (V1).

The dev worker route may only edit this file. It starts in a deliberately
wrong state ("valor-incorrecto") and the bounded dev goal is to restore the
expected value ("valor-correcto") and verify it with the focal test.
"""

CONTROL_VALUE = "valor-correcto"
