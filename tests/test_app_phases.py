"""Phase selection behavior tests for ARMirrorApp."""

import pytest

pytest.importorskip("cv2")
pytest.importorskip("numpy")

from app import ARMirrorApp


def test_phase_0_is_preserved() -> None:
    app = ARMirrorApp(phase=0, demo_duration=1)
    assert app.phase == 0


def test_phase_2_is_preserved() -> None:
    app = ARMirrorApp(phase=2, demo_duration=1)
    assert app.phase == 2


def test_phase_1_is_normalized_to_fallback() -> None:
    app = ARMirrorApp(phase=1, demo_duration=1)
    assert app.phase == 0
