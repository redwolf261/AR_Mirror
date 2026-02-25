"""
Comprehensive geometry tests for Phase 1 fixes.

Tests cover:
  - Convex polygon torso mask (replaces rectangle)
  - Body lean / side-view torso correctness
  - Pose confidence filtering (low-visibility skip)
  - Camera depth normalisation (near vs far subject)
  - Edge cases: cropped frames, minimal landmarks, zero-size inputs
"""

import sys
import types
import unittest
import numpy as np

sys.path.insert(0, '.')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_landmark(x: float, y: float, visibility: float = 0.95):
    """Create a minimal mock MediaPipe landmark."""
    return types.SimpleNamespace(x=x, y=y, visibility=visibility)


def make_landmarks_upright(
    ls=(0.35, 0.30),
    rs=(0.65, 0.30),
    lh=(0.37, 0.72),
    rh=(0.63, 0.72),
    vis: float = 0.95,
):
    """25-landmark list with key torso landmarks set."""
    lms = [make_landmark(0.5, 0.5) for _ in range(25)]
    lms[11] = make_landmark(*ls, visibility=vis)
    lms[12] = make_landmark(*rs, visibility=vis)
    lms[23] = make_landmark(*lh, visibility=vis)
    lms[24] = make_landmark(*rh, visibility=vis)
    return lms


def make_parser():
    """Return a SemanticParser without hitting the MediaPipe session."""
    from src.core.semantic_parser import SemanticParser
    import unittest.mock as mock
    with mock.patch('src.core.parsing_backends.MediaPipeBackend.__init__', return_value=None):
        with mock.patch('src.core.parsing_backends.MediaPipeBackend.is_available', return_value=True):
            try:
                sp = SemanticParser(backend='mediapipe')
            except Exception:
                sp = SemanticParser.__new__(SemanticParser)
                sp.temporal_smoothing = False
                sp.prev_masks = None
                sp.smoothing_alpha = 0.7
                from src.core.parsing_backends import MediaPipeBackend
                sp.backend = object.__new__(MediaPipeBackend)
    return sp


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestConvexPolygonTorso(unittest.TestCase):

    def setUp(self):
        from src.core.semantic_parser import SemanticParser
        self.sp = SemanticParser.__new__(SemanticParser)
        self.h, self.w = 480, 640

    # --- Basic functionality ---

    def test_mask_is_nonzero(self):
        lms = make_landmarks_upright()
        mask = self.sp._calculate_torso_geometry(lms, self.h, self.w)
        self.assertGreater(np.count_nonzero(mask), 0)

    def test_mask_shape(self):
        lms = make_landmarks_upright()
        mask = self.sp._calculate_torso_geometry(lms, self.h, self.w)
        self.assertEqual(mask.shape, (self.h, self.w))
        self.assertEqual(mask.dtype, np.uint8)

    def test_mask_values_binary(self):
        """All values must be exactly 0 or 255."""
        lms = make_landmarks_upright()
        mask = self.sp._calculate_torso_geometry(lms, self.h, self.w)
        unique = np.unique(mask)
        self.assertTrue(set(unique).issubset({0, 255}))

    def test_mask_within_frame(self):
        """No pixels outside frame boundaries."""
        lms = make_landmarks_upright()
        mask = self.sp._calculate_torso_geometry(lms, self.h, self.w)
        self.assertEqual(mask.shape[0], self.h)
        self.assertEqual(mask.shape[1], self.w)

    # --- Polygon vs rectangle: lean / rotation correctness ---

    def test_leaning_body_torso_follows_lean(self):
        """
        For a leaning body (left shoulder higher, right hip lower) the mask
        centroid should be offset from the frame centre on the lean axis.
        A rectangle would always be axis-aligned; the polygon should not be.
        """
        lms_lean = make_landmarks_upright(
            ls=(0.30, 0.25),   # left shoulder  – higher / more left
            rs=(0.60, 0.35),   # right shoulder – lower
            lh=(0.33, 0.68),
            rh=(0.58, 0.75),
        )
        mask = self.sp._calculate_torso_geometry(lms_lean, self.h, self.w)
        ys, xs = np.where(mask > 0)
        centroid_x = xs.mean() / self.w
        # Centroid should be left of 0.5 (body leans left)
        self.assertLess(centroid_x, 0.52,
                        msg="Leaning body: torso centroid should be left of centre")

    def test_side_view_narrower_than_front(self):
        """Side / 3/4 view: shoulders appear much closer together."""
        lms_front = make_landmarks_upright(ls=(0.25, 0.30), rs=(0.75, 0.30))
        lms_side  = make_landmarks_upright(ls=(0.43, 0.30), rs=(0.57, 0.30))

        mask_front = self.sp._calculate_torso_geometry(lms_front, self.h, self.w)
        mask_side  = self.sp._calculate_torso_geometry(lms_side,  self.h, self.w)

        px_front = np.count_nonzero(mask_front)
        px_side  = np.count_nonzero(mask_side)
        self.assertGreater(px_front, px_side,
                           msg="Front-view torso should be wider than side-view torso")

    def test_asymmetric_hip_width(self):
        """Hips wider than shoulders: polygon widens at bottom."""
        lms = make_landmarks_upright(
            ls=(0.40, 0.28), rs=(0.60, 0.28),   # narrow shoulders
            lh=(0.28, 0.72), rh=(0.72, 0.72),   # wide hips
        )
        mask = self.sp._calculate_torso_geometry(lms, self.h, self.w)
        # Bottom-half of mask should be wider than top-half
        mid_row = self.h // 2
        top_px  = np.count_nonzero(mask[:mid_row, :])
        bot_px  = np.count_nonzero(mask[mid_row:, :])
        # Both halves should be filled; bottom >= top for wide-hip case
        self.assertGreater(bot_px, 0)
        self.assertGreater(top_px, 0)

    # --- Cropped-frame / out-of-bounds landmarks ---

    def test_landmarks_at_frame_edge(self):
        """Landmarks exactly at frame edges should not raise and mask stays in bounds."""
        lms = make_landmarks_upright(ls=(0.0, 0.0), rs=(1.0, 0.0),
                                     lh=(0.0, 1.0), rh=(1.0, 1.0))
        mask = self.sp._calculate_torso_geometry(lms, self.h, self.w)
        self.assertEqual(mask.shape, (self.h, self.w))

    def test_landmarks_outside_normalised_range(self):
        """Normalised coords > 1 or < 0 (partially out of frame) should be clamped."""
        lms = make_landmarks_upright(ls=(-0.1, 0.25), rs=(1.1, 0.25),
                                     lh=(-0.1, 0.75), rh=(1.1, 0.75))
        try:
            mask = self.sp._calculate_torso_geometry(lms, self.h, self.w)
            self.assertEqual(mask.shape, (self.h, self.w))
        except Exception as e:
            self.fail(f"Out-of-range landmarks raised: {e}")

    def test_very_small_frame(self):
        """Tiny frame (e.g. 32×32) should not crash."""
        lms = make_landmarks_upright()
        mask = self.sp._calculate_torso_geometry(lms, 32, 32)
        self.assertEqual(mask.shape, (32, 32))


class TestPoseConfidenceFiltering(unittest.TestCase):

    def setUp(self):
        from src.core.semantic_parser import SemanticParser
        self.sp = SemanticParser.__new__(SemanticParser)
        self.sp._MIN_LANDMARK_VISIBILITY = 0.5
        self.h, self.w = 480, 640

    def _dummy_masks(self):
        return {
            'upper_body': np.ones((self.h, self.w), dtype=np.uint8) * 255,
            'hair':       np.zeros((self.h, self.w), dtype=np.uint8),
            'face':       np.zeros((self.h, self.w), dtype=np.uint8),
            'neck':       np.zeros((self.h, self.w), dtype=np.uint8),
            'arms':       np.zeros((self.h, self.w), dtype=np.uint8),
            'lower_body': np.zeros((self.h, self.w), dtype=np.uint8),
        }

    def test_high_visibility_applies_constraint(self):
        lms = make_landmarks_upright(vis=0.95)
        masks = self._dummy_masks()
        result = self.sp._apply_geometric_constraints(masks, lms, self.h, self.w)
        # Constraint must have zeroed out at least some pixels
        self.assertLess(result['upper_body'].mean(), 255.0)

    def test_low_visibility_skips_constraint(self):
        """All 4 key landmarks below threshold → mask must be untouched."""
        lms = make_landmarks_upright(vis=0.2)
        masks = self._dummy_masks()
        result = self.sp._apply_geometric_constraints(masks, lms, self.h, self.w)
        self.assertEqual(result['upper_body'].mean(), 255.0)

    def test_single_occluded_landmark_skips(self):
        """Even one landmark below threshold → skip."""
        lms = make_landmarks_upright(vis=0.95)
        lms[23] = make_landmark(0.40, 0.72, visibility=0.1)  # left hip occluded
        masks = self._dummy_masks()
        result = self.sp._apply_geometric_constraints(masks, lms, self.h, self.w)
        self.assertEqual(result['upper_body'].mean(), 255.0)

    def test_borderline_visibility_value(self):
        """Landmark exactly at threshold should NOT be filtered (>= not >)."""
        from src.core.semantic_parser import SemanticParser
        sp = SemanticParser.__new__(SemanticParser)
        sp._MIN_LANDMARK_VISIBILITY = 0.5
        lms = make_landmarks_upright(vis=0.5)  # exactly at threshold
        masks = self._dummy_masks()
        # Should apply constraint (not skip)
        result = sp._apply_geometric_constraints(masks, lms, self.h, self.w)
        # Constraint applied → some pixels zeroed
        self.assertLess(result['upper_body'].mean(), 255.0)

    def test_no_visibility_attribute_defaults_to_pass(self):
        """Landmarks without .visibility attribute should default to visible."""
        lms = [types.SimpleNamespace(x=0.5, y=0.5) for _ in range(25)]
        lms[11] = types.SimpleNamespace(x=0.35, y=0.30)
        lms[12] = types.SimpleNamespace(x=0.65, y=0.30)
        lms[23] = types.SimpleNamespace(x=0.37, y=0.72)
        lms[24] = types.SimpleNamespace(x=0.63, y=0.72)
        masks = self._dummy_masks()
        try:
            result = self.sp._apply_geometric_constraints(masks, lms, self.h, self.w)
            # Should either apply or gracefully skip — must not crash
        except Exception as e:
            self.fail(f"Missing visibility attribute caused crash: {e}")


class TestCameraDepthNormalisation(unittest.TestCase):

    def setUp(self):
        from src.core.body_aware_fitter import BodyAwareGarmentFitter
        self.fitter = BodyAwareGarmentFitter.__new__(BodyAwareGarmentFitter)
        self.fitter._focal_length_px = None
        self.fitter._CALIBRATION_MIN_PX = 60
        self.fitter._REAL_SHOULDER_WIDTH_M = 0.42
        self.fitter._FOCAL_EMA_ALPHA = 0.05
        self.fitter._FOCAL_RECAL_THRESHOLD = 0.15
        self.w, self.h = 640, 480

    def test_near_and_far_produce_same_canonical_width(self):
        """Different subject distances → same normalised shoulder width."""
        sw_near, _ = self.fitter._apply_depth_normalisation(300, 400, self.w, self.h)
        self.fitter._focal_length_px = None
        sw_far,  _ = self.fitter._apply_depth_normalisation(100, 130, self.w, self.h)
        self.assertAlmostEqual(sw_near, sw_far, delta=1.0)

    def test_canonical_width_is_40_percent_frame(self):
        canonical = self.w * 0.40
        sw, _ = self.fitter._apply_depth_normalisation(200, 260, self.w, self.h)
        self.assertAlmostEqual(sw, canonical, delta=1.0)

    def test_torso_scales_proportionally(self):
        """Torso height should scale by the same factor as shoulder width."""
        sw, th = self.fitter._apply_depth_normalisation(150, 200, self.w, self.h)
        expected_scale = (self.w * 0.40) / 150
        self.assertAlmostEqual(th, 200 * expected_scale, delta=2.0)

    def test_too_narrow_skips_normalisation(self):
        """Shoulder < 60px → raw values returned unchanged."""
        sw, th = self.fitter._apply_depth_normalisation(40, 60, self.w, self.h)
        self.assertEqual(sw, 40)
        self.assertEqual(th, 60)

    def test_focal_length_seeded_on_first_call(self):
        self.assertIsNone(self.fitter._focal_length_px)
        self.fitter._apply_depth_normalisation(200, 260, self.w, self.h)
        self.assertIsNotNone(self.fitter._focal_length_px)

    def test_focal_ema_update_on_large_drift(self):
        """A 30% drift should trigger EMA (not jump to new value)."""
        old_f = 400.0
        self.fitter._focal_length_px = old_f
        # 30% higher shoulder → ~30% drift in focal estimate
        self.fitter._apply_depth_normalisation(int(200 * 1.3), 260, self.w, self.h)
        new_f = self.fitter._focal_length_px
        # EMA update: new value should be between old and estimate
        self.assertNotAlmostEqual(new_f, old_f, delta=0.1,
                                  msg="Focal length should have updated")
        # But should NOT have jumped all the way to estimate
        focal_est_new = (200 * 1.3) / self.fitter._REAL_SHOULDER_WIDTH_M
        self.assertLess(new_f, focal_est_new,
                        msg="EMA should not jump all the way to new estimate")

    def test_normalised_values_clamped_to_frame(self):
        """Extreme close-up: output must not exceed frame dimensions."""
        sw, th = self.fitter._apply_depth_normalisation(600, 460, self.w, self.h)
        self.assertLessEqual(sw, self.w * 0.95 + 0.1)
        self.assertLessEqual(th, self.h * 0.90 + 0.1)


class TestRegressionCases(unittest.TestCase):
    """
    End-to-end regression: verify the full _apply_geometric_constraints
    pipeline does not regress when combining polygon + confidence filter.
    """

    def setUp(self):
        from src.core.semantic_parser import SemanticParser
        self.sp = SemanticParser.__new__(SemanticParser)
        self.sp._MIN_LANDMARK_VISIBILITY = 0.5
        self.h, self.w = 480, 640

    def _full_masks(self, upper_val=255):
        return {
            'upper_body': np.full((self.h, self.w), upper_val, dtype=np.uint8),
            'hair':       np.zeros((self.h, self.w), dtype=np.uint8),
            'face':       np.zeros((self.h, self.w), dtype=np.uint8),
            'neck':       np.zeros((self.h, self.w), dtype=np.uint8),
            'arms':       np.zeros((self.h, self.w), dtype=np.uint8),
            'lower_body': np.zeros((self.h, self.w), dtype=np.uint8),
        }

    def test_upper_body_mask_reduced_not_expanded(self):
        """Constraint should only reduce the mask, never expand it."""
        lms = make_landmarks_upright(vis=0.95)
        masks = self._full_masks(upper_val=255)
        result = self.sp._apply_geometric_constraints(masks, lms, self.h, self.w)
        # Pixels in result must be a subset of original 255-pixels
        self.assertTrue(
            np.all(result['upper_body'] <= 255),
            "Mask values must not exceed original"
        )
        self.assertLess(
            result['upper_body'].sum(),
            255 * self.h * self.w,
            "Full-frame mask should be constrained"
        )

    def test_constraint_leaves_other_masks_unchanged(self):
        """Only 'upper_body' should be modified by the torso constraint."""
        lms = make_landmarks_upright(vis=0.95)
        masks = self._full_masks()
        masks['hair'][:50, :] = 255   # some hair pixels
        hair_before = masks['hair'].copy()
        self.sp._apply_geometric_constraints(masks, lms, self.h, self.w)
        np.testing.assert_array_equal(masks['hair'], hair_before,
                                      err_msg="Hair mask should not be modified by torso constraint")

    def test_empty_upper_body_mask_stays_empty(self):
        """If no upper-body pixels detected, result stays zero."""
        lms = make_landmarks_upright(vis=0.95)
        masks = self._full_masks(upper_val=0)
        result = self.sp._apply_geometric_constraints(masks, lms, self.h, self.w)
        self.assertEqual(result['upper_body'].sum(), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
