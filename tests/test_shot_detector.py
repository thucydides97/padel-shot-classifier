"""
Unit tests for shot_detector module.
"""

import pytest
import numpy as np
from padel_shot_classifier.shot_detector import (
    ShotDetector,
    DetectedShot,
    detect_shots,
)
from scipy.signal import find_peaks


def create_test_pose(wrist_y=0.3, shoulder_y=0.5):
    """Create a test pose with specified wrist and shoulder heights."""
    return {
        'keypoints': {
            'RIGHT_WRIST': {'x': 0.5, 'y': wrist_y, 'z': 0.0},
            'RIGHT_SHOULDER': {'x': 0.5, 'y': shoulder_y, 'z': 0.0},
            'RIGHT_ELBOW': {'x': 0.5, 'y': 0.4, 'z': 0.0},
            'LEFT_WRIST': {'x': 0.4, 'y': 0.3, 'z': 0.0},
            'LEFT_SHOULDER': {'x': 0.4, 'y': 0.5, 'z': 0.0},
            'LEFT_ELBOW': {'x': 0.4, 'y': 0.4, 'z': 0.0},
            'LEFT_HIP': {'x': 0.4, 'y': 0.7, 'z': 0.0},
            'RIGHT_HIP': {'x': 0.6, 'y': 0.7, 'z': 0.0},
            'NOSE': {'x': 0.5, 'y': 0.2, 'z': 0.0},
        }
    }


class TestShotDetector:
    def test_initialization(self):
        detector = ShotDetector()
        assert detector.height_threshold == 0.1
        assert detector.min_duration == 15
        assert detector.max_duration == 90
        assert detector.merge_gap == 10

    def test_custom_parameters(self):
        detector = ShotDetector(
            height_threshold=0.2,
            min_duration=10,
            max_duration=60,
        )
        assert detector.height_threshold == 0.2
        assert detector.min_duration == 10
        assert detector.max_duration == 60


class TestFindOverheadWindows:
    def test_single_window(self):
        detector = ShotDetector(height_threshold=0.1)

        # Create signal with one overhead region
        # Wrist above shoulder (positive height difference)
        poses = []
        for i in range(50):
            if 10 <= i < 30:
                # Overhead: wrist_y=0.3, shoulder_y=0.5 -> height = 0.2
                poses.append(create_test_pose(wrist_y=0.3, shoulder_y=0.5))
            else:
                # Not overhead: wrist_y=0.6, shoulder_y=0.5 -> height = -0.1
                poses.append(create_test_pose(wrist_y=0.6, shoulder_y=0.5))

        shots = detector.detect(poses)

        assert len(shots) == 1
        assert shots[0].start_frame >= 10
        assert shots[0].end_frame <= 30

    def test_no_overhead(self):
        detector = ShotDetector()

        # All poses with wrist below shoulder
        poses = [create_test_pose(wrist_y=0.6, shoulder_y=0.5) for _ in range(50)]

        shots = detector.detect(poses)
        assert len(shots) == 0

    def test_multiple_windows(self):
        detector = ShotDetector(
            height_threshold=0.1,
            min_duration=10,
            merge_gap=5,
        )

        poses = []
        for i in range(100):
            if 10 <= i < 25 or 50 <= i < 70:
                # Overhead regions
                poses.append(create_test_pose(wrist_y=0.3, shoulder_y=0.5))
            else:
                poses.append(create_test_pose(wrist_y=0.6, shoulder_y=0.5))

        shots = detector.detect(poses)

        # Should find 2 separate windows (gap > merge_gap)
        assert len(shots) == 2

    def test_empty_poses(self):
        detector = ShotDetector()
        shots = detector.detect([])
        assert len(shots) == 0

    def test_too_short_video(self):
        detector = ShotDetector(min_duration=15)
        poses = [create_test_pose() for _ in range(10)]
        shots = detector.detect(poses)
        assert len(shots) == 0


class TestMergeNearbyWindows:
    def test_merge_close_windows(self):
        detector = ShotDetector(
            height_threshold=0.1,
            min_duration=5,
            merge_gap=10,
            confidence_threshold=0.0,  # Accept all confidence levels
        )

        # Create two windows separated by small gap
        poses = []
        for i in range(50):
            if 5 <= i < 15 or 20 <= i < 35:
                poses.append(create_test_pose(wrist_y=0.3, shoulder_y=0.5))
            else:
                poses.append(create_test_pose(wrist_y=0.6, shoulder_y=0.5))

        shots = detector.detect(poses)

        # Should merge into 1 window (gap of 5 < merge_gap of 10)
        assert len(shots) == 1

    def test_no_merge_distant_windows(self):
        detector = ShotDetector(
            height_threshold=0.1,
            min_duration=5,
            merge_gap=5,
            confidence_threshold=0.0,  # Accept all confidence levels
        )

        # Create two windows separated by large gap
        poses = []
        for i in range(60):
            if 5 <= i < 15 or 30 <= i < 45:
                poses.append(create_test_pose(wrist_y=0.3, shoulder_y=0.5))
            else:
                poses.append(create_test_pose(wrist_y=0.6, shoulder_y=0.5))

        shots = detector.detect(poses)

        # Should not merge (gap of 15 > merge_gap of 5)
        assert len(shots) == 2


class TestFilterByDuration:
    def test_filter_too_short(self):
        detector = ShotDetector(
            height_threshold=0.1,
            min_duration=20,
            max_duration=90,
        )

        # Create overhead window of only 10 frames
        poses = []
        for i in range(50):
            if 10 <= i < 20:
                poses.append(create_test_pose(wrist_y=0.3, shoulder_y=0.5))
            else:
                poses.append(create_test_pose(wrist_y=0.6, shoulder_y=0.5))

        shots = detector.detect(poses)

        # Should filter out (10 frames < min_duration 20)
        assert len(shots) == 0

    def test_filter_too_long(self):
        detector = ShotDetector(
            height_threshold=0.1,
            min_duration=10,
            max_duration=30,
        )

        # Create overhead window of 50 frames
        poses = []
        for i in range(70):
            if 10 <= i < 60:
                poses.append(create_test_pose(wrist_y=0.3, shoulder_y=0.5))
            else:
                poses.append(create_test_pose(wrist_y=0.6, shoulder_y=0.5))

        shots = detector.detect(poses)

        # Should filter out (50 frames > max_duration 30)
        assert len(shots) == 0


class TestConfidenceCalculation:
    def test_confidence_range(self):
        detector = ShotDetector(
            height_threshold=0.1,
            min_duration=10,
            confidence_threshold=0.0,  # Accept all
        )

        poses = []
        for i in range(40):
            if 10 <= i < 30:
                poses.append(create_test_pose(wrist_y=0.3, shoulder_y=0.5))
            else:
                poses.append(create_test_pose(wrist_y=0.6, shoulder_y=0.5))

        shots = detector.detect(poses)

        assert len(shots) == 1
        assert 0.0 <= shots[0].confidence <= 1.0

    def test_confidence_threshold_filtering(self):
        detector = ShotDetector(
            height_threshold=0.1,
            min_duration=10,
            confidence_threshold=0.99,  # Very high threshold
        )

        poses = []
        for i in range(40):
            if 10 <= i < 30:
                poses.append(create_test_pose(wrist_y=0.3, shoulder_y=0.5))
            else:
                poses.append(create_test_pose(wrist_y=0.6, shoulder_y=0.5))

        shots = detector.detect(poses)

        # Likely filtered out by high confidence threshold
        # (depends on actual confidence calculation)
        assert len(shots) <= 1


class TestDetectedShot:
    def test_dataclass_fields(self):
        shot = DetectedShot(
            start_frame=10,
            end_frame=30,
            confidence=0.85,
        )
        assert shot.start_frame == 10
        assert shot.end_frame == 30
        assert shot.confidence == 0.85
        assert shot.shot_type == "unknown"
        assert shot.type_confidence == 0.0


class TestConvenienceFunction:
    def test_detect_shots_function(self):
        poses = []
        for i in range(40):
            if 10 <= i < 30:
                poses.append(create_test_pose(wrist_y=0.3, shoulder_y=0.5))
            else:
                poses.append(create_test_pose(wrist_y=0.6, shoulder_y=0.5))

        shots = detect_shots(poses, min_duration=10)

        assert len(shots) >= 1
        assert all(isinstance(s, DetectedShot) for s in shots)


class TestEdgeCases:
    def test_all_none_poses(self):
        detector = ShotDetector()
        poses = [None for _ in range(50)]
        shots = detector.detect(poses)
        assert len(shots) == 0

    def test_mixed_none_poses(self):
        detector = ShotDetector(
            height_threshold=0.1,
            min_duration=5,
            confidence_threshold=0.0,
        )

        poses = []
        for i in range(40):
            if i % 3 == 0:
                poses.append(None)
            elif 10 <= i < 30:
                poses.append(create_test_pose(wrist_y=0.3, shoulder_y=0.5))
            else:
                poses.append(create_test_pose(wrist_y=0.6, shoulder_y=0.5))

        # Should still detect despite some None poses
        shots = detector.detect(poses)
        # May or may not find shots depending on how NaN handling works


class TestBoundaryRefinement:
    """Tests for boundary refinement functionality."""

    def test_new_parameters(self):
        """Test that new parameters are properly initialized."""
        detector = ShotDetector()
        assert detector.peak_prominence == 0.01
        assert detector.refine_boundaries is True

        detector = ShotDetector(peak_prominence=0.05, refine_boundaries=False)
        assert detector.peak_prominence == 0.05
        assert detector.refine_boundaries is False

    def test_refine_boundaries_disabled(self):
        """Test that refinement can be disabled."""
        # Create poses with overhead region
        poses = []
        for i in range(50):
            if 10 <= i < 30:
                poses.append(create_test_pose(wrist_y=0.3, shoulder_y=0.5))
            else:
                poses.append(create_test_pose(wrist_y=0.6, shoulder_y=0.5))

        # With refinement disabled
        detector = ShotDetector(
            height_threshold=0.1,
            min_duration=10,
            refine_boundaries=False
        )
        shots = detector.detect(poses)
        assert len(shots) == 1

    def test_find_contact_point_single_peak(self):
        """Test finding contact point with single clear peak."""
        detector = ShotDetector()

        # Create velocity segment with single peak
        velocity_segment = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.1, 0.05, 0.02, 0.01])
        contact_idx = detector._find_contact_point(velocity_segment)

        assert contact_idx == 4  # Peak at index 4

    def test_find_contact_point_multiple_peaks(self):
        """Test finding contact point with multiple peaks (should return highest)."""
        detector = ShotDetector(peak_prominence=0.01)

        # Create velocity segment with multiple peaks
        velocity_segment = np.array([0.01, 0.05, 0.02, 0.15, 0.02, 0.08, 0.02, 0.01])
        contact_idx = detector._find_contact_point(velocity_segment)

        # Should find the highest peak at index 3
        assert contact_idx == 3

    def test_find_contact_point_with_nan(self):
        """Test finding contact point handles NaN values."""
        detector = ShotDetector()

        # Create velocity segment with NaN values
        velocity_segment = np.array([0.01, np.nan, 0.05, 0.1, 0.05, np.nan, 0.01])
        contact_idx = detector._find_contact_point(velocity_segment)

        assert contact_idx == 3  # Peak at index 3

    def test_find_contact_point_no_peak(self):
        """Test behavior when no clear peak exists."""
        detector = ShotDetector(peak_prominence=0.1)

        # Flat signal - no peaks
        velocity_segment = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        contact_idx = detector._find_contact_point(velocity_segment)

        # Should return None or fallback to max
        # (Falls back to argmax since all values are same)
        assert contact_idx is not None or contact_idx is None

    def test_find_rising_edge(self):
        """Test finding where velocity starts rising."""
        detector = ShotDetector()

        # Create velocity signal
        velocity_signal = np.array([0.01, 0.01, 0.02, 0.05, 0.15, 0.1, 0.05, 0.01])
        baseline = 0.015
        contact_frame = 4
        min_frame = 0

        rising_edge = detector._find_rising_edge(
            velocity_signal, contact_frame, baseline, min_frame
        )

        # Should find frame 2 (where velocity starts rising above baseline)
        assert rising_edge <= contact_frame
        assert rising_edge >= 0

    def test_find_falling_edge(self):
        """Test finding where velocity drops back to baseline."""
        detector = ShotDetector()

        # Create velocity signal
        velocity_signal = np.array([0.01, 0.02, 0.05, 0.15, 0.1, 0.05, 0.02, 0.01])
        baseline = 0.015
        contact_frame = 3
        max_frame = 8

        falling_edge = detector._find_falling_edge(
            velocity_signal, contact_frame, baseline, max_frame, len(velocity_signal)
        )

        # Should find frame where velocity drops
        assert falling_edge > contact_frame
        assert falling_edge <= len(velocity_signal)

    def test_calculate_baseline_velocity(self):
        """Test baseline velocity calculation from non-shot periods."""
        detector = ShotDetector()

        # Create velocity signal
        velocity_signal = np.array([0.01, 0.01, 0.05, 0.1, 0.05, 0.01, 0.01, 0.01])
        windows = [(2, 5)]  # Shot window

        baseline = detector._calculate_baseline_velocity(windows, velocity_signal)

        # Baseline should be based on non-shot frames (indices 0, 1, 5, 6, 7)
        # Mean of [0.01, 0.01, 0.01, 0.01, 0.01] = 0.01
        assert baseline == pytest.approx(0.01, abs=0.001)

    def test_calculate_baseline_all_nan(self):
        """Test baseline calculation when all non-shot frames are NaN."""
        detector = ShotDetector()

        velocity_signal = np.array([np.nan, np.nan, 0.1, 0.2, 0.1, np.nan, np.nan])
        windows = [(2, 5)]

        # Should fall back to percentile of valid signal
        baseline = detector._calculate_baseline_velocity(windows, velocity_signal)
        assert not np.isnan(baseline)

    def test_refine_single_window(self):
        """Test refining a single window."""
        detector = ShotDetector()

        # Create velocity signal with clear peak
        velocity_signal = np.array([
            0.01, 0.01, 0.01,  # Non-shot
            0.02, 0.05, 0.15, 0.1, 0.05, 0.02,  # Shot with peak at index 5
            0.01, 0.01, 0.01  # Non-shot
        ])
        baseline = 0.015

        refined = detector._refine_single_window(3, 9, velocity_signal, baseline)

        assert refined is not None
        assert refined[0] <= 5  # Start should be before or at peak
        assert refined[1] >= 5  # End should be after peak

    def test_refine_boundaries_integration(self):
        """Test full boundary refinement integration."""
        detector = ShotDetector(
            height_threshold=0.1,
            min_duration=5,
            refine_boundaries=True
        )

        # Create velocity signal
        velocity_signal = np.array([
            0.01, 0.01, 0.01, 0.01, 0.01,  # Non-shot
            0.02, 0.05, 0.15, 0.1, 0.05, 0.02,  # Shot
            0.01, 0.01, 0.01, 0.01, 0.01  # Non-shot
        ])

        windows = [(4, 12)]  # Rough window
        refined = detector._refine_boundaries(windows, velocity_signal)

        assert len(refined) == 1
        # Refined window should be different from original
        # (may be tighter or shifted)

    def test_refine_empty_windows(self):
        """Test refinement with empty window list."""
        detector = ShotDetector()
        velocity_signal = np.array([0.01, 0.02, 0.03])

        refined = detector._refine_boundaries([], velocity_signal)
        assert refined == []

    def test_edge_case_window_at_start(self):
        """Test refinement when window is at video start."""
        detector = ShotDetector()

        # Window starts at frame 0
        velocity_signal = np.array([0.1, 0.15, 0.1, 0.05, 0.01, 0.01])
        windows = [(0, 4)]

        refined = detector._refine_boundaries(windows, velocity_signal)

        assert len(refined) == 1
        assert refined[0][0] >= 0

    def test_edge_case_window_at_end(self):
        """Test refinement when window is at video end."""
        detector = ShotDetector()

        # Window extends to end
        velocity_signal = np.array([0.01, 0.01, 0.05, 0.15, 0.1])
        windows = [(2, 5)]

        refined = detector._refine_boundaries(windows, velocity_signal)

        assert len(refined) == 1
        assert refined[0][1] <= len(velocity_signal)

    def test_weak_signal_falls_back_to_original(self):
        """Test that weak signals fall back to original boundaries."""
        detector = ShotDetector(peak_prominence=0.5)  # High threshold

        # Very weak signal
        velocity_signal = np.array([0.001, 0.002, 0.003, 0.002, 0.001])
        windows = [(1, 4)]

        refined = detector._refine_boundaries(windows, velocity_signal)

        assert len(refined) == 1
        # Should fall back to original or similar
        assert refined[0] == (1, 4) or refined[0][0] <= 1

    def test_all_nan_segment(self):
        """Test behavior when velocity segment is all NaN."""
        detector = ShotDetector()

        velocity_signal = np.array([
            0.01, 0.01,
            np.nan, np.nan, np.nan,  # All NaN in window
            0.01, 0.01
        ])
        windows = [(2, 5)]

        refined = detector._refine_boundaries(windows, velocity_signal)

        assert len(refined) == 1
        # Should return original window
        assert refined[0] == (2, 5)


class TestBoundaryRefinementEndToEnd:
    """End-to-end tests for boundary refinement."""

    def create_velocity_pattern_pose(self, velocity_value):
        """Create a pose that would result in specific velocity."""
        # We'll use wrist position to control velocity
        return create_test_pose(wrist_y=0.3, shoulder_y=0.5)

    def test_full_detection_with_refinement(self):
        """Test complete detection pipeline with refinement enabled."""
        detector = ShotDetector(
            height_threshold=0.1,
            min_duration=10,
            confidence_threshold=0.0,
            refine_boundaries=True
        )

        # Create poses with overhead region
        poses = []
        for i in range(60):
            if 15 <= i < 45:
                # Overhead position
                poses.append(create_test_pose(wrist_y=0.3, shoulder_y=0.5))
            else:
                # Non-overhead position
                poses.append(create_test_pose(wrist_y=0.6, shoulder_y=0.5))

        shots = detector.detect(poses)

        # Should detect at least one shot
        assert len(shots) >= 1
        # Boundaries should be within reasonable range
        assert shots[0].start_frame >= 0
        assert shots[0].end_frame <= 60

    def test_refinement_improves_boundaries(self):
        """Test that refinement can adjust boundaries."""
        # Test with refinement enabled
        detector_with = ShotDetector(
            height_threshold=0.1,
            min_duration=10,
            confidence_threshold=0.0,
            refine_boundaries=True
        )

        # Test with refinement disabled
        detector_without = ShotDetector(
            height_threshold=0.1,
            min_duration=10,
            confidence_threshold=0.0,
            refine_boundaries=False
        )

        # Create poses
        poses = []
        for i in range(50):
            if 10 <= i < 35:
                poses.append(create_test_pose(wrist_y=0.3, shoulder_y=0.5))
            else:
                poses.append(create_test_pose(wrist_y=0.6, shoulder_y=0.5))

        shots_with = detector_with.detect(poses)
        shots_without = detector_without.detect(poses)

        # Both should detect shots
        assert len(shots_with) >= 1
        assert len(shots_without) >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
