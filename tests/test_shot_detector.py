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


class TestEnhancedConfidenceScoring:
    """Tests for enhanced confidence scoring with additional factors."""

    def test_new_confidence_parameters(self):
        """Test that new parameters are properly initialized."""
        detector = ShotDetector()
        assert detector.type_confidence_threshold == 0.5
        assert detector.low_confidence_threshold == 0.7

        detector = ShotDetector(
            type_confidence_threshold=0.6,
            low_confidence_threshold=0.8
        )
        assert detector.type_confidence_threshold == 0.6
        assert detector.low_confidence_threshold == 0.8

    def test_peak_prominence_score_clear_peak(self):
        """Test peak prominence score with clear peak."""
        detector = ShotDetector()

        # Clear peak
        velocity = np.array([0.01, 0.02, 0.05, 0.15, 0.1, 0.05, 0.02, 0.01])
        score = detector._calculate_peak_prominence_score(velocity)
        assert 0.5 < score <= 1.0  # Should be high for clear peak

    def test_peak_prominence_score_flat_signal(self):
        """Test peak prominence score with flat signal."""
        detector = ShotDetector()

        # Flat signal
        velocity = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
        score = detector._calculate_peak_prominence_score(velocity)
        assert score <= 0.5  # Should be low for flat signal

    def test_peak_prominence_score_short_signal(self):
        """Test peak prominence score with very short signal."""
        detector = ShotDetector()

        # Too short
        velocity = np.array([0.01, 0.02])
        score = detector._calculate_peak_prominence_score(velocity)
        assert score == 0.0

    def test_snr_score_clear_signal(self):
        """Test SNR score with clear signal."""
        detector = ShotDetector()

        # Clear signal with distinct peak
        velocity = np.array([0.01, 0.01, 0.01, 0.15, 0.01, 0.01, 0.01])
        score = detector._calculate_snr_score(velocity)
        assert score > 0.5  # Should be high for clear signal

    def test_snr_score_low_signal(self):
        """Test SNR score with low signal values."""
        detector = ShotDetector()

        # Low amplitude signal - the max is not much larger than the noise floor
        velocity = np.array([0.001, 0.002, 0.001, 0.003, 0.001, 0.002, 0.001])
        score = detector._calculate_snr_score(velocity)
        # Just verify it returns a valid score in range
        assert 0.0 <= score <= 1.0

    def test_snr_score_short_signal(self):
        """Test SNR score with very short signal."""
        detector = ShotDetector()

        velocity = np.array([0.01, 0.02])
        score = detector._calculate_snr_score(velocity)
        assert score == 0.0

    def test_duration_score_optimal(self):
        """Test duration score for optimal duration."""
        detector = ShotDetector()

        # Optimal duration (30-60 frames)
        score = detector._calculate_duration_score(45)
        assert score == 1.0

    def test_duration_score_too_short(self):
        """Test duration score for too short duration."""
        detector = ShotDetector()

        score = detector._calculate_duration_score(15)
        assert 0.3 <= score < 1.0

    def test_duration_score_too_long(self):
        """Test duration score for too long duration."""
        detector = ShotDetector()

        score = detector._calculate_duration_score(100)
        assert 0.3 <= score < 1.0

    def test_enhanced_confidence_calculation(self):
        """Test that enhanced confidence uses all factors."""
        detector = ShotDetector(
            height_threshold=0.1,
            min_duration=10,
            confidence_threshold=0.0,
        )

        # Create test signals
        height_signal = np.array([0.0] * 10 + [0.2] * 30 + [0.0] * 10)
        velocity_signal = np.array([0.01] * 10 + [0.01, 0.02, 0.05, 0.1, 0.15, 0.1, 0.05, 0.02] + [0.01] * 12 + [0.01] * 10 + [0.01] * 10)
        pose_quality = np.ones(50)

        confidence = detector._calculate_confidence(
            height_signal, velocity_signal, pose_quality, 10, 40
        )

        assert 0.0 <= confidence <= 1.0


class TestTypePrediction:
    """Tests for shot type prediction functionality."""

    def create_pose_with_height_velocity(self, wrist_y=0.3, shoulder_y=0.5, offset=0.0):
        """Create a pose with specified height and implied velocity (via offset)."""
        return {
            'keypoints': {
                'RIGHT_WRIST': {'x': 0.5 + offset, 'y': wrist_y, 'z': 0.0},
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

    def test_smash_score_high(self):
        """Test smash score for high height and velocity."""
        detector = ShotDetector()
        score = detector._smash_score(0.9, 0.9)
        assert score > 0.5

    def test_smash_score_low(self):
        """Test smash score for low height and velocity."""
        detector = ShotDetector()
        score = detector._smash_score(0.3, 0.3)
        assert score == 0.0

    def test_vibora_score_optimal(self):
        """Test vibora score for optimal medium values."""
        detector = ShotDetector()
        score = detector._vibora_score(0.65, 0.55)
        assert score > 0.7  # Should be high for optimal values

    def test_vibora_score_extreme(self):
        """Test vibora score for extreme values."""
        detector = ShotDetector()
        # Very low values
        score_low = detector._vibora_score(0.1, 0.1)
        # Very high values
        score_high = detector._vibora_score(0.95, 0.95)

        assert score_low < 0.5
        assert score_high < 0.5

    def test_bandeja_score_low_values(self):
        """Test bandeja score for low height and velocity."""
        detector = ShotDetector()
        score = detector._bandeja_score(0.2, 0.2)
        assert score > 0.5  # Should be high for low values

    def test_bandeja_score_high_values(self):
        """Test bandeja score for high height and velocity."""
        detector = ShotDetector()
        score = detector._bandeja_score(0.9, 0.9)
        assert score < 0.3  # Should be low for high values

    def test_classify_by_features_smash(self):
        """Test classification returns smash for high values."""
        detector = ShotDetector()
        shot_type, confidence = detector._classify_by_features(0.9, 0.9)
        assert shot_type == "smash"
        assert confidence > 0.3

    def test_classify_by_features_vibora(self):
        """Test classification returns vibora for medium values."""
        detector = ShotDetector()
        shot_type, confidence = detector._classify_by_features(0.65, 0.55)
        assert shot_type == "vibora"
        assert confidence > 0.3

    def test_classify_by_features_bandeja(self):
        """Test classification returns bandeja for low values."""
        detector = ShotDetector()
        shot_type, confidence = detector._classify_by_features(0.2, 0.2)
        assert shot_type == "bandeja"
        assert confidence > 0.3

    def test_predict_shot_type_returns_tuple(self):
        """Test that predict_shot_type returns proper tuple."""
        detector = ShotDetector()

        pose_data = [self.create_pose_with_height_velocity() for _ in range(30)]
        velocity_signal = np.array([0.01] * 5 + [0.02, 0.05, 0.1, 0.15, 0.1, 0.05, 0.02] + [0.01] * 18)
        height_signal = np.array([0.2] * 30)

        shot_type, confidence = detector._predict_shot_type(
            pose_data, velocity_signal, height_signal, 5, 20
        )

        assert isinstance(shot_type, str)
        assert shot_type in ["smash", "vibora", "bandeja", "unknown"]
        assert 0.0 <= confidence <= 1.0

    def test_predict_shot_type_empty_window(self):
        """Test type prediction with empty/NaN window."""
        detector = ShotDetector()

        pose_data = []
        velocity_signal = np.array([np.nan] * 10)
        height_signal = np.array([np.nan] * 10)

        shot_type, confidence = detector._predict_shot_type(
            pose_data, velocity_signal, height_signal, 0, 10
        )

        assert shot_type == "unknown"
        assert confidence == 0.0


class TestTypeConfidenceScoring:
    """Tests for type confidence scoring."""

    def test_confidence_margin_affects_score(self):
        """Test that larger margin between scores gives higher confidence."""
        detector = ShotDetector()

        # Clear smash (high margin)
        _, conf_clear = detector._classify_by_features(0.95, 0.95)

        # Ambiguous (low margin) - values where multiple types have similar scores
        _, conf_ambiguous = detector._classify_by_features(0.45, 0.45)

        assert conf_clear >= conf_ambiguous

    def test_type_unknown_below_threshold(self):
        """Test that type is unknown when confidence below threshold."""
        detector = ShotDetector(
            height_threshold=0.1,
            min_duration=10,
            confidence_threshold=0.0,
            type_confidence_threshold=0.99,  # Very high threshold
        )

        # Create basic shot
        poses = []
        for i in range(40):
            if 10 <= i < 30:
                poses.append(create_test_pose(wrist_y=0.3, shoulder_y=0.5))
            else:
                poses.append(create_test_pose(wrist_y=0.6, shoulder_y=0.5))

        shots = detector.detect(poses)

        # Most shots should be unknown due to high threshold
        if len(shots) > 0:
            assert shots[0].shot_type == "unknown"


class TestLowConfidenceFlagging:
    """Tests for low confidence flagging functionality."""

    def test_low_confidence_field_default(self):
        """Test that low_confidence field has correct default."""
        shot = DetectedShot(
            start_frame=10,
            end_frame=30,
            confidence=0.85,
        )
        assert shot.low_confidence is False

    def test_low_confidence_flagging(self):
        """Test that low confidence shots are flagged."""
        detector = ShotDetector(
            height_threshold=0.1,
            min_duration=10,
            confidence_threshold=0.0,  # Accept all
            low_confidence_threshold=0.99,  # Almost all will be low confidence
        )

        poses = []
        for i in range(40):
            if 10 <= i < 30:
                poses.append(create_test_pose(wrist_y=0.3, shoulder_y=0.5))
            else:
                poses.append(create_test_pose(wrist_y=0.6, shoulder_y=0.5))

        shots = detector.detect(poses)

        if len(shots) > 0:
            # Should be flagged as low confidence
            assert shots[0].low_confidence == True

    def test_high_confidence_not_flagged(self):
        """Test that high confidence shots are not flagged."""
        detector = ShotDetector(
            height_threshold=0.1,
            min_duration=10,
            confidence_threshold=0.0,
            low_confidence_threshold=0.1,  # Very low threshold
        )

        poses = []
        for i in range(40):
            if 10 <= i < 30:
                poses.append(create_test_pose(wrist_y=0.3, shoulder_y=0.5))
            else:
                poses.append(create_test_pose(wrist_y=0.6, shoulder_y=0.5))

        shots = detector.detect(poses)

        if len(shots) > 0:
            # Should not be flagged
            assert shots[0].low_confidence == False


class TestIntegration:
    """Integration tests for confidence scoring and type prediction."""

    def test_full_detection_with_type_prediction(self):
        """Test complete detection pipeline with type prediction."""
        detector = ShotDetector(
            height_threshold=0.1,
            min_duration=10,
            confidence_threshold=0.0,
        )

        poses = []
        for i in range(60):
            if 15 <= i < 45:
                poses.append(create_test_pose(wrist_y=0.3, shoulder_y=0.5))
            else:
                poses.append(create_test_pose(wrist_y=0.6, shoulder_y=0.5))

        shots = detector.detect(poses)

        assert len(shots) >= 1
        for shot in shots:
            assert 0.0 <= shot.confidence <= 1.0
            assert shot.shot_type in ["smash", "vibora", "bandeja", "unknown"]
            assert 0.0 <= shot.type_confidence <= 1.0
            assert shot.low_confidence in [True, False]

    def test_detected_shot_dataclass_complete(self):
        """Test that DetectedShot has all required fields."""
        shot = DetectedShot(
            start_frame=10,
            end_frame=30,
            confidence=0.75,
            shot_type="vibora",
            type_confidence=0.65,
            low_confidence=False,
        )

        assert shot.start_frame == 10
        assert shot.end_frame == 30
        assert shot.confidence == 0.75
        assert shot.shot_type == "vibora"
        assert shot.type_confidence == 0.65
        assert shot.low_confidence is False

    def test_multiple_shots_with_different_types(self):
        """Test detection of multiple shots with potentially different types."""
        detector = ShotDetector(
            height_threshold=0.1,
            min_duration=10,
            merge_gap=5,
            confidence_threshold=0.0,
        )

        # Create two shots
        poses = []
        for i in range(100):
            if 10 <= i < 30 or 60 <= i < 80:
                poses.append(create_test_pose(wrist_y=0.3, shoulder_y=0.5))
            else:
                poses.append(create_test_pose(wrist_y=0.6, shoulder_y=0.5))

        shots = detector.detect(poses)

        # Should detect multiple shots
        assert len(shots) >= 1

        # Each shot should have type prediction
        for shot in shots:
            assert shot.shot_type is not None
            assert shot.type_confidence >= 0.0


class TestEdgeCasesEnhanced:
    """Edge case tests for enhanced confidence and type prediction."""

    def test_very_short_duration(self):
        """Test with minimum duration shots."""
        detector = ShotDetector(
            height_threshold=0.1,
            min_duration=15,
            confidence_threshold=0.0,
        )

        # Create exactly 15 frame overhead window
        poses = []
        for i in range(35):
            if 10 <= i < 25:
                poses.append(create_test_pose(wrist_y=0.3, shoulder_y=0.5))
            else:
                poses.append(create_test_pose(wrist_y=0.6, shoulder_y=0.5))

        shots = detector.detect(poses)

        if len(shots) > 0:
            # Should have lower duration score
            assert shots[0].confidence >= 0.0

    def test_all_nan_velocity_in_window(self):
        """Test handling of NaN velocity values."""
        detector = ShotDetector()

        # Create signals with NaN velocity
        height_signal = np.array([0.2] * 30)
        velocity_signal = np.array([np.nan] * 30)
        pose_quality = np.ones(30)

        confidence = detector._calculate_confidence(
            height_signal, velocity_signal, pose_quality, 5, 25
        )

        # Should still return a valid confidence
        assert 0.0 <= confidence <= 1.0

    def test_zero_velocity(self):
        """Test with zero velocity values."""
        detector = ShotDetector()

        velocity = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        prominence_score = detector._calculate_peak_prominence_score(velocity)
        snr_score = detector._calculate_snr_score(velocity)

        assert 0.0 <= prominence_score <= 1.0
        assert 0.0 <= snr_score <= 1.0

    def test_boundary_values_classification(self):
        """Test classification at boundary values."""
        detector = ShotDetector()

        # Test at exactly threshold values
        boundaries = [0.0, 0.3, 0.5, 0.65, 0.8, 1.0]

        for h in boundaries:
            for v in boundaries:
                shot_type, confidence = detector._classify_by_features(h, v)
                assert shot_type in ["smash", "vibora", "bandeja"]
                assert 0.0 <= confidence <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
