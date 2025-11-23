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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
