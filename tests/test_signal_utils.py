"""
Unit tests for signal_utils module.
"""

import pytest
import numpy as np
from padel_shot_classifier.signal_utils import (
    get_keypoint_3d,
    calculate_angle,
    calculate_wrist_shoulder_height_signal,
    calculate_wrist_velocity_signal,
    calculate_elbow_angle_signal,
    smooth_signal,
    get_signal_statistics,
    calculate_pose_quality,
)


def create_test_pose(wrist_pos=(0.5, 0.3, 0.0), shoulder_pos=(0.5, 0.5, 0.0),
                     elbow_pos=(0.5, 0.4, 0.0)):
    """Create a test pose dictionary with specified keypoints."""
    return {
        'keypoints': {
            'RIGHT_WRIST': {'x': wrist_pos[0], 'y': wrist_pos[1], 'z': wrist_pos[2]},
            'RIGHT_SHOULDER': {'x': shoulder_pos[0], 'y': shoulder_pos[1], 'z': shoulder_pos[2]},
            'RIGHT_ELBOW': {'x': elbow_pos[0], 'y': elbow_pos[1], 'z': elbow_pos[2]},
            'LEFT_WRIST': {'x': 0.4, 'y': 0.3, 'z': 0.0},
            'LEFT_SHOULDER': {'x': 0.4, 'y': 0.5, 'z': 0.0},
            'LEFT_ELBOW': {'x': 0.4, 'y': 0.4, 'z': 0.0},
            'LEFT_HIP': {'x': 0.4, 'y': 0.7, 'z': 0.0},
            'RIGHT_HIP': {'x': 0.6, 'y': 0.7, 'z': 0.0},
            'NOSE': {'x': 0.5, 'y': 0.2, 'z': 0.0},
            'LEFT_ANKLE': {'x': 0.4, 'y': 0.95, 'z': 0.0},
            'RIGHT_ANKLE': {'x': 0.6, 'y': 0.95, 'z': 0.0},
        }
    }


class TestGetKeypoint3d:
    def test_valid_keypoint(self):
        pose = create_test_pose()
        result = get_keypoint_3d(pose, 'RIGHT_WRIST')
        assert result is not None
        assert len(result) == 3
        np.testing.assert_array_almost_equal(result, [0.5, 0.3, 0.0])

    def test_missing_keypoint(self):
        pose = create_test_pose()
        result = get_keypoint_3d(pose, 'INVALID_KEYPOINT')
        assert result is None

    def test_none_pose(self):
        result = get_keypoint_3d(None, 'RIGHT_WRIST')
        assert result is None

    def test_empty_pose(self):
        result = get_keypoint_3d({}, 'RIGHT_WRIST')
        assert result is None


class TestCalculateAngle:
    def test_right_angle(self):
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])
        angle = calculate_angle(p1, p2, p3)
        assert abs(angle - 90.0) < 0.1

    def test_straight_line(self):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([2.0, 0.0, 0.0])
        angle = calculate_angle(p1, p2, p3)
        assert abs(angle - 180.0) < 0.1

    def test_acute_angle(self):
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([1.0, 1.0, 0.0])
        angle = calculate_angle(p1, p2, p3)
        assert abs(angle - 45.0) < 0.1


class TestWristShoulderHeightSignal:
    def test_basic_calculation(self):
        poses = [
            create_test_pose(wrist_pos=(0.5, 0.3, 0.0), shoulder_pos=(0.5, 0.5, 0.0)),
            create_test_pose(wrist_pos=(0.5, 0.4, 0.0), shoulder_pos=(0.5, 0.5, 0.0)),
            create_test_pose(wrist_pos=(0.5, 0.6, 0.0), shoulder_pos=(0.5, 0.5, 0.0)),
        ]
        signal = calculate_wrist_shoulder_height_signal(poses)

        assert len(signal) == 3
        # First pose: wrist above shoulder (0.5 - 0.3 = 0.2)
        assert abs(signal[0] - 0.2) < 0.01
        # Second pose: wrist slightly above (0.5 - 0.4 = 0.1)
        assert abs(signal[1] - 0.1) < 0.01
        # Third pose: wrist below shoulder (0.5 - 0.6 = -0.1)
        assert abs(signal[2] - (-0.1)) < 0.01

    def test_with_none_poses(self):
        poses = [
            create_test_pose(),
            None,
            create_test_pose(),
        ]
        signal = calculate_wrist_shoulder_height_signal(poses)

        assert len(signal) == 3
        assert not np.isnan(signal[0])
        assert np.isnan(signal[1])
        assert not np.isnan(signal[2])

    def test_empty_poses(self):
        signal = calculate_wrist_shoulder_height_signal([])
        assert len(signal) == 0


class TestWristVelocitySignal:
    def test_basic_velocity(self):
        poses = [
            create_test_pose(wrist_pos=(0.0, 0.0, 0.0)),
            create_test_pose(wrist_pos=(0.1, 0.0, 0.0)),
            create_test_pose(wrist_pos=(0.3, 0.0, 0.0)),
        ]
        signal = calculate_wrist_velocity_signal(poses)

        assert len(signal) == 3
        assert signal[0] == 0.0  # First frame has no velocity
        assert abs(signal[1] - 0.1) < 0.01  # Moved 0.1 in x
        assert abs(signal[2] - 0.2) < 0.01  # Moved 0.2 in x

    def test_diagonal_movement(self):
        poses = [
            create_test_pose(wrist_pos=(0.0, 0.0, 0.0)),
            create_test_pose(wrist_pos=(0.3, 0.4, 0.0)),  # 3-4-5 triangle
        ]
        signal = calculate_wrist_velocity_signal(poses)

        assert abs(signal[1] - 0.5) < 0.01  # sqrt(0.3^2 + 0.4^2) = 0.5


class TestElbowAngleSignal:
    def test_basic_angle(self):
        # Create a pose with a 90-degree elbow angle
        poses = [create_test_pose(
            shoulder_pos=(0.5, 0.5, 0.0),
            elbow_pos=(0.5, 0.6, 0.0),
            wrist_pos=(0.6, 0.6, 0.0)
        )]
        signal = calculate_elbow_angle_signal(poses)

        assert len(signal) == 1
        assert abs(signal[0] - 90.0) < 1.0  # Should be close to 90 degrees


class TestSmoothSignal:
    def test_no_smoothing(self):
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = smooth_signal(signal, window_size=1)
        np.testing.assert_array_almost_equal(signal, result)

    def test_basic_smoothing(self):
        signal = np.array([1.0, 5.0, 1.0, 5.0, 1.0])
        result = smooth_signal(signal, window_size=3)

        # Middle value should be average of window (indices 1,2,3)
        expected = (5.0 + 1.0 + 5.0) / 3
        assert abs(result[2] - expected) < 0.01

    def test_handles_nan(self):
        signal = np.array([1.0, np.nan, 3.0])
        result = smooth_signal(signal, window_size=3)

        # Should ignore NaN in averaging
        assert not np.isnan(result[0])
        assert not np.isnan(result[2])


class TestGetSignalStatistics:
    def test_basic_stats(self):
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = get_signal_statistics(signal)

        assert stats['min'] == 1.0
        assert stats['max'] == 5.0
        assert stats['mean'] == 3.0
        assert stats['range'] == 4.0

    def test_with_nan(self):
        signal = np.array([1.0, np.nan, 5.0])
        stats = get_signal_statistics(signal)

        assert stats['min'] == 1.0
        assert stats['max'] == 5.0
        assert stats['mean'] == 3.0

    def test_all_nan(self):
        signal = np.array([np.nan, np.nan])
        stats = get_signal_statistics(signal)

        assert stats['min'] == 0.0
        assert stats['max'] == 0.0
        assert stats['mean'] == 0.0


class TestCalculatePoseQuality:
    def test_full_quality(self):
        poses = [create_test_pose()]
        quality = calculate_pose_quality(poses)

        assert len(quality) == 1
        assert quality[0] == 1.0  # All key landmarks present

    def test_none_pose(self):
        poses = [None]
        quality = calculate_pose_quality(poses)

        assert len(quality) == 1
        assert quality[0] == 0.0

    def test_mixed_quality(self):
        poses = [create_test_pose(), None, create_test_pose()]
        quality = calculate_pose_quality(poses)

        assert len(quality) == 3
        assert quality[0] == 1.0
        assert quality[1] == 0.0
        assert quality[2] == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
