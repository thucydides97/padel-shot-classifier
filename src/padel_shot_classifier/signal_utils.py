"""
Signal Calculation Utilities

Reusable functions for calculating biomechanical signals from pose data.
Used by both FeatureCalculator and ShotDetector.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def get_keypoint_3d(pose: Dict, landmark_name: str) -> Optional[np.ndarray]:
    """
    Get 3D coordinates of a keypoint from pose data.

    Args:
        pose: Pose data dictionary with 'keypoints' field
        landmark_name: Name of the landmark (e.g., 'RIGHT_WRIST')

    Returns:
        3D coordinates [x, y, z] as numpy array, or None if not available
    """
    if pose is None:
        return None

    keypoints = pose.get('keypoints', {})
    landmark = keypoints.get(landmark_name)

    if landmark is None:
        return None

    return np.array([landmark['x'], landmark['y'], landmark['z']])


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate angle between three points (p2 is the vertex).

    Args:
        p1, p2, p3: 3D coordinates of points

    Returns:
        Angle in degrees
    """
    v1 = p1 - p2
    v2 = p3 - p2

    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        return 0.0

    cos_angle = np.dot(v1, v2) / norm_product
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle = np.arccos(cos_angle)
    return np.degrees(angle)


def calculate_wrist_shoulder_height_signal(poses: List[Dict]) -> np.ndarray:
    """
    Calculate wrist height relative to shoulder for each frame.

    Positive values indicate wrist is above shoulder.

    Args:
        poses: List of pose dictionaries

    Returns:
        numpy array of height differences (NaN for invalid frames)
    """
    heights = np.full(len(poses), np.nan)

    for i, pose in enumerate(poses):
        wrist = get_keypoint_3d(pose, 'RIGHT_WRIST')
        shoulder = get_keypoint_3d(pose, 'RIGHT_SHOULDER')

        if wrist is not None and shoulder is not None:
            # Height difference (y-coordinate)
            # In MediaPipe, y increases downward, so shoulder[1] - wrist[1] is positive when wrist is above
            heights[i] = shoulder[1] - wrist[1]

    return heights


def calculate_wrist_velocity_signal(poses: List[Dict]) -> np.ndarray:
    """
    Calculate frame-to-frame wrist velocity.

    Args:
        poses: List of pose dictionaries

    Returns:
        numpy array of velocities (NaN for invalid frames, first frame is 0)
    """
    velocities = np.full(len(poses), np.nan)

    prev_wrist = None
    for i, pose in enumerate(poses):
        wrist = get_keypoint_3d(pose, 'RIGHT_WRIST')

        if wrist is not None:
            if prev_wrist is not None:
                # Calculate displacement (2D, x and y)
                displacement = np.linalg.norm(wrist[:2] - prev_wrist[:2])
                velocities[i] = displacement
            else:
                velocities[i] = 0.0
            prev_wrist = wrist.copy()
        else:
            prev_wrist = None

    return velocities


def calculate_elbow_angle_signal(poses: List[Dict]) -> np.ndarray:
    """
    Calculate elbow angle for each frame.

    Args:
        poses: List of pose dictionaries

    Returns:
        numpy array of elbow angles in degrees (NaN for invalid frames)
    """
    angles = np.full(len(poses), np.nan)

    for i, pose in enumerate(poses):
        shoulder = get_keypoint_3d(pose, 'RIGHT_SHOULDER')
        elbow = get_keypoint_3d(pose, 'RIGHT_ELBOW')
        wrist = get_keypoint_3d(pose, 'RIGHT_WRIST')

        if all(p is not None for p in [shoulder, elbow, wrist]):
            angles[i] = calculate_angle(shoulder, elbow, wrist)

    return angles


def calculate_shoulder_rotation_signal(poses: List[Dict]) -> np.ndarray:
    """
    Calculate shoulder rotation angle relative to hips for each frame.

    Args:
        poses: List of pose dictionaries

    Returns:
        numpy array of rotation angles in degrees (NaN for invalid frames)
    """
    rotations = np.full(len(poses), np.nan)

    for i, pose in enumerate(poses):
        left_shoulder = get_keypoint_3d(pose, 'LEFT_SHOULDER')
        right_shoulder = get_keypoint_3d(pose, 'RIGHT_SHOULDER')
        left_hip = get_keypoint_3d(pose, 'LEFT_HIP')
        right_hip = get_keypoint_3d(pose, 'RIGHT_HIP')

        if all(p is not None for p in [left_shoulder, right_shoulder, left_hip, right_hip]):
            # Calculate shoulder and hip line vectors
            shoulder_vec = right_shoulder - left_shoulder
            hip_vec = right_hip - left_hip

            # Calculate angle between them (2D, ignoring z)
            norm_product = np.linalg.norm(shoulder_vec[:2]) * np.linalg.norm(hip_vec[:2])
            if norm_product > 0:
                cos_angle = np.dot(shoulder_vec[:2], hip_vec[:2]) / norm_product
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                rotations[i] = np.degrees(np.arccos(cos_angle))

    return rotations


def smooth_signal(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply moving average smoothing to a signal.

    Handles NaN values by ignoring them in the average.

    Args:
        signal: Input signal array
        window_size: Size of the smoothing window

    Returns:
        Smoothed signal array
    """
    if window_size <= 1:
        return signal.copy()

    smoothed = np.full_like(signal, np.nan)
    half_window = window_size // 2

    for i in range(len(signal)):
        start = max(0, i - half_window)
        end = min(len(signal), i + half_window + 1)
        window = signal[start:end]

        # Calculate mean ignoring NaN
        valid_values = window[~np.isnan(window)]
        if len(valid_values) > 0:
            smoothed[i] = np.mean(valid_values)

    return smoothed


def get_signal_statistics(signal: np.ndarray) -> Dict[str, float]:
    """
    Calculate common statistics for a signal.

    Args:
        signal: Input signal array (may contain NaN)

    Returns:
        Dictionary with min, max, mean, std, range
    """
    valid = signal[~np.isnan(signal)]

    if len(valid) == 0:
        return {
            'min': 0.0,
            'max': 0.0,
            'mean': 0.0,
            'std': 0.0,
            'range': 0.0,
        }

    return {
        'min': float(np.min(valid)),
        'max': float(np.max(valid)),
        'mean': float(np.mean(valid)),
        'std': float(np.std(valid)),
        'range': float(np.max(valid) - np.min(valid)),
    }


def calculate_pose_quality(poses: List[Dict]) -> np.ndarray:
    """
    Calculate pose quality score for each frame.

    Quality is based on number of valid keypoints detected.

    Args:
        poses: List of pose dictionaries

    Returns:
        numpy array of quality scores (0-1)
    """
    quality = np.zeros(len(poses))

    key_landmarks = [
        'RIGHT_WRIST', 'RIGHT_ELBOW', 'RIGHT_SHOULDER',
        'LEFT_WRIST', 'LEFT_ELBOW', 'LEFT_SHOULDER',
        'RIGHT_HIP', 'LEFT_HIP', 'NOSE'
    ]

    for i, pose in enumerate(poses):
        if pose is None:
            continue

        valid_count = sum(
            1 for lm in key_landmarks
            if get_keypoint_3d(pose, lm) is not None
        )
        quality[i] = valid_count / len(key_landmarks)

    return quality
