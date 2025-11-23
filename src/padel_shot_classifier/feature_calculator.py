"""
Feature Engineering Module

Calculates biomechanical features from pose sequences for shot classification.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas as pd
from .shot_segmenter import Shot
from .signal_utils import (
    get_keypoint_3d,
    calculate_angle,
    calculate_wrist_shoulder_height_signal,
    calculate_wrist_velocity_signal,
    calculate_elbow_angle_signal,
    calculate_shoulder_rotation_signal,
    get_signal_statistics,
)


class FeatureCalculator:
    """Calculate biomechanical features from pose sequences."""

    def __init__(self):
        """Initialize feature calculator."""
        pass

    def calculate_shot_features(self, shot: Shot) -> Optional[Dict[str, float]]:
        """
        Calculate all features for a single shot.

        Args:
            shot: Shot object with pose sequence

        Returns:
            Dictionary of features, or None if unable to calculate
        """
        # Filter out None poses
        valid_poses = [p for p in shot.pose_sequence if p is not None]

        if len(valid_poses) < 3:  # Need at least 3 frames for velocity
            return None

        try:
            features = {
                'shot_type': shot.shot_type,
                'duration_frames': len(shot.pose_sequence),
                'valid_pose_ratio': shot.get_valid_pose_ratio(),
            }

            # Calculate biomechanical features
            features.update(self._calculate_wrist_shoulder_height(valid_poses))
            features.update(self._calculate_elbow_angle(valid_poses))
            features.update(self._calculate_shoulder_rotation(valid_poses))
            features.update(self._calculate_wrist_velocity(valid_poses))
            features.update(self._calculate_contact_height(valid_poses))
            features.update(self._calculate_racket_angle(valid_poses))

            return features

        except Exception as e:
            print(f"Warning: Failed to calculate features for shot: {e}")
            return None

    def calculate_features_for_shots(self, shots: List[Shot]) -> pd.DataFrame:
        """
        Calculate features for multiple shots.

        Args:
            shots: List of Shot objects

        Returns:
            DataFrame with features for each shot
        """
        features_list = []

        for shot in shots:
            features = self.calculate_shot_features(shot)
            if features is not None:
                features_list.append(features)

        df = pd.DataFrame(features_list)

        print(f"\n✓ Calculated features for {len(df)}/{len(shots)} shots")
        print(f"  Feature columns: {len(df.columns)}")
        print(f"  Features: {list(df.columns)}")

        return df

    def _calculate_wrist_shoulder_height(self, poses: List[Dict]) -> Dict[str, float]:
        """
        Calculate wrist height relative to shoulder.

        Returns:
            Dictionary with statistics about wrist-shoulder height
        """
        signal = calculate_wrist_shoulder_height_signal(poses)
        stats = get_signal_statistics(signal)

        return {
            'wrist_shoulder_height_max': stats['max'],
            'wrist_shoulder_height_mean': stats['mean'],
            'wrist_shoulder_height_std': stats['std'],
        }

    def _calculate_elbow_angle(self, poses: List[Dict]) -> Dict[str, float]:
        """
        Calculate elbow angle throughout swing.

        Returns:
            Dictionary with elbow angle statistics
        """
        signal = calculate_elbow_angle_signal(poses)
        stats = get_signal_statistics(signal)

        return {
            'elbow_angle_min': stats['min'],
            'elbow_angle_max': stats['max'],
            'elbow_angle_mean': stats['mean'],
            'elbow_angle_range': stats['range'],
        }

    def _calculate_shoulder_rotation(self, poses: List[Dict]) -> Dict[str, float]:
        """
        Calculate shoulder rotation angle using shoulder-hip alignment.

        Returns:
            Dictionary with shoulder rotation statistics
        """
        signal = calculate_shoulder_rotation_signal(poses)
        stats = get_signal_statistics(signal)

        return {
            'shoulder_rotation_max': stats['max'],
            'shoulder_rotation_mean': stats['mean'],
            'shoulder_rotation_range': stats['range'],
        }

    def _calculate_wrist_velocity(self, poses: List[Dict]) -> Dict[str, float]:
        """
        Calculate maximum wrist velocity.

        Returns:
            Dictionary with wrist velocity statistics
        """
        signal = calculate_wrist_velocity_signal(poses)
        stats = get_signal_statistics(signal)

        return {
            'wrist_velocity_max': stats['max'],
            'wrist_velocity_mean': stats['mean'],
        }

    def _calculate_contact_height(self, poses: List[Dict]) -> Dict[str, float]:
        """
        Calculate contact point height relative to player height.

        Returns:
            Dictionary with contact height statistics
        """
        # Estimate player height using head-to-feet distance
        player_heights = []
        wrist_heights = []

        for pose in poses:
            nose = get_keypoint_3d(pose, 'NOSE')
            left_ankle = get_keypoint_3d(pose, 'LEFT_ANKLE')
            right_ankle = get_keypoint_3d(pose, 'RIGHT_ANKLE')
            wrist = get_keypoint_3d(pose, 'RIGHT_WRIST')

            if nose is not None and left_ankle is not None and right_ankle is not None:
                ankle_y = (left_ankle[1] + right_ankle[1]) / 2
                player_height = abs(nose[1] - ankle_y)
                player_heights.append(player_height)

                if wrist is not None:
                    wrist_height = abs(wrist[1] - ankle_y)
                    wrist_heights.append(wrist_height)

        if not player_heights or not wrist_heights:
            return {'contact_height_relative': 0.0}

        avg_player_height = np.mean(player_heights)
        max_wrist_height = np.max(wrist_heights)

        return {
            'contact_height_relative': max_wrist_height / avg_player_height if avg_player_height > 0 else 0.0,
            'contact_height_absolute': max_wrist_height,
        }

    def _calculate_racket_angle(self, poses: List[Dict]) -> Dict[str, float]:
        """
        Calculate racket face angle approximation using wrist-elbow vector.

        Returns:
            Dictionary with racket angle statistics
        """
        angles = []

        for pose in poses:
            elbow = get_keypoint_3d(pose, 'RIGHT_ELBOW')
            wrist = get_keypoint_3d(pose, 'RIGHT_WRIST')

            if elbow is not None and wrist is not None:
                # Calculate angle of wrist-elbow vector relative to horizontal
                vec = wrist - elbow
                angle = np.degrees(np.arctan2(vec[1], vec[0]))  # Angle from horizontal
                angles.append(angle)

        if not angles:
            return {'racket_angle_at_contact': 0.0, 'racket_angle_mean': 0.0}

        # Assume contact point is near maximum wrist height
        wrist_heights = []
        for pose in poses:
            wrist = get_keypoint_3d(pose, 'RIGHT_WRIST')
            if wrist is not None:
                wrist_heights.append(wrist[1])

        contact_idx = np.argmin(wrist_heights) if wrist_heights else len(angles) // 2

        return {
            'racket_angle_at_contact': angles[min(contact_idx, len(angles)-1)],
            'racket_angle_mean': np.mean(angles),
            'racket_angle_std': np.std(angles),
        }


def calculate_features(shots: List[Shot], output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to calculate features for all shots.

    Args:
        shots: List of Shot objects
        output_path: Optional path to save features CSV

    Returns:
        DataFrame with features
    """
    calculator = FeatureCalculator()
    features_df = calculator.calculate_features_for_shots(shots)

    if output_path:
        features_df.to_csv(output_path, index=False)
        print(f"  Saved features to: {output_path}")

    return features_df


if __name__ == "__main__":
    import argparse
    from .pose_extractor import PoseExtractor
    from .shot_segmenter import segment_shots_from_csv

    parser = argparse.ArgumentParser(description="Calculate features from segmented shots")
    parser.add_argument("--labels", required=True, help="Path to labels CSV")
    parser.add_argument("--poses", required=True, help="Path to pose data JSON")
    parser.add_argument("--output", default="results/features.csv",
                       help="Path to output features CSV")

    args = parser.parse_args()

    # Load pose data and segment shots
    pose_data = PoseExtractor.load_pose_data(args.poses)
    shots = segment_shots_from_csv(args.labels, pose_data)

    # Calculate features
    features_df = calculate_features(shots, args.output)

    print(f"\n✓ Feature calculation complete")
    print(f"\nFeature summary:")
    print(features_df.describe())
