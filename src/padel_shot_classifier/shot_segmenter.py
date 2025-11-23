"""
Shot Segmentation Module

Reads manual shot labels from CSV and extracts pose sequences for each shot.
Adds buffer frames before/after each shot for context.
"""

import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class Shot:
    """Represents a single shot with pose sequence and label."""
    shot_type: str
    start_frame: int
    end_frame: int
    pose_sequence: List[Optional[Dict]]
    frames: List[int]

    def __len__(self):
        """Return number of frames in the shot."""
        return len(self.frames)

    def has_valid_poses(self) -> bool:
        """Check if shot has at least some valid pose data."""
        valid_count = sum(1 for pose in self.pose_sequence if pose is not None)
        return valid_count > 0

    def get_valid_pose_ratio(self) -> float:
        """Get ratio of valid poses to total frames."""
        valid_count = sum(1 for pose in self.pose_sequence if pose is not None)
        return valid_count / len(self.pose_sequence) if len(self.pose_sequence) > 0 else 0.0


class ShotSegmenter:
    """Segment video into individual shots based on manual labels."""

    def __init__(self, buffer_frames: int = 10):
        """
        Initialize shot segmenter.

        Args:
            buffer_frames: Number of frames to add before/after each shot
        """
        self.buffer_frames = buffer_frames

    def load_labels(self, csv_path: str) -> pd.DataFrame:
        """
        Load shot labels from CSV.

        Args:
            csv_path: Path to CSV file with columns: shot_type, start_frame, end_frame

        Returns:
            DataFrame with shot labels
        """
        df = pd.read_csv(csv_path)

        # Validate columns
        required_cols = ['shot_type', 'start_frame', 'end_frame']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")

        # Validate shot types
        valid_types = {'bandeja', 'vibora', 'smash'}
        invalid_types = set(df['shot_type'].unique()) - valid_types
        if invalid_types:
            raise ValueError(f"Invalid shot types found: {invalid_types}. "
                           f"Must be one of: {valid_types}")

        # Validate frame numbers
        if (df['start_frame'] >= df['end_frame']).any():
            raise ValueError("start_frame must be less than end_frame for all shots")

        print(f"✓ Loaded {len(df)} shots from {csv_path}")
        print(f"  Shot type distribution:")
        for shot_type, count in df['shot_type'].value_counts().items():
            print(f"    {shot_type}: {count}")

        return df

    def segment_shots(
        self,
        labels_df: pd.DataFrame,
        pose_data: Dict[int, Dict]
    ) -> List[Shot]:
        """
        Extract pose sequences for each labeled shot.

        Args:
            labels_df: DataFrame with shot labels
            pose_data: Dictionary mapping frame numbers to pose data

        Returns:
            List of Shot objects with pose sequences
        """
        shots = []

        for idx, row in labels_df.iterrows():
            # Add buffer frames
            buffered_start = max(0, row['start_frame'] - self.buffer_frames)
            buffered_end = row['end_frame'] + self.buffer_frames

            # Extract frames in range
            frames = list(range(buffered_start, buffered_end + 1))

            # Extract pose sequence
            pose_sequence = []
            for frame_idx in frames:
                pose_sequence.append(pose_data.get(frame_idx))

            # Create Shot object
            shot = Shot(
                shot_type=row['shot_type'],
                start_frame=row['start_frame'],
                end_frame=row['end_frame'],
                pose_sequence=pose_sequence,
                frames=frames
            )

            shots.append(shot)

        # Report statistics
        self._print_statistics(shots)

        return shots

    def _print_statistics(self, shots: List[Shot]):
        """Print statistics about segmented shots."""
        print(f"\n✓ Segmented {len(shots)} shots")

        # Overall statistics
        total_frames = sum(len(shot) for shot in shots)
        avg_frames = total_frames / len(shots) if shots else 0
        print(f"  Total frames: {total_frames}")
        print(f"  Average frames per shot: {avg_frames:.1f}")

        # Pose detection statistics
        shots_with_valid_poses = sum(1 for shot in shots if shot.has_valid_poses())
        print(f"  Shots with valid poses: {shots_with_valid_poses}/{len(shots)}")

        # Per shot type statistics
        shot_types = set(shot.shot_type for shot in shots)
        print(f"\n  Per shot type statistics:")
        for shot_type in sorted(shot_types):
            type_shots = [s for s in shots if s.shot_type == shot_type]
            avg_duration = np.mean([len(s) for s in type_shots])
            avg_valid_ratio = np.mean([s.get_valid_pose_ratio() for s in type_shots])
            print(f"    {shot_type}:")
            print(f"      Count: {len(type_shots)}")
            print(f"      Avg duration: {avg_duration:.1f} frames")
            print(f"      Avg valid pose ratio: {avg_valid_ratio:.2%}")

    def filter_shots(
        self,
        shots: List[Shot],
        min_valid_ratio: float = 0.5
    ) -> List[Shot]:
        """
        Filter shots based on quality criteria.

        Args:
            shots: List of Shot objects
            min_valid_ratio: Minimum ratio of valid poses required

        Returns:
            Filtered list of shots
        """
        filtered = [
            shot for shot in shots
            if shot.get_valid_pose_ratio() >= min_valid_ratio
        ]

        removed = len(shots) - len(filtered)
        if removed > 0:
            print(f"\n⚠️  Filtered out {removed} shots with <{min_valid_ratio:.0%} valid poses")

        return filtered


def segment_shots_from_csv(
    csv_path: str,
    pose_data: Dict[int, Dict],
    buffer_frames: int = 10,
    min_valid_ratio: float = 0.5
) -> List[Shot]:
    """
    Convenience function to segment shots from CSV labels.

    Args:
        csv_path: Path to CSV with shot labels
        pose_data: Dictionary mapping frame numbers to pose data
        buffer_frames: Number of buffer frames to add
        min_valid_ratio: Minimum ratio of valid poses required

    Returns:
        List of Shot objects
    """
    segmenter = ShotSegmenter(buffer_frames=buffer_frames)
    labels_df = segmenter.load_labels(csv_path)
    shots = segmenter.segment_shots(labels_df, pose_data)
    shots = segmenter.filter_shots(shots, min_valid_ratio=min_valid_ratio)
    return shots


if __name__ == "__main__":
    import argparse
    from .pose_extractor import PoseExtractor

    parser = argparse.ArgumentParser(description="Segment shots from video based on labels")
    parser.add_argument("--labels", required=True, help="Path to labels CSV")
    parser.add_argument("--poses", required=True, help="Path to pose data JSON")
    parser.add_argument("--buffer", type=int, default=10,
                       help="Buffer frames before/after shot (default: 10)")

    args = parser.parse_args()

    # Load pose data
    pose_data = PoseExtractor.load_pose_data(args.poses)

    # Segment shots
    shots = segment_shots_from_csv(args.labels, pose_data, args.buffer)

    print(f"\n✓ Segmentation complete: {len(shots)} shots ready for analysis")
