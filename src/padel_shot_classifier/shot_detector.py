"""
Shot Detection Module

Automatically detects overhead shot boundaries in padel videos by analyzing pose data signals.
Uses wrist-shoulder height relationships and velocity patterns to identify shot windows.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .signal_utils import (
    calculate_wrist_shoulder_height_signal,
    calculate_wrist_velocity_signal,
    smooth_signal,
    calculate_pose_quality,
)


@dataclass
class DetectedShot:
    """Represents a detected shot with frame boundaries and confidence."""
    start_frame: int
    end_frame: int
    confidence: float
    shot_type: str = "unknown"
    type_confidence: float = 0.0


class ShotDetector:
    """Detect overhead shots from pose data using signal analysis."""

    def __init__(
        self,
        height_threshold: float = 0.1,
        min_duration: int = 15,
        max_duration: int = 90,
        merge_gap: int = 10,
        smoothing_window: int = 5,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize shot detector with configurable parameters.

        Args:
            height_threshold: Minimum wrist-shoulder height ratio for overhead detection
            min_duration: Minimum shot duration in frames
            max_duration: Maximum shot duration in frames
            merge_gap: Maximum gap between windows to merge
            smoothing_window: Window size for signal smoothing
            confidence_threshold: Minimum confidence to report a detection
        """
        self.height_threshold = height_threshold
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.merge_gap = merge_gap
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold

    def detect(self, pose_data: List[Optional[Dict]]) -> List[DetectedShot]:
        """
        Detect overhead shots from pose sequence.

        Args:
            pose_data: List of pose dictionaries (one per frame)

        Returns:
            List of DetectedShot objects
        """
        if len(pose_data) < self.min_duration:
            return []

        # Calculate and smooth the wrist-shoulder height signal
        height_signal = calculate_wrist_shoulder_height_signal(pose_data)
        smoothed_signal = smooth_signal(height_signal, self.smoothing_window)

        # Find overhead windows
        windows = self._find_overhead_windows(smoothed_signal)

        # Merge nearby windows
        windows = self._merge_nearby_windows(windows)

        # Filter by duration
        windows = self._filter_by_duration(windows)

        # Calculate confidence for each window
        pose_quality = calculate_pose_quality(pose_data)
        detected_shots = []

        for start, end in windows:
            confidence = self._calculate_confidence(
                smoothed_signal, pose_quality, start, end
            )
            if confidence >= self.confidence_threshold:
                detected_shots.append(DetectedShot(
                    start_frame=start,
                    end_frame=end,
                    confidence=confidence,
                ))

        return detected_shots

    def _find_overhead_windows(self, signal: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find contiguous regions where signal exceeds threshold.

        Args:
            signal: Smoothed wrist-shoulder height signal

        Returns:
            List of (start, end) frame tuples
        """
        # Create mask for overhead positions (handle NaN)
        overhead_mask = np.zeros(len(signal), dtype=bool)
        valid_mask = ~np.isnan(signal)
        overhead_mask[valid_mask] = signal[valid_mask] > self.height_threshold

        # Find contiguous regions
        windows = []
        in_window = False
        start = 0

        for i in range(len(overhead_mask)):
            if overhead_mask[i] and not in_window:
                # Start of new window
                start = i
                in_window = True
            elif not overhead_mask[i] and in_window:
                # End of window
                windows.append((start, i))
                in_window = False

        # Handle window that extends to end
        if in_window:
            windows.append((start, len(overhead_mask)))

        return windows

    def _merge_nearby_windows(self, windows: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Merge windows that are close together.

        Args:
            windows: List of (start, end) tuples

        Returns:
            Merged list of windows
        """
        if len(windows) <= 1:
            return windows

        merged = [windows[0]]

        for start, end in windows[1:]:
            prev_start, prev_end = merged[-1]

            if start - prev_end <= self.merge_gap:
                # Merge with previous window
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))

        return merged

    def _filter_by_duration(self, windows: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Filter windows by minimum and maximum duration.

        Args:
            windows: List of (start, end) tuples

        Returns:
            Filtered list of windows
        """
        return [
            (start, end)
            for start, end in windows
            if self.min_duration <= (end - start) <= self.max_duration
        ]

    def _calculate_confidence(
        self,
        signal: np.ndarray,
        pose_quality: np.ndarray,
        start: int,
        end: int,
    ) -> float:
        """
        Calculate confidence score for a detected window.

        Based on:
        - Signal strength (how high above threshold)
        - Pose quality (how many valid keypoints)
        - Signal clarity (low variance = clearer pattern)

        Args:
            signal: Smoothed height signal
            pose_quality: Quality scores per frame
            start: Window start frame
            end: Window end frame

        Returns:
            Confidence score between 0 and 1
        """
        window_signal = signal[start:end]
        window_quality = pose_quality[start:end]

        # Remove NaN for calculations
        valid_signal = window_signal[~np.isnan(window_signal)]

        if len(valid_signal) == 0:
            return 0.0

        # Signal strength: how much above threshold
        mean_height = np.mean(valid_signal)
        strength_score = min(1.0, (mean_height - self.height_threshold) / 0.2)

        # Pose quality: average quality in window
        quality_score = np.mean(window_quality)

        # Signal clarity: inverse of coefficient of variation
        if np.mean(valid_signal) > 0:
            cv = np.std(valid_signal) / np.mean(valid_signal)
            clarity_score = max(0.0, 1.0 - cv)
        else:
            clarity_score = 0.0

        # Weighted combination
        confidence = (
            0.4 * strength_score +
            0.4 * quality_score +
            0.2 * clarity_score
        )

        return max(0.0, min(1.0, confidence))


def detect_shots(
    pose_data: List[Optional[Dict]],
    **kwargs
) -> List[DetectedShot]:
    """
    Convenience function to detect shots from pose data.

    Args:
        pose_data: List of pose dictionaries
        **kwargs: Parameters passed to ShotDetector

    Returns:
        List of DetectedShot objects
    """
    detector = ShotDetector(**kwargs)
    return detector.detect(pose_data)
