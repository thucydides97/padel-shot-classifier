"""
Shot Detection Module

Automatically detects overhead shot boundaries in padel videos by analyzing pose data signals.
Uses wrist-shoulder height relationships and velocity patterns to identify shot windows.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.signal import find_peaks

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
    low_confidence: bool = False


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
        peak_prominence: float = 0.01,
        refine_boundaries: bool = True,
        type_confidence_threshold: float = 0.5,
        low_confidence_threshold: float = 0.7,
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
            peak_prominence: Minimum prominence for velocity peak detection
            refine_boundaries: Whether to refine boundaries using velocity analysis
            type_confidence_threshold: Minimum confidence to assign a shot type
            low_confidence_threshold: Threshold below which to flag as low confidence
        """
        self.height_threshold = height_threshold
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.merge_gap = merge_gap
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold
        self.peak_prominence = peak_prominence
        self.refine_boundaries = refine_boundaries
        self.type_confidence_threshold = type_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold

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
        smoothed_height = smooth_signal(height_signal, self.smoothing_window)

        # Find overhead windows
        windows = self._find_overhead_windows(smoothed_height)

        # Merge nearby windows
        windows = self._merge_nearby_windows(windows)

        # Filter by duration
        windows = self._filter_by_duration(windows)

        # Calculate velocity signal (needed for refinement and confidence)
        velocity_signal = calculate_wrist_velocity_signal(pose_data)
        smoothed_velocity = smooth_signal(velocity_signal, self.smoothing_window)

        # Refine boundaries using velocity analysis
        if self.refine_boundaries and len(windows) > 0:
            windows = self._refine_boundaries(windows, smoothed_velocity)
            # Re-filter by duration after refinement
            windows = self._filter_by_duration(windows)

        # Calculate confidence for each window
        pose_quality = calculate_pose_quality(pose_data)
        detected_shots = []

        for start, end in windows:
            confidence = self._calculate_confidence(
                smoothed_height, smoothed_velocity, pose_quality, start, end
            )
            if confidence >= self.confidence_threshold:
                # Predict shot type
                shot_type, type_confidence = self._predict_shot_type(
                    pose_data, smoothed_velocity, smoothed_height, start, end
                )

                # Mark as unknown if type confidence is too low
                if type_confidence < self.type_confidence_threshold:
                    shot_type = "unknown"

                # Flag low confidence detections
                low_confidence = confidence < self.low_confidence_threshold

                detected_shots.append(DetectedShot(
                    start_frame=start,
                    end_frame=end,
                    confidence=confidence,
                    shot_type=shot_type,
                    type_confidence=type_confidence,
                    low_confidence=low_confidence,
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
        height_signal: np.ndarray,
        velocity_signal: np.ndarray,
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
        - Peak velocity prominence (clear peak = high confidence)
        - Signal-to-noise ratio in velocity
        - Window duration reasonableness

        Args:
            height_signal: Smoothed height signal
            velocity_signal: Smoothed velocity signal
            pose_quality: Quality scores per frame
            start: Window start frame
            end: Window end frame

        Returns:
            Confidence score between 0 and 1
        """
        window_height = height_signal[start:end]
        window_velocity = velocity_signal[start:end]
        window_quality = pose_quality[start:end]

        # Remove NaN for calculations
        valid_height = window_height[~np.isnan(window_height)]
        valid_velocity = window_velocity[~np.isnan(window_velocity)]

        if len(valid_height) == 0:
            return 0.0

        # Signal strength: how much above threshold
        mean_height = np.mean(valid_height)
        strength_score = min(1.0, (mean_height - self.height_threshold) / 0.2)

        # Pose quality: average quality in window
        quality_score = np.mean(window_quality)

        # Signal clarity: inverse of coefficient of variation
        if np.mean(valid_height) > 0:
            cv = np.std(valid_height) / np.mean(valid_height)
            clarity_score = max(0.0, 1.0 - cv)
        else:
            clarity_score = 0.0

        # Peak velocity prominence score
        prominence_score = self._calculate_peak_prominence_score(valid_velocity)

        # Signal-to-noise ratio score
        snr_score = self._calculate_snr_score(valid_velocity)

        # Duration reasonableness score
        duration = end - start
        duration_score = self._calculate_duration_score(duration)

        # Weighted combination (enhanced with new factors)
        confidence = (
            0.25 * strength_score +
            0.25 * quality_score +
            0.15 * clarity_score +
            0.15 * prominence_score +
            0.10 * snr_score +
            0.10 * duration_score
        )

        return max(0.0, min(1.0, confidence))

    def _calculate_peak_prominence_score(self, velocity: np.ndarray) -> float:
        """
        Calculate score based on peak velocity prominence.

        A clear, prominent peak indicates a confident detection.

        Args:
            velocity: Valid velocity values in window

        Returns:
            Score between 0 and 1
        """
        if len(velocity) < 3:
            return 0.0

        # Find peaks
        peaks, properties = find_peaks(velocity, prominence=0.001)

        if len(peaks) == 0:
            return 0.3  # No peaks, low confidence

        # Get the highest prominence
        prominences = properties['prominences']
        max_prominence = np.max(prominences)

        # Normalize by signal range
        signal_range = np.max(velocity) - np.min(velocity)
        if signal_range == 0:
            return 0.3

        # Score based on prominence relative to signal range
        relative_prominence = max_prominence / signal_range
        return min(1.0, relative_prominence * 1.5)

    def _calculate_snr_score(self, velocity: np.ndarray) -> float:
        """
        Calculate signal-to-noise ratio score.

        Higher SNR indicates clearer signal.

        Args:
            velocity: Valid velocity values in window

        Returns:
            Score between 0 and 1
        """
        if len(velocity) < 3:
            return 0.0

        # Signal: peak value
        signal_power = np.max(velocity)

        # Noise: standard deviation of lower half of values
        sorted_vals = np.sort(velocity)
        lower_half = sorted_vals[:len(sorted_vals) // 2]
        noise_power = np.std(lower_half) if len(lower_half) > 1 else 0.01

        if noise_power == 0:
            return 1.0

        # SNR in linear scale
        snr = signal_power / noise_power

        # Convert to score (SNR of 5 = good, 10+ = excellent)
        return min(1.0, snr / 10.0)

    def _calculate_duration_score(self, duration: int) -> float:
        """
        Calculate score based on duration reasonableness.

        Shots with typical durations get higher scores.

        Args:
            duration: Shot duration in frames

        Returns:
            Score between 0 and 1
        """
        # Optimal duration range (roughly 30-60 frames at 30fps = 1-2 seconds)
        optimal_min = 30
        optimal_max = 60

        if optimal_min <= duration <= optimal_max:
            return 1.0

        # Score decreases as we move away from optimal range
        if duration < optimal_min:
            # Too short
            return max(0.3, duration / optimal_min)
        else:
            # Too long
            ratio = optimal_max / duration
            return max(0.3, ratio)

    def _predict_shot_type(
        self,
        pose_data: List[Optional[Dict]],
        velocity_signal: np.ndarray,
        height_signal: np.ndarray,
        start: int,
        end: int,
    ) -> Tuple[str, float]:
        """
        Predict shot type using biomechanical heuristics.

        Uses contact height and velocity patterns to classify:
        - Smash: highest contact point, fastest velocity
        - Vibora: medium contact, moderate velocity
        - Bandeja: lower contact, controlled velocity

        Args:
            pose_data: List of pose dictionaries
            velocity_signal: Smoothed velocity signal
            height_signal: Smoothed height signal
            start: Window start frame
            end: Window end frame

        Returns:
            Tuple of (shot_type, type_confidence)
        """
        # Extract features from window
        window_velocity = velocity_signal[start:end]
        window_height = height_signal[start:end]

        # Remove NaN values
        valid_velocity = window_velocity[~np.isnan(window_velocity)]
        valid_height = window_height[~np.isnan(window_height)]

        if len(valid_velocity) == 0 or len(valid_height) == 0:
            return ("unknown", 0.0)

        # Calculate key features
        max_velocity = float(np.max(valid_velocity))
        max_height = float(np.max(valid_height))

        # Normalize height to relative scale (0-1)
        # Height signal is shoulder_y - wrist_y, positive when wrist above shoulder
        # Typical range: 0.1 to 0.4 for overhead shots
        relative_height = min(1.0, max_height / 0.4)

        # Normalize velocity (typical range: 0.01 to 0.15 for shots)
        relative_velocity = min(1.0, max_velocity / 0.15)

        # Predict shot type using heuristics
        shot_type, type_confidence = self._classify_by_features(
            relative_height, relative_velocity
        )

        return (shot_type, type_confidence)

    def _classify_by_features(
        self,
        relative_height: float,
        relative_velocity: float,
    ) -> Tuple[str, float]:
        """
        Classify shot type based on relative height and velocity.

        Heuristic thresholds:
        - Smash: height > 0.8, velocity > 0.7
        - Vibora: height 0.5-0.8, velocity 0.4-0.7
        - Bandeja: height < 0.5, velocity < 0.5

        Args:
            relative_height: Normalized contact height (0-1)
            relative_velocity: Normalized max velocity (0-1)

        Returns:
            Tuple of (shot_type, type_confidence)
        """
        # Calculate scores for each type
        smash_score = self._smash_score(relative_height, relative_velocity)
        vibora_score = self._vibora_score(relative_height, relative_velocity)
        bandeja_score = self._bandeja_score(relative_height, relative_velocity)

        # Find best match
        scores = {
            "smash": smash_score,
            "vibora": vibora_score,
            "bandeja": bandeja_score,
        }

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        # Calculate confidence based on score margin
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            margin = sorted_scores[0] - sorted_scores[1]
            # Higher margin = higher confidence
            type_confidence = min(1.0, best_score * (1 + margin))
        else:
            type_confidence = best_score

        return (best_type, type_confidence)

    def _smash_score(self, height: float, velocity: float) -> float:
        """
        Calculate score for smash classification.

        Smash: highest contact point (> 0.8), fastest velocity (> 0.7)
        """
        height_score = max(0, (height - 0.6) / 0.4)  # 0 at 0.6, 1 at 1.0
        velocity_score = max(0, (velocity - 0.5) / 0.5)  # 0 at 0.5, 1 at 1.0

        return (height_score * 0.5 + velocity_score * 0.5)

    def _vibora_score(self, height: float, velocity: float) -> float:
        """
        Calculate score for vibora classification.

        Vibora: medium contact (0.5-0.8), moderate velocity (0.4-0.7)
        """
        # Optimal height: 0.5-0.8, peak at 0.65
        if height < 0.3:
            height_score = 0
        elif height < 0.5:
            height_score = (height - 0.3) / 0.2
        elif height <= 0.8:
            height_score = 1.0 - abs(height - 0.65) / 0.35
        else:
            height_score = max(0, 1.0 - (height - 0.8) / 0.2)

        # Optimal velocity: 0.4-0.7, peak at 0.55
        if velocity < 0.2:
            velocity_score = 0
        elif velocity < 0.4:
            velocity_score = (velocity - 0.2) / 0.2
        elif velocity <= 0.7:
            velocity_score = 1.0 - abs(velocity - 0.55) / 0.35
        else:
            velocity_score = max(0, 1.0 - (velocity - 0.7) / 0.3)

        return (height_score * 0.5 + velocity_score * 0.5)

    def _bandeja_score(self, height: float, velocity: float) -> float:
        """
        Calculate score for bandeja classification.

        Bandeja: lower contact (< 0.5), controlled/lower velocity (< 0.5)
        """
        # Lower height is better, peak at 0.3
        if height <= 0.5:
            height_score = 1.0 - height / 0.5
        else:
            height_score = max(0, 0.5 - (height - 0.5))

        # Lower velocity is better, peak at 0.3
        if velocity <= 0.5:
            velocity_score = 1.0 - velocity / 0.5
        else:
            velocity_score = max(0, 0.5 - (velocity - 0.5))

        return (height_score * 0.5 + velocity_score * 0.5)

    def _refine_boundaries(
        self,
        windows: List[Tuple[int, int]],
        velocity_signal: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """
        Refine rough overhead windows into precise shot boundaries using velocity analysis.

        Args:
            windows: List of (start, end) tuples from overhead detection
            velocity_signal: Smoothed wrist velocity signal

        Returns:
            List of refined (start, end) tuples
        """
        if len(windows) == 0:
            return []

        # Calculate baseline velocity from non-shot periods
        baseline = self._calculate_baseline_velocity(windows, velocity_signal)

        refined_windows = []
        for start, end in windows:
            refined = self._refine_single_window(
                start, end, velocity_signal, baseline
            )
            if refined is not None:
                refined_windows.append(refined)

        return refined_windows

    def _calculate_baseline_velocity(
        self,
        windows: List[Tuple[int, int]],
        velocity_signal: np.ndarray,
    ) -> float:
        """
        Calculate baseline velocity from non-shot periods.

        Args:
            windows: List of shot windows to exclude
            velocity_signal: Wrist velocity signal

        Returns:
            Mean velocity during non-shot periods
        """
        # Create mask for non-shot frames
        non_shot_mask = np.ones(len(velocity_signal), dtype=bool)
        for start, end in windows:
            non_shot_mask[start:end] = False

        # Get valid (non-NaN) non-shot velocities
        non_shot_velocities = velocity_signal[non_shot_mask]
        valid_velocities = non_shot_velocities[~np.isnan(non_shot_velocities)]

        if len(valid_velocities) == 0:
            # Fallback: use overall signal statistics
            valid_signal = velocity_signal[~np.isnan(velocity_signal)]
            if len(valid_signal) == 0:
                return 0.0
            return float(np.percentile(valid_signal, 25))

        return float(np.mean(valid_velocities))

    def _refine_single_window(
        self,
        start: int,
        end: int,
        velocity_signal: np.ndarray,
        baseline: float,
    ) -> Optional[Tuple[int, int]]:
        """
        Refine a single window using velocity peak detection.

        Args:
            start: Original window start frame
            end: Original window end frame
            velocity_signal: Wrist velocity signal
            baseline: Baseline velocity for edge detection

        Returns:
            Refined (start, end) tuple, or None if refinement fails
        """
        # Extract velocity segment for this window
        velocity_segment = velocity_signal[start:end]

        # Handle NaN values
        valid_mask = ~np.isnan(velocity_segment)
        if not np.any(valid_mask):
            return (start, end)  # No valid data, return original

        # Check if there's any meaningful velocity variation
        valid_velocities = velocity_segment[valid_mask]
        if len(valid_velocities) == 0 or np.max(valid_velocities) == 0:
            return (start, end)  # No velocity data, return original

        # Find velocity peaks within the window
        contact_idx = self._find_contact_point(velocity_segment)
        if contact_idx is None:
            return (start, end)  # No clear peak, return original

        # Convert to absolute frame index
        contact_frame = start + contact_idx

        # Find rising and falling edges
        refined_start = self._find_rising_edge(
            velocity_signal, contact_frame, baseline, start
        )
        refined_end = self._find_falling_edge(
            velocity_signal, contact_frame, baseline, end, len(velocity_signal)
        )

        # Validate refined boundaries
        if refined_end <= refined_start:
            return (start, end)

        # Ensure minimum duration is maintained
        if refined_end - refined_start < self.min_duration:
            return (start, end)

        return (refined_start, refined_end)

    def _find_contact_point(self, velocity_segment: np.ndarray) -> Optional[int]:
        """
        Find the contact point (highest velocity peak) within a velocity segment.

        Args:
            velocity_segment: Velocity values for the window

        Returns:
            Index of the contact point within the segment, or None if no peak found
        """
        # Replace NaN with 0 for peak detection
        segment_clean = np.nan_to_num(velocity_segment, nan=0.0)

        if len(segment_clean) < 3:
            return None

        # Find peaks with prominence threshold
        peaks, properties = find_peaks(
            segment_clean,
            prominence=self.peak_prominence,
        )

        if len(peaks) == 0:
            # No peaks found with prominence threshold
            # Fall back to finding maximum
            if np.max(segment_clean) > 0:
                return int(np.argmax(segment_clean))
            return None

        # Find the highest peak (contact point)
        peak_values = segment_clean[peaks]
        highest_peak_idx = np.argmax(peak_values)
        return int(peaks[highest_peak_idx])

    def _find_rising_edge(
        self,
        velocity_signal: np.ndarray,
        contact_frame: int,
        baseline: float,
        min_frame: int,
    ) -> int:
        """
        Find where velocity begins rising above baseline before contact.

        Args:
            velocity_signal: Full velocity signal
            contact_frame: Frame of the contact point (peak velocity)
            baseline: Baseline velocity threshold
            min_frame: Minimum frame to search (original window start)

        Returns:
            Frame where velocity starts rising
        """
        # Search backward from contact point
        # Allow some padding before the original window start
        search_start = max(0, min_frame - 10)

        for i in range(contact_frame - 1, search_start - 1, -1):
            if np.isnan(velocity_signal[i]):
                continue

            # Found frame where velocity is at or below baseline
            if velocity_signal[i] <= baseline:
                # Return the next frame (where it starts rising)
                return min(i + 1, contact_frame)

        # Reached search limit, return the search start
        return search_start

    def _find_falling_edge(
        self,
        velocity_signal: np.ndarray,
        contact_frame: int,
        baseline: float,
        max_frame: int,
        signal_length: int,
    ) -> int:
        """
        Find where velocity drops back to baseline after contact.

        Args:
            velocity_signal: Full velocity signal
            contact_frame: Frame of the contact point (peak velocity)
            baseline: Baseline velocity threshold
            max_frame: Maximum frame to search (original window end)
            signal_length: Total length of the signal

        Returns:
            Frame where velocity returns to baseline
        """
        # Search forward from contact point
        # Allow some padding after the original window end
        search_end = min(signal_length, max_frame + 10)

        for i in range(contact_frame + 1, search_end):
            if np.isnan(velocity_signal[i]):
                continue

            # Found frame where velocity is at or below baseline
            if velocity_signal[i] <= baseline:
                return i

        # Reached search limit, return the search end
        return search_end


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
