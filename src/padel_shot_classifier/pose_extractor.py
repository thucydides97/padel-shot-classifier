"""
Pose Extraction Module

Extracts body keypoints from video frames using MediaPipe Pose.
Saves pose data with frame numbers and confidence scores.
"""

import cv2
import mediapipe as mp
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


class PoseExtractor:
    """Extract pose keypoints from video frames using MediaPipe."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize the pose extractor.

        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=2  # Use most accurate model
        )

    def extract_from_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None
    ) -> Dict[int, Dict]:
        """
        Extract pose data from all frames in a video.

        Args:
            video_path: Path to input video
            output_path: Path to save JSON output (optional)
            start_frame: First frame to process
            end_frame: Last frame to process (None = process all)

        Returns:
            Dictionary mapping frame numbers to pose data
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if end_frame is None:
            end_frame = total_frames

        print(f"Extracting poses from video: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}")
        print(f"Processing frames {start_frame} to {end_frame}")

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        pose_data = {}
        failed_frames = 0

        # Process frames with progress bar
        for frame_idx in tqdm(range(start_frame, end_frame), desc="Extracting poses"):
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                # Extract keypoints and confidence scores
                pose_data[frame_idx] = self._extract_keypoints(
                    results.pose_landmarks,
                    frame.shape
                )
            else:
                # No pose detected
                pose_data[frame_idx] = None
                failed_frames += 1

        cap.release()

        print(f"✓ Pose extraction complete")
        print(f"  Successful frames: {len(pose_data) - failed_frames}/{len(pose_data)}")
        print(f"  Failed detections: {failed_frames}")

        # Save to JSON if output path provided
        if output_path:
            self._save_pose_data(pose_data, output_path)
            print(f"  Saved to: {output_path}")

        return pose_data

    def _extract_keypoints(
        self,
        landmarks,
        frame_shape: Tuple[int, int, int]
    ) -> Dict:
        """
        Extract keypoint coordinates and confidence scores.

        Args:
            landmarks: MediaPipe pose landmarks
            frame_shape: Shape of the video frame (height, width, channels)

        Returns:
            Dictionary with keypoints and metadata
        """
        height, width, _ = frame_shape

        keypoints = {}
        for idx, landmark in enumerate(landmarks.landmark):
            landmark_name = self.mp_pose.PoseLandmark(idx).name

            keypoints[landmark_name] = {
                'x': landmark.x,  # Normalized [0, 1]
                'y': landmark.y,  # Normalized [0, 1]
                'z': landmark.z,  # Depth relative to hips
                'visibility': landmark.visibility,  # Confidence score
                'x_px': int(landmark.x * width),  # Pixel coordinates
                'y_px': int(landmark.y * height)
            }

        return {
            'keypoints': keypoints,
            'frame_shape': frame_shape
        }

    def _save_pose_data(self, pose_data: Dict, output_path: str):
        """
        Save pose data to JSON file.

        Args:
            pose_data: Dictionary of pose data by frame
            output_path: Path to output JSON file
        """
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        serializable_data = {}
        for frame_idx, data in pose_data.items():
            serializable_data[str(frame_idx)] = data

        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)

    @staticmethod
    def load_pose_data(json_path: str) -> Dict[int, Dict]:
        """
        Load pose data from JSON file.

        Args:
            json_path: Path to JSON file

        Returns:
            Dictionary mapping frame numbers to pose data
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Convert string keys back to integers
        return {int(k): v for k, v in data.items()}

    def get_landmark_names(self) -> List[str]:
        """Get list of all landmark names."""
        return [landmark.name for landmark in self.mp_pose.PoseLandmark]

    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'pose'):
            self.pose.close()


def extract_pose_from_video(
    video_path: str,
    output_path: str = "results/pose_data.json"
) -> Dict[int, Dict]:
    """
    Convenience function to extract poses from a video.

    Args:
        video_path: Path to video file
        output_path: Path to save pose data JSON

    Returns:
        Dictionary mapping frame numbers to pose data
    """
    extractor = PoseExtractor()
    return extractor.extract_from_video(video_path, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract pose data from video")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--output", default="results/pose_data.json",
                       help="Path to output JSON file")
    parser.add_argument("--start", type=int, default=0,
                       help="Start frame (default: 0)")
    parser.add_argument("--end", type=int, default=None,
                       help="End frame (default: all frames)")

    args = parser.parse_args()

    extract_pose_from_video(args.video, args.output)
