#!/usr/bin/env python3
"""
Video Annotation Tool for Padel Shot Classification

Interactive tool to manually label padel overhead shots in a video.
Saves annotations to shots.csv with format: shot_type,start_frame,end_frame
"""

import cv2
import pandas as pd
import os
import sys
from pathlib import Path


class VideoAnnotator:
    """Interactive video annotator for labeling padel shots."""

    SHOT_TYPES = {
        ord('1'): 'bandeja',
        ord('2'): 'vibora',
        ord('3'): 'smash'
    }

    def __init__(self, video_path: str, csv_path: str = "shots.csv"):
        """
        Initialize the video annotator.

        Args:
            video_path: Path to the video file
            csv_path: Path to save annotations CSV
        """
        self.video_path = video_path
        self.csv_path = csv_path

        # Open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame = 0

        # Annotation state
        self.start_frame = None
        self.end_frame = None
        self.current_shot_type = None
        self.playing = False

        # Load existing annotations
        self.annotations = self._load_annotations()

        print(f"Video loaded: {video_path}")
        print(f"Total frames: {self.total_frames}, FPS: {self.fps}")
        print(f"Loaded {len(self.annotations)} existing annotations")
        print("\n=== CONTROLS ===")
        print("SPACE:       Play/Pause")
        print("S:           Mark start frame")
        print("E:           Mark end frame")
        print("1/2/3:       Label as bandeja/vibora/smash")
        print("ENTER:       Save annotation")
        print("LEFT/RIGHT:  Navigate frame-by-frame")
        print("SHIFT+LEFT/RIGHT: Jump 10 frames")
        print("Q/ESC:       Quit")
        print("================\n")

    def _load_annotations(self) -> pd.DataFrame:
        """Load existing annotations from CSV if it exists."""
        if os.path.exists(self.csv_path):
            return pd.read_csv(self.csv_path)
        else:
            return pd.DataFrame(columns=['shot_type', 'start_frame', 'end_frame'])

    def _save_annotation(self):
        """Save current annotation to CSV."""
        if self.start_frame is None or self.end_frame is None or self.current_shot_type is None:
            print("⚠️  Cannot save: Start frame, end frame, and shot type must all be set")
            return

        if self.start_frame >= self.end_frame:
            print("⚠️  Cannot save: Start frame must be before end frame")
            return

        # Add new annotation
        new_annotation = pd.DataFrame([{
            'shot_type': self.current_shot_type,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame
        }])

        self.annotations = pd.concat([self.annotations, new_annotation], ignore_index=True)

        # Save to CSV
        self.annotations.to_csv(self.csv_path, index=False)

        print(f"✓ Saved: {self.current_shot_type} from frame {self.start_frame} to {self.end_frame}")
        print(f"  Total annotations: {len(self.annotations)}")

        # Reset state
        self.start_frame = None
        self.end_frame = None
        self.current_shot_type = None

    def _draw_info(self, frame):
        """Draw annotation info on the frame."""
        height, width = frame.shape[:2]

        # Create info overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Current frame info
        text = f"Frame: {self.current_frame}/{self.total_frames}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Current annotation state
        state_text = []
        if self.start_frame is not None:
            state_text.append(f"Start: {self.start_frame}")
        if self.end_frame is not None:
            state_text.append(f"End: {self.end_frame}")
        if self.current_shot_type is not None:
            state_text.append(f"Type: {self.current_shot_type}")

        if state_text:
            cv2.putText(frame, " | ".join(state_text), (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Total annotations
        cv2.putText(frame, f"Annotations: {len(self.annotations)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Playing status
        if self.playing:
            cv2.putText(frame, "PLAYING", (width - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "PAUSED", (width - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def _seek_frame(self, frame_number: int):
        """Seek to a specific frame."""
        frame_number = max(0, min(frame_number, self.total_frames - 1))
        self.current_frame = frame_number
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def run(self):
        """Run the annotation tool."""
        window_name = "Padel Shot Annotator"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            ret, frame = self.cap.read()

            if not ret:
                # Loop back to beginning if at end
                self._seek_frame(0)
                self.playing = False
                continue

            # Draw info overlay
            frame = self._draw_info(frame)

            # Display frame
            cv2.imshow(window_name, frame)

            # Handle keyboard input
            wait_time = 1 if self.playing else 0
            key = cv2.waitKey(wait_time) & 0xFF

            # Quit
            if key == ord('q') or key == 27:  # ESC
                break

            # Play/Pause
            elif key == ord(' '):
                self.playing = not self.playing
                print(f"{'Playing' if self.playing else 'Paused'} at frame {self.current_frame}")

            # Mark start frame
            elif key == ord('s'):
                self.start_frame = self.current_frame
                print(f"Start frame set: {self.start_frame}")

            # Mark end frame
            elif key == ord('e'):
                self.end_frame = self.current_frame
                print(f"End frame set: {self.end_frame}")

            # Set shot type
            elif key in self.SHOT_TYPES:
                self.current_shot_type = self.SHOT_TYPES[key]
                print(f"Shot type set: {self.current_shot_type}")

            # Save annotation
            elif key == 13:  # ENTER
                self._save_annotation()

            # Frame navigation
            elif key == 81:  # LEFT arrow
                self.playing = False
                self._seek_frame(self.current_frame - 1)

            elif key == 83:  # RIGHT arrow
                self.playing = False
                self._seek_frame(self.current_frame + 1)

            # Jump 10 frames (SHIFT + arrows would be complex, using A/D instead)
            elif key == ord('a'):  # Jump backward
                self.playing = False
                self._seek_frame(self.current_frame - 10)

            elif key == ord('d'):  # Jump forward
                self.playing = False
                self._seek_frame(self.current_frame + 10)

            # Update current frame counter
            if self.playing:
                self.current_frame += 1
            else:
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

        print(f"\n✓ Annotation session complete")
        print(f"  Total annotations saved: {len(self.annotations)}")
        print(f"  CSV file: {self.csv_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Annotate padel overhead shots in a video")
    parser.add_argument("--video", type=str, default="data/padel_overheads.mp4",
                       help="Path to video file (default: data/padel_overheads.mp4)")
    parser.add_argument("--output", type=str, default="shots.csv",
                       help="Path to output CSV file (default: shots.csv)")

    args = parser.parse_args()

    # Check if video exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        print("\nPlease specify the correct video path with --video")
        sys.exit(1)

    # Run annotator
    annotator = VideoAnnotator(args.video, args.output)
    annotator.run()


if __name__ == "__main__":
    main()
