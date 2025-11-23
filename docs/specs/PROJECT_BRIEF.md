# Padel Shot Classification System - Development Brief

## Task: Build Padel Shot Classification Skeleton

Create a Python application that processes a video file of padel overhead shots and builds a basic classification system to distinguish between three shot types: bandeja, vibora, and smash.

## Input
- Single MP4 video file (60fps, HD resolution)
- CSV file with manual shot labels: `shot_type,start_frame,end_frame`

## Required Implementation

### 1. Pose Extraction Module (`pose_extractor.py`)
- Use MediaPipe Pose to extract body keypoints from every frame
- Save pose data as JSON/pickle with frame numbers
- Include confidence scores for each keypoint
- Handle frames where pose detection fails gracefully

### 2. Feature Engineering Module (`feature_calculator.py`)
Calculate these biomechanical features for each shot:
- Wrist height relative to shoulder at contact
- Elbow angle throughout swing
- Shoulder rotation angle (using shoulder-hip alignment)
- Maximum wrist velocity
- Contact point height (relative to player height)
- Swing duration (frames from backswing start to follow-through)
- Racket face angle approximation (using wrist-elbow vector)

### 3. Shot Segmentation Module (`shot_segmenter.py`)
- Read the CSV with manual labels
- Extract pose sequences for each labeled shot
- Add 10-frame buffer before/after each shot for context
- Output: List of shot objects with pose sequences and labels

### 4. Classification Module (`classifier.py`)
- Implement three approaches:
  1. Rule-based: Simple thresholds on key features
  2. RandomForest on feature vectors
  3. LSTM on raw pose sequences
- Use leave-one-out cross-validation (since dataset is small)
- Output accuracy metrics and confusion matrix

### 5. Visualization Module (`visualizer.py`)
- Overlay pose skeleton on video frames
- Plot feature distributions grouped by shot type
- Create side-by-side video comparison of different shots
- Generate confusion matrix heatmap
- Show feature importance from RandomForest

### 6. Main Pipeline (`main.py`)
```python
def process_video(video_path, labels_csv):
    # 1. Extract poses
    # 2. Segment shots based on labels
    # 3. Calculate features
    # 4. Train classifiers
    # 5. Generate visualizations
    # 6. Save results and model
    return classification_report
```

## Output Requirements

### 1. `results/` directory containing:
- Pose data (JSON)
- Feature matrix (CSV)
- Trained models (pickle)
- Performance metrics (TXT)
- Visualizations (PNG/MP4)

### 2. Console output showing:
- Processing progress
- Feature statistics per shot type
- Classification accuracy
- Most discriminative features

## Technical Requirements
- Python 3.8+
- Dependencies: mediapipe, opencv-python, scikit-learn, numpy, pandas, matplotlib, seaborn
- Modular design with clear separation of concerns
- Error handling for video processing failures
- Docstrings and type hints
- Progress bars for long operations (use tqdm)

## Bonus Features (If Time Allows)
- Auto-detect shot boundaries using pose velocity peaks
- Export annotated video with shot type labels
- Real-time classification mode for live video feed
- Confidence scores for each prediction

## Example Usage
```bash
python main.py --video padel_overheads.mp4 --labels shots.csv --output results/
```

## Evaluation Criteria
The system should:
1. Successfully extract poses from all frames
2. Calculate meaningful features that show variance between shot types
3. Achieve >50% classification accuracy (baseline for 3 classes is 33%)
4. Identify which features are most useful for classification
5. Provide clear visualizations showing the differences between shot types

## Sample CSV Format
```csv
shot_type,start_frame,end_frame
bandeja,120,180
vibora,245,290
smash,410,455
bandeja,580,635
```

## Notes
- This is a prototype/skeleton. Focus on getting the full pipeline working end-to-end rather than optimizing any single component.
- Document which parts would need improvement for production use.
- Include a README with setup instructions and findings from initial analysis.