# Padel Shot Classifier

A machine learning system for classifying padel overhead shots into three categories: **bandeja**, **vibora**, and **smash**.

This project uses pose estimation (MediaPipe) and biomechanical feature analysis to distinguish between different shot types based on body movement patterns.

## Features

- **Pose Extraction**: Extract body keypoints from video using MediaPipe Pose
- **Shot Segmentation**: Manual annotation tool + automatic shot extraction
- **Feature Engineering**: Calculate 7+ biomechanical features (wrist velocity, elbow angle, shoulder rotation, etc.)
- **Multiple Classifiers**:
  - Rule-based (threshold-based)
  - Random Forest (feature-based)
  - LSTM (sequence-based with PyTorch)
- **Visualizations**: Confusion matrices, feature distributions, pose overlays, and more

## Installation

This project uses Poetry for dependency management. Python 3.13+ is required.

```bash
# Clone the repository
git clone <repository-url>
cd padel-shot-classifier

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

## Quick Start

### Step 1: Annotate Your Video

First, use the annotation tool to label shots in your video:

```bash
python annotator.py --video data/your_video.mp4 --output shots.csv
```

**Controls:**
- `SPACE`: Play/Pause
- `S`: Mark start frame
- `E`: Mark end frame
- `1/2/3`: Label as bandeja/vibora/smash
- `ENTER`: Save annotation
- `LEFT/RIGHT`: Navigate frame-by-frame
- `A/D`: Jump 10 frames backward/forward
- `Q/ESC`: Quit

The tool will create a `shots.csv` file with format:
```csv
shot_type,start_frame,end_frame
bandeja,120,180
vibora,245,290
smash,410,455
```

### Step 2: Run the Classification Pipeline

Once you have your annotated shots, run the full pipeline:

```bash
poetry run python -m padel_shot_classifier.main \
    --video data/your_video.mp4 \
    --labels shots.csv \
    --output results/
```

This will:
1. Extract poses from all video frames
2. Segment shots based on your labels
3. Calculate biomechanical features
4. Train three classifiers (rule-based, Random Forest, LSTM)
5. Generate visualizations
6. Save all results to `results/` directory

### Step 3: View Results

After the pipeline completes, check the `results/` directory:

```
results/
├── pose_data.json              # Raw pose data
├── features.csv                # Calculated features
├── results_summary.txt         # Performance metrics
├── models/                     # Trained models
│   ├── rule_based.pkl
│   ├── random_forest.pkl
│   ├── lstm.pth
│   └── ...
└── visualizations/             # All plots
    ├── summary.png
    ├── confusion_matrix_*.png
    ├── feature_distributions.png
    ├── feature_importance.png
    └── shot_comparison.png
```

## Usage Examples

### Skip LSTM training (faster)
```bash
poetry run python -m padel_shot_classifier.main \
    --video data/video.mp4 \
    --labels shots.csv \
    --skip-lstm
```

### Reuse existing pose data (skip extraction)
```bash
poetry run python -m padel_shot_classifier.main \
    --video data/video.mp4 \
    --labels shots.csv \
    --skip-pose-extraction
```

### Run individual modules

**Extract poses only:**
```bash
poetry run python -m padel_shot_classifier.pose_extractor \
    --video data/video.mp4 \
    --output results/pose_data.json
```

**Calculate features only:**
```bash
poetry run python -m padel_shot_classifier.feature_calculator \
    --labels shots.csv \
    --poses results/pose_data.json \
    --output results/features.csv
```

**Train classifiers only:**
```bash
poetry run python -m padel_shot_classifier.classifier \
    --labels shots.csv \
    --poses results/pose_data.json \
    --output results/models/
```

## Project Structure

```
padel-shot-classifier/
├── annotator.py                      # Video annotation tool
├── src/padel_shot_classifier/
│   ├── pose_extractor.py            # MediaPipe pose extraction
│   ├── shot_segmenter.py            # Shot segmentation from labels
│   ├── feature_calculator.py        # Biomechanical feature calculation
│   ├── classifier.py                # Three classification approaches
│   ├── visualizer.py                # Visualization functions
│   └── main.py                      # Main pipeline orchestration
├── data/                            # Your video files
├── docs/specs/                      # Project specifications
├── results/                         # Output directory
└── pyproject.toml                   # Poetry dependencies
```

## Biomechanical Features

The system calculates the following features for each shot:

1. **Wrist-Shoulder Height**: Wrist height relative to shoulder (max, mean, std)
2. **Elbow Angle**: Elbow angle throughout swing (min, max, mean, range)
3. **Shoulder Rotation**: Shoulder rotation using shoulder-hip alignment (max, mean, range)
4. **Wrist Velocity**: Frame-to-frame wrist displacement (max, mean)
5. **Contact Height**: Contact point height relative to player height (relative, absolute)
6. **Racket Angle**: Racket face angle approximation using wrist-elbow vector (at contact, mean, std)
7. **Duration**: Number of frames in the shot

## Performance Expectations

- **Baseline**: ~33% accuracy (random guessing for 3 classes)
- **Target**: >50% accuracy (as per project requirements)
- **Typical Results**: 60-80% accuracy with 15-20 labeled shots

The Random Forest classifier typically performs best, followed by LSTM, then rule-based.

## Requirements

- Python 3.13+
- Video: HD 60fps MP4 (or similar)
- At least 15-20 labeled shots (5+ per shot type recommended)
- GPU optional (for faster LSTM training)

## Troubleshooting

**Issue: Pose detection fails**
- Ensure full body is visible in frame
- Check lighting conditions
- Verify video quality (HD recommended)

**Issue: Low accuracy**
- Increase number of labeled shots
- Ensure shots are properly labeled (correct types and frame ranges)
- Check that pose detection is working (>50% valid poses per shot)

**Issue: LSTM training is slow**
- Use `--skip-lstm` flag to skip LSTM training
- Reduce number of epochs in `classifier.py`
- Use a GPU if available

## Development

To add dependencies:

```bash
poetry add <package-name>
```

To add development dependencies:

```bash
poetry add --group dev <package-name>
```

## Future Improvements

Potential areas for enhancement:

- Automatic shot boundary detection (currently manual)
- Real-time classification mode
- Export annotated videos with shot labels
- More sophisticated temporal features
- Data augmentation for small datasets
- Hyperparameter tuning for classifiers

## License

[Add your license here]

## Author

Jake Bayley <jakebayley97@gmail.com>
