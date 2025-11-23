---
created: 2025-11-22T21:32:40Z
last_updated: 2025-11-22T21:32:40Z
version: 1.0
author: Claude Code PM System
---

# Project Structure

## Directory Organization

```
padel-shot-classifier/
├── annotator.py                      # Video annotation tool (standalone)
├── src/padel_shot_classifier/        # Main package
│   ├── __init__.py                   # Package init
│   ├── pose_extractor.py             # MediaPipe pose extraction
│   ├── shot_segmenter.py             # Shot segmentation from labels
│   ├── feature_calculator.py         # Biomechanical feature calculation
│   ├── classifier.py                 # Three classification approaches
│   ├── visualizer.py                 # Visualization functions
│   └── main.py                       # Main pipeline orchestration
├── data/                             # Video files and input data
├── docs/                             # Project documentation
├── results/                          # Output directory (generated)
│   ├── pose_data.json
│   ├── features.csv
│   ├── models/
│   └── visualizations/
├── pyproject.toml                    # Poetry dependencies
├── poetry.lock                       # Locked dependencies
├── README.md                         # Documentation
├── shots.csv                         # Annotated shot labels
├── CLAUDE.md                         # Claude Code instructions
└── .claude/                          # Claude Code PM system
```

## Key Directories

### src/padel_shot_classifier/
Main application package with modular architecture:
- Each module can be run standalone or as part of pipeline
- Clear separation of concerns

### data/
Input video files for processing. Not tracked in git.

### results/
Generated outputs (pose data, features, models, visualizations).

## File Naming Patterns
- Python modules: snake_case.py
- Output files: descriptive_name.extension

## Module Dependencies
```
main.py
├── pose_extractor.py
├── shot_segmenter.py
├── feature_calculator.py
├── classifier.py
└── visualizer.py
```
