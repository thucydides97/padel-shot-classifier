---
created: 2025-11-22T21:32:40Z
last_updated: 2025-11-22T21:32:40Z
version: 1.0
author: Claude Code PM System
---

# Technical Context

## Language & Runtime
- **Python:** ^3.11
- **Package Manager:** Poetry (poetry-core >= 2.0.0)

## Core Dependencies

### Computer Vision
- **mediapipe:** ^0.10.9 - Pose estimation (33 body landmarks)
- **opencv-python:** ^4.10.0 - Video processing and frame extraction

### Machine Learning
- **scikit-learn:** ^1.5.0 - Random Forest classifier
- **torch:** 2.1.2 - LSTM neural network
- **jaxlib:** 0.4.25 - Required by MediaPipe

### Data Processing
- **numpy:** ^1.24.0 - Numerical operations
- **pandas:** ^2.2.0 - Data manipulation and CSV handling

### Visualization
- **matplotlib:** ^3.9.0 - Plotting and charts
- **seaborn:** ^0.13.0 - Statistical visualizations

### Utilities
- **tqdm:** ^4.66.0 - Progress bars

## Development Tools
- Poetry for dependency management
- Git for version control
- GitHub for repository hosting

## Key Technical Decisions
1. MediaPipe over OpenPose - better performance, easier setup
2. PyTorch for LSTM - flexible, good for small projects
3. Poetry over pip - better dependency resolution

## Hardware Requirements
- HD video input (60fps recommended)
- GPU optional (for faster LSTM training)
- Standard CPU sufficient for inference

## External Services
- GitHub (repository hosting)
- No cloud APIs required
