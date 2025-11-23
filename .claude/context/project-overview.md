---
created: 2025-11-22T21:32:40Z
last_updated: 2025-11-22T21:32:40Z
version: 1.0
author: Claude Code PM System
---

# Project Overview

## High-Level Summary
Padel Shot Classifier is a Python-based ML system that analyzes video footage of padel players to automatically classify overhead shots (bandeja, vibora, smash) using pose estimation and biomechanical features.

## Feature List

### Core Features
- **Pose Extraction:** Extract 33 body keypoints per frame using MediaPipe
- **Shot Segmentation:** Isolate individual shots based on manual labels
- **Feature Calculation:** Compute 7+ biomechanical features per shot
- **Classification:** Three approaches (rule-based, Random Forest, LSTM)
- **Visualization:** Confusion matrices, feature distributions, shot comparisons

### Supporting Features
- **Video Annotation Tool:** GUI for labeling shot boundaries and types
- **Modular CLI:** Run individual components or full pipeline
- **Model Persistence:** Save and load trained models

## Current State
- **Development Phase:** MVP complete
- **Accuracy:** 60-80% typical with 15-20 labeled shots
- **Stability:** Functional but needs more testing

## Biomechanical Features
1. Wrist-Shoulder Height (max, mean, std)
2. Elbow Angle (min, max, mean, range)
3. Shoulder Rotation (max, mean, range)
4. Wrist Velocity (max, mean)
5. Contact Height (relative, absolute)
6. Racket Angle (at contact, mean, std)
7. Duration (frame count)

## Integration Points
- **Input:** MP4 video files, CSV labels
- **Output:** JSON poses, CSV features, PKL/PTH models, PNG visualizations
- **Dependencies:** MediaPipe for pose estimation, PyTorch for LSTM

## Performance Expectations
- Baseline: ~33% (random for 3 classes)
- Target: >50%
- Typical: 60-80% with sufficient data
