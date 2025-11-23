---
created: 2025-11-22T21:32:40Z
last_updated: 2025-11-22T21:32:40Z
version: 1.0
author: Claude Code PM System
---

# System Patterns

## Architectural Style
**Pipeline Architecture** - Sequential data processing stages

```
Video Input → Pose Extraction → Segmentation → Feature Calculation → Classification → Visualization
```

## Design Patterns

### Module Pattern
- Each component is a standalone module that can be run independently
- Modules have CLI interfaces for standalone usage
- Main orchestrator coordinates the full pipeline

### Strategy Pattern
- Three classifier strategies: Rule-based, Random Forest, LSTM
- Same interface, different implementations
- Easy to add new classification approaches

### Data Flow Pattern
- Input: Video file + Labels CSV
- Intermediate: JSON (poses), CSV (features)
- Output: Models (.pkl, .pth) + Visualizations (.png)

## Data Structures

### Pose Data (JSON)
- Frame-by-frame keypoint coordinates
- 33 MediaPipe landmarks per frame
- Confidence scores per landmark

### Features (CSV)
- Per-shot biomechanical features
- 7+ calculated metrics
- Normalized and statistical aggregates

### Labels (CSV)
```csv
shot_type,start_frame,end_frame
bandeja,120,180
```

## Code Organization
- Clear separation between extraction, processing, and output
- Each module handles one responsibility
- Shared utilities kept minimal

## Error Handling
- Graceful degradation for missing poses
- Warnings for low confidence detections
- Skip shots with insufficient valid poses
