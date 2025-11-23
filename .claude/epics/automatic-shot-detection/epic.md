---
name: automatic-shot-detection
status: backlog
created: 2025-11-22T21:38:28Z
progress: 0%
prd: .claude/prds/automatic-shot-detection.md
github: https://github.com/thucydides97/padel-shot-classifier/issues/1
---

# Epic: automatic-shot-detection

## Overview

Implement a new `shot_detector.py` module that automatically identifies overhead shot boundaries in padel videos by analyzing pose data signals. The detector will use wrist-shoulder height relationships and velocity patterns to identify shot windows, then output a CSV compatible with the existing classification pipeline.

**Core Approach:** Leverage existing pose_extractor.py output and reuse signal calculations from feature_calculator.py to minimize new code while achieving 85% detection accuracy.

## Architecture Decisions

### Technology Choices
- **Signal Processing:** NumPy for time-series analysis (already a dependency)
- **Peak Detection:** scipy.signal.find_peaks for identifying shot events
- **Smoothing:** Moving average filter to reduce noise in pose signals

### Design Patterns
- **Pipeline Pattern:** Consistent with existing module architecture
- **Strategy Pattern:** Configurable thresholds for different detection sensitivity
- **Reuse over rebuild:** Extract signal calculation utilities from feature_calculator.py

### Key Technical Decisions
1. **Reuse pose_extractor output** - Don't duplicate pose extraction logic
2. **Signal-based detection** - Use time-series analysis rather than ML model
3. **Configurable thresholds** - Allow tuning without code changes
4. **Conservative defaults** - Prefer precision over recall initially

## Technical Approach

### Detection Algorithm
```
1. Load pose data (from pose_extractor or extract fresh)
2. Calculate signals:
   - wrist_height_ratio[t] = (wrist_y - shoulder_y) / body_height
   - wrist_velocity[t] = ||wrist[t] - wrist[t-1]||
   - elbow_angle[t] = angle(shoulder, elbow, wrist)
3. Smooth signals with moving average (window=5 frames)
4. Find overhead windows: wrist_height_ratio > threshold (0.1)
5. Within each window:
   - Find velocity peak (contact point)
   - Expand backward to find start (arm raising)
   - Expand forward to find end (follow-through)
6. Filter windows by duration (min 15, max 90 frames)
7. Score confidence based on signal clarity
8. Predict shot type using feature ranges (reuse classifier logic)
```

### Module Structure
```python
# shot_detector.py
class ShotDetector:
    def __init__(self, confidence_threshold=0.7):
        ...

    def detect(self, pose_data) -> List[DetectedShot]:
        signals = self._calculate_signals(pose_data)
        windows = self._find_overhead_windows(signals)
        shots = self._refine_boundaries(windows, signals)
        return self._filter_and_score(shots)

    def to_csv(self, shots, output_path):
        # Format: shot_type,start_frame,end_frame,confidence,type_confidence
        ...
```

### Integration with Existing Code
- **pose_extractor.py:** Call to get pose data if not provided
- **feature_calculator.py:** Extract signal calculation helpers
- **main.py:** Add --auto-detect flag to skip manual labels

## Implementation Strategy

### Development Phases
1. **Signal utilities** - Extract and test signal calculations
2. **Window detection** - Find overhead positions
3. **Boundary refinement** - Precise start/end detection
4. **Confidence scoring** - Quality metrics
5. **CLI and integration** - Connect to pipeline

### Risk Mitigation
- **Accuracy risk:** Start with high-confidence detections only, expand threshold
- **Performance risk:** Profile early, optimize signal calculations if needed
- **Integration risk:** Test with existing pipeline before adding features

### Testing Approach
- Unit tests for signal calculations
- Integration tests with known videos
- Accuracy validation against manual annotations (5+ videos)

## Task Breakdown Preview

High-level task categories (will be decomposed into ~8 tasks):
- [ ] **Signal utilities:** Extract wrist/elbow calculations from feature_calculator
- [ ] **Core detection:** Implement overhead window detection algorithm
- [ ] **Boundary refinement:** Precise start/end frame detection
- [ ] **Confidence scoring:** Detection and type confidence metrics
- [ ] **CSV output:** Generate pipeline-compatible output format
- [ ] **CLI interface:** Command-line arguments and progress display
- [ ] **Pipeline integration:** Add to main.py workflow
- [ ] **Validation:** Test accuracy on annotated videos

## Dependencies

### Internal Dependencies
- pose_extractor.py - Must be functional (✅ exists)
- feature_calculator.py - Reuse signal calculations (✅ exists)
- shot_segmenter.py - Output format compatibility (✅ exists)

### External Dependencies
- scipy - For find_peaks (need to add to dependencies)
- Existing: numpy, pandas, tqdm

### Prerequisites
- At least 3 videos with manual annotations for validation
- Diverse shot examples across all types

## Success Criteria (Technical)

### Performance Benchmarks
- Detection: >= 2x realtime (15 fps processing for 30fps video)
- Memory: < 4GB for 10-minute video
- Startup: < 5 seconds

### Quality Gates
- Precision >= 85% on test set
- Recall >= 85% on test set
- Boundary accuracy ±10 frames

### Acceptance Criteria
- [ ] Single command processes video end-to-end
- [ ] Output works with existing classifier without modification
- [ ] Low-confidence shots clearly marked
- [ ] Progress bar during processing
- [ ] Handles videos up to 30 minutes

## Estimated Effort

### Overall Estimate
- **Size:** Medium (M)
- **Tasks:** ~8 tasks
- **Parallelizable:** 3-4 tasks can run in parallel

### Resource Requirements
- Primary developer: 1
- Test videos with annotations: 5+

### Critical Path
1. Signal utilities (foundation)
2. Core detection (main algorithm)
3. CLI integration (user-facing)
4. Validation (quality gate)

## Simplification Opportunities

### Leverage Existing Code
- Reuse wrist/elbow angle calculations from feature_calculator.py
- Use same pose data format as other modules
- Copy CLI patterns from existing modules

### Defer to Future Versions
- Shot type prediction (can use "unknown" for v1)
- Batch processing (single video first)
- Configuration file (use CLI args only)

## Tasks Created
- [ ] #2 - Extract signal calculation utilities (parallel: true)
- [ ] #3 - Implement overhead window detection (parallel: false, depends: #2)
- [ ] #4 - Implement boundary refinement (parallel: false, depends: #2, #3)
- [ ] #5 - Implement confidence scoring and type prediction (parallel: false, depends: #4)
- [ ] #6 - Implement CSV output generation (parallel: false, depends: #5)
- [ ] #7 - Add CLI interface and progress display (parallel: false, depends: #6)
- [ ] #8 - Integrate with main pipeline (parallel: true, depends: #7)
- [ ] #9 - Validate accuracy on test videos (parallel: true, depends: #7)

Total tasks: 8
Parallel tasks: 3 (#2, #8, #9)
Sequential tasks: 5 (#3, #4, #5, #6, #7)
Estimated total effort: 26-35 hours
