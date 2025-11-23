---
name: automatic-shot-detection
description: Automatically detect overhead shot boundaries in padel videos using pose-based analysis
status: backlog
created: 2025-11-22T21:35:20Z
---

# PRD: automatic-shot-detection

## Executive Summary

Implement automatic detection of overhead shot boundaries (start/end frames) in padel videos using pose-based analysis. This feature will eliminate the manual annotation process currently required, enabling fully automated video analysis while providing confidence scores for shot type classification.

**Value Proposition:** Reduce video annotation time from 10-15 minutes per video to near-zero, while maintaining ~85% accuracy in shot boundary detection.

## Problem Statement

### Current Pain Point
Users must manually annotate shot boundaries using the annotator tool, which is:
- Time-consuming (5-10 seconds per shot × many shots per video)
- Error-prone (inconsistent frame selection)
- Barrier to adoption (requires tedious manual work before any analysis)

### Why Now
- Core classification pipeline is functional
- Pose extraction already captures the data needed for detection
- Users are requesting automated workflows
- Manual annotation is the primary friction point for new users

## User Stories

### Primary Persona: Padel Player
**As a** padel player recording my practice sessions,
**I want** shots to be automatically detected in my video,
**So that** I can quickly get classification feedback without manual frame-by-frame annotation.

**Acceptance Criteria:**
- [ ] Run single command to detect all shots in video
- [ ] Output file compatible with existing classification pipeline
- [ ] Processing time < 2x video duration
- [ ] Clear indication of detection confidence

### Secondary Persona: Coach
**As a** coach analyzing multiple player videos,
**I want** batch processing of videos with automatic detection,
**So that** I can efficiently review many sessions without manual annotation.

**Acceptance Criteria:**
- [ ] Process multiple videos in sequence
- [ ] Consistent detection quality across different players
- [ ] Easy to review and correct any misdetections

### User Journey
1. User records padel practice video
2. User runs: `python -m padel_shot_classifier.shot_detector --video data/video.mp4`
3. System extracts poses and analyzes movement patterns
4. System identifies overhead shot boundaries
5. System outputs shots.csv with detected shots and confidence scores
6. User runs classification pipeline with auto-generated labels
7. User reviews any flagged low-confidence detections

## Requirements

### Functional Requirements

#### Core Detection
- **FR1:** Detect start frame of overhead shots (arm raising phase)
- **FR2:** Detect end frame of overhead shots (follow-through completion)
- **FR3:** Filter non-overhead movements (walking, serving, volleys)
- **FR4:** Provide confidence score (0-1) for each detected shot

#### Pose-Based Signals
- **FR5:** Track wrist position relative to shoulder over time
- **FR6:** Identify "overhead position" (wrist above shoulder height)
- **FR7:** Detect arm extension patterns characteristic of overhead shots
- **FR8:** Use elbow angle changes to identify swing phases

#### Output Generation
- **FR9:** Generate CSV in same format as manual annotation (shot_type,start_frame,end_frame)
- **FR10:** Include confidence score as additional column
- **FR11:** Flag low-confidence detections with warning marker
- **FR12:** Support configurable confidence threshold (default 0.7)

#### Shot Type Confidence
- **FR13:** Provide preliminary shot type prediction (bandeja/vibora/smash)
- **FR14:** Include type confidence score separate from detection confidence
- **FR15:** Mark type as "unknown" when confidence below threshold

### Non-Functional Requirements

#### Performance
- **NFR1:** Process video at minimum 2x realtime speed (30fps video = 15fps processing)
- **NFR2:** Memory usage < 4GB for 10-minute HD video
- **NFR3:** Support videos up to 30 minutes duration

#### Accuracy
- **NFR4:** Detection precision >= 85% (few false positives)
- **NFR5:** Detection recall >= 85% (few missed shots)
- **NFR6:** Boundary accuracy within ±10 frames of ground truth

#### Usability
- **NFR7:** Single command execution (no intermediate steps)
- **NFR8:** Clear progress indication during processing
- **NFR9:** Meaningful error messages for common issues

#### Compatibility
- **NFR10:** Output compatible with existing classification pipeline
- **NFR11:** Support same video formats as current system (MP4, AVI)
- **NFR12:** Work with existing pose extraction module

## Success Criteria

### Quantitative Metrics
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Detection Precision | >= 85% | Manual verification on test set |
| Detection Recall | >= 85% | Manual verification on test set |
| Boundary Accuracy | ±10 frames | Compare to manual annotations |
| Processing Speed | >= 2x realtime | Benchmark on standard videos |
| User Time Saved | >= 80% | Compare to manual annotation time |

### Qualitative Metrics
- Users can run full pipeline without manual annotation
- Low-confidence flags are meaningful (actual uncertain cases)
- Output integrates seamlessly with existing workflow

### Definition of Done
- [ ] Core detection algorithm implemented and tested
- [ ] CLI interface matches existing tool patterns
- [ ] Documentation updated with usage examples
- [ ] Accuracy metrics validated on 5+ test videos
- [ ] Integration tested with classification pipeline

## Constraints & Assumptions

### Technical Constraints
- Must work with existing MediaPipe pose extraction (33 landmarks)
- Single player in frame (no multi-player tracking)
- Requires HD video for reliable pose detection
- Cannot detect shots where player is partially occluded

### Assumptions
- Overhead shots have distinctive pose patterns differentiable from other movements
- Wrist-shoulder relationship is sufficient signal for detection
- Users accept 85% accuracy as reasonable for automation
- Existing pose extraction quality is sufficient for detection

### Timeline Constraints
- Initial implementation should be achievable in 1-2 development cycles
- Can iterate on accuracy improvements after MVP

## Out of Scope

### Explicitly Not Included (v1)
- Multi-player tracking and detection
- Real-time/streaming detection
- Detection of non-overhead shots (volleys, serves)
- Automatic video segmentation (splitting video into clips)
- GUI for reviewing/correcting detections
- Training custom detection models

### Future Considerations
- Semi-automated mode with user confirmation
- Learning from user corrections
- Detection confidence calibration
- Support for different camera angles

## Dependencies

### Internal Dependencies
- **pose_extractor.py** - Requires pose data extraction
- **shot_segmenter.py** - Output format must match segmenter input
- **feature_calculator.py** - May reuse feature calculation logic

### External Dependencies
- **MediaPipe** - Pose estimation quality directly impacts detection
- **NumPy** - Signal processing for pattern detection
- **Pandas** - CSV output generation

### Data Dependencies
- Requires test videos with manual annotations for validation
- Need diverse examples of each shot type
- Need negative examples (non-overhead movements)

## Technical Approach (High-Level)

### Detection Algorithm Concept
1. Extract poses for all frames
2. Calculate wrist-shoulder height ratio over time
3. Identify "overhead windows" where wrist > shoulder
4. Within windows, detect swing pattern (acceleration → contact → deceleration)
5. Refine boundaries using elbow angle and velocity signals
6. Score confidence based on signal clarity

### Key Signals
- Wrist height relative to shoulder (primary)
- Wrist velocity profile (secondary)
- Elbow angle changes (tertiary)
- Shoulder rotation (supporting)

### Integration Point
New module: `shot_detector.py` that:
- Uses pose_extractor for pose data
- Outputs CSV compatible with shot_segmenter
- Can be called from main.py as optional step
