---
issue: 5
title: Implement confidence scoring and type prediction
analyzed: 2025-11-23T10:47:24Z
estimated_hours: 4.5
parallelization_factor: 1.0
---

# Parallel Work Analysis: Issue #5

## Overview
Add confidence scoring for detected shots based on signal clarity, and implement preliminary shot type prediction (bandeja/vibora/smash) using feature ranges. This is a single-file modification with sequential algorithm steps.

## Parallel Streams

### Stream A: Implementation
**Scope**: Confidence scoring and type prediction algorithms
**Files**:
- src/padel_shot_classifier/shot_detector.py
**Agent Type**: general-purpose
**Can Start**: immediately
**Estimated Hours**: 3.5
**Dependencies**: none

**Work Items**:
1. Enhance existing confidence calculation with additional factors:
   - Peak velocity prominence
   - Signal-to-noise ratio
   - Window duration reasonableness
   - Pose quality during shot
2. Implement type prediction heuristics:
   - Smash: highest contact, fastest velocity
   - Vibora: medium contact, side indicators
   - Bandeja: controlled velocity, lower contact
3. Add low-confidence flagging (< 0.7 threshold)
4. Handle edge cases

### Stream B: Testing
**Scope**: Unit tests for scoring and prediction
**Files**:
- tests/test_shot_detector.py
**Agent Type**: general-purpose
**Can Start**: after Stream A completes
**Estimated Hours**: 1
**Dependencies**: Stream A

## Coordination Points

### Shared Files
None - single file modification with tests added after.

### Sequential Requirements
1. Confidence scoring logic before type prediction
2. Implementation before tests

## Conflict Risk Assessment
- **Low Risk**: Single file modification

## Parallelization Strategy

**Recommended Approach**: sequential

This task has limited parallelization potential:
- Only one file is modified
- Type prediction depends on confidence scoring
- Tests depend on implementation

## Expected Timeline

- Wall time: 4.5 hours
- Total work: 4.5 hours
- Efficiency gain: 0%

Sequential task by nature.

## Notes
- Existing confidence calculation can be enhanced, not replaced
- Type prediction is preliminary (heuristic-based, not ML)
- Should mark type as "unknown" when confidence < threshold
- Consider using existing classifier knowledge for feature ranges
