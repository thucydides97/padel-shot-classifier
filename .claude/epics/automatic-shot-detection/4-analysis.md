---
issue: 4
title: Implement boundary refinement
analyzed: 2025-11-23T10:35:27Z
estimated_hours: 5
parallelization_factor: 1.0
---

# Parallel Work Analysis: Issue #4

## Overview
Refine rough overhead windows into precise shot boundaries by analyzing wrist velocity peaks to find contact points, then expanding to find true start/end frames. This is a single-file modification task with sequential algorithm steps.

## Parallel Streams

### Stream A: Implementation
**Scope**: Core boundary refinement algorithm
**Files**:
- src/padel_shot_classifier/shot_detector.py
**Agent Type**: general-purpose
**Can Start**: immediately
**Estimated Hours**: 4
**Dependencies**: none

**Work Items**:
1. Add velocity peak detection using scipy.signal.find_peaks
2. Implement find_rising_edge function
3. Implement find_falling_edge function
4. Add boundary refinement method to ShotDetector
5. Handle multiple peaks and edge cases

### Stream B: Testing
**Scope**: Unit tests for refinement functions
**Files**:
- tests/test_shot_detector.py
**Agent Type**: general-purpose
**Can Start**: after Stream A completes
**Estimated Hours**: 1
**Dependencies**: Stream A

**Work Items**:
1. Test velocity peak detection
2. Test edge finding functions
3. Test boundary refinement with various scenarios
4. Test edge cases (weak signals, multiple peaks)

## Coordination Points

### Shared Files
None - single stream modifies implementation, second stream adds tests.

### Sequential Requirements
1. Implementation must complete before tests can validate behavior
2. Algorithm design must be finalized before edge case tests

## Conflict Risk Assessment
- **Low Risk**: Single file modification with clear ownership

## Parallelization Strategy

**Recommended Approach**: sequential

This task has limited parallelization potential because:
- Only one file is modified (shot_detector.py)
- Tests depend on implementation being complete
- Algorithm steps are interdependent

Implementation should complete first, then tests added.

## Expected Timeline

With parallel execution: Not applicable for this task
- Wall time: 5 hours
- Total work: 5 hours
- Efficiency gain: 0%

This is a sequential task by nature.

## Notes
- Uses scipy.signal.find_peaks which is already installed
- Builds on existing ShotDetector class from Issue #3
- Should integrate with existing _find_overhead_windows method
- Consider making peak prominence threshold configurable
