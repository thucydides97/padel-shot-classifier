---
created: 2025-11-22T21:32:40Z
last_updated: 2025-11-22T21:32:40Z
version: 1.0
author: Claude Code PM System
---

# Project Brief

## What It Does
A machine learning system for classifying padel overhead shots into three categories: **bandeja**, **vibora**, and **smash**.

Uses pose estimation (MediaPipe) and biomechanical feature analysis to distinguish between shot types based on body movement patterns.

## Why It Exists
- No existing tools for automated padel shot analysis
- Manual technique review is time-consuming
- Players want quantitative feedback on their shots
- Coaches need objective metrics for comparison

## Problem Statement
Padel players struggle to consistently distinguish between and improve their overhead shots. Current feedback is subjective and doesn't leverage available video data.

## Solution Approach
Extract biomechanical features from video using pose estimation, then classify shots using machine learning to provide objective, quantitative feedback.

## Success Criteria
1. **Accuracy:** >50% classification accuracy (baseline ~33%)
2. **Usability:** Process videos with simple CLI commands
3. **Interpretability:** Provide feature importance and visualizations
4. **Extensibility:** Easy to add new shot types or features

## Scope

### In Scope
- Overhead shots (bandeja, vibora, smash)
- Single-player video analysis
- Offline batch processing

### Out of Scope (Current Version)
- Real-time classification
- Automatic shot detection
- Multi-player tracking
- Mobile app interface

## Key Constraints
- Requires HD video (60fps recommended)
- Manual shot annotation required
- Needs 15-20+ labeled shots for training
