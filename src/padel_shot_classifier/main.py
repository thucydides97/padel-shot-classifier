#!/usr/bin/env python3
"""
Main Pipeline for Padel Shot Classification

End-to-end pipeline that:
1. Extracts poses from video
2. Segments shots based on labels
3. Calculates biomechanical features
4. Trains classifiers (rule-based, RandomForest, LSTM)
5. Generates visualizations
6. Saves results and models
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from .pose_extractor import PoseExtractor
from .shot_segmenter import segment_shots_from_csv
from .feature_calculator import calculate_features
from .classifier import ShotClassifier
from .visualizer import create_all_visualizations


def process_video(
    video_path: str,
    labels_csv: str,
    output_dir: str = "results",
    skip_pose_extraction: bool = False,
    skip_lstm: bool = False
) -> dict:
    """
    Run complete shot classification pipeline.

    Args:
        video_path: Path to input video file
        labels_csv: Path to CSV with shot labels
        output_dir: Directory to save all results
        skip_pose_extraction: Skip pose extraction if pose data already exists
        skip_lstm: Skip LSTM training (faster)

    Returns:
        Dictionary with classification results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PADEL SHOT CLASSIFICATION PIPELINE")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Labels: {labels_csv}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Step 1: Extract poses
    print("\n[1/6] Extracting poses from video...")
    pose_data_path = output_path / "pose_data.json"

    if skip_pose_extraction and pose_data_path.exists():
        print(f"Loading existing pose data from {pose_data_path}")
        pose_data = PoseExtractor.load_pose_data(str(pose_data_path))
    else:
        extractor = PoseExtractor()
        pose_data = extractor.extract_from_video(
            video_path,
            output_path=str(pose_data_path)
        )

    # Step 2: Segment shots based on labels
    print("\n[2/6] Segmenting shots...")
    shots = segment_shots_from_csv(
        labels_csv,
        pose_data,
        buffer_frames=10,
        min_valid_ratio=0.5
    )

    if len(shots) == 0:
        raise ValueError("No valid shots found! Check your labels CSV and pose extraction.")

    # Step 3: Calculate features
    print("\n[3/6] Calculating biomechanical features...")
    features_path = output_path / "features.csv"
    features_df = calculate_features(shots, output_path=str(features_path))

    if len(features_df) == 0:
        raise ValueError("Failed to calculate features for any shots!")

    # Print feature statistics
    print("\n=== Feature Statistics ===")
    print(features_df.groupby('shot_type').mean())

    # Step 4: Train classifiers
    print("\n[4/6] Training classifiers...")
    classifier = ShotClassifier()
    results = classifier.train_all(
        features_df,
        shots,
        use_lstm=not skip_lstm
    )

    # Step 5: Save models
    print("\n[5/6] Saving models...")
    models_dir = output_path / "models"
    classifier.save_models(str(models_dir))

    # Step 6: Generate visualizations
    print("\n[6/6] Generating visualizations...")
    viz_dir = output_path / "visualizations"
    create_all_visualizations(
        video_path,
        shots,
        features_df,
        results,
        classifier.random_forest,
        output_dir=str(viz_dir)
    )

    # Save results summary
    print("\n=== Saving Results Summary ===")
    _save_results_summary(results, features_df, output_path)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    _print_final_summary(results, shots, features_df, output_path)

    return results


def _save_results_summary(results: dict, features_df, output_path: Path):
    """Save results summary to text file."""
    summary_path = output_path / "results_summary.txt"

    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("PADEL SHOT CLASSIFICATION - RESULTS SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Dataset info
        f.write("DATASET INFORMATION\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total shots: {len(features_df)}\n")
        f.write(f"Shot type distribution:\n")
        for shot_type, count in features_df['shot_type'].value_counts().items():
            f.write(f"  {shot_type}: {count}\n")
        f.write("\n")

        # Classification results
        f.write("CLASSIFICATION RESULTS\n")
        f.write("-" * 60 + "\n")

        for model_name, result in results.items():
            if result is not None:
                f.write(f"\n{model_name.upper().replace('_', ' ')}\n")
                f.write(f"  Accuracy: {result['accuracy']:.2%}\n")
                f.write(f"  Confusion Matrix:\n")
                cm = result['confusion_matrix']
                f.write(f"    {cm}\n")

        # Most discriminative features
        f.write("\nMOST DISCRIMINATIVE FEATURES\n")
        f.write("-" * 60 + "\n")

        numeric_features = features_df.select_dtypes(include=['float64', 'int64']).columns
        feature_variance = {}

        for feature in numeric_features:
            by_type = features_df.groupby('shot_type')[feature].mean()
            feature_variance[feature] = by_type.var()

        top_features = sorted(feature_variance.items(), key=lambda x: x[1], reverse=True)[:10]

        for rank, (feature, variance) in enumerate(top_features, 1):
            f.write(f"{rank}. {feature}: {variance:.6f}\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"✓ Results summary saved to {summary_path}")


def _print_final_summary(results: dict, shots, features_df, output_path: Path):
    """Print final summary to console."""
    print("\n=== RESULTS SUMMARY ===\n")

    # Dataset info
    print(f"Dataset:")
    print(f"  Total shots: {len(shots)}")
    print(f"  Shots with features: {len(features_df)}")
    print(f"  Shot types: {', '.join(sorted(features_df['shot_type'].unique()))}")

    # Classification results
    print(f"\nClassification Accuracy:")
    for model_name, result in results.items():
        if result is not None:
            print(f"  {model_name.replace('_', ' ').title()}: {result['accuracy']:.2%}")

    # Most discriminative features
    numeric_features = features_df.select_dtypes(include=['float64', 'int64']).columns
    feature_variance = {}

    for feature in numeric_features:
        by_type = features_df.groupby('shot_type')[feature].mean()
        feature_variance[feature] = by_type.var()

    top_features = sorted(feature_variance.items(), key=lambda x: x[1], reverse=True)[:5]

    print(f"\nTop 5 Most Discriminative Features:")
    for rank, (feature, variance) in enumerate(top_features, 1):
        print(f"  {rank}. {feature}")

    # Output files
    print(f"\nOutput Files:")
    print(f"  Pose data: {output_path}/pose_data.json")
    print(f"  Features: {output_path}/features.csv")
    print(f"  Models: {output_path}/models/")
    print(f"  Visualizations: {output_path}/visualizations/")
    print(f"  Summary: {output_path}/results_summary.txt")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Padel Shot Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python -m padel_shot_classifier.main --video data/padel.mp4 --labels shots.csv

  python -m padel_shot_classifier.main --video data/padel.mp4 --labels shots.csv --output results/ --skip-lstm

  python -m padel_shot_classifier.main --video data/padel.mp4 --labels shots.csv --skip-pose-extraction
        """
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file"
    )

    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to CSV with shot labels (shot_type, start_frame, end_frame)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results (default: results/)"
    )

    parser.add_argument(
        "--skip-pose-extraction",
        action="store_true",
        help="Skip pose extraction if pose_data.json already exists"
    )

    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        help="Skip LSTM training (faster, but less accurate)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return 1

    if not Path(args.labels).exists():
        print(f"Error: Labels CSV not found: {args.labels}")
        return 1

    # Run pipeline
    try:
        process_video(
            args.video,
            args.labels,
            args.output,
            skip_pose_extraction=args.skip_pose_extraction,
            skip_lstm=args.skip_lstm
        )
        return 0

    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
