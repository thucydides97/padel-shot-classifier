"""
Visualization Module

Creates visualizations for pose data, features, and classification results.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import mediapipe as mp

from .shot_segmenter import Shot


class PoseVisualizer:
    """Create visualizations for pose data and classification results."""

    def __init__(self, output_dir: str = "results/visualizations"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Set visualization style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

    def draw_pose_on_frame(
        self,
        frame: np.ndarray,
        pose_data: Dict,
        draw_landmarks: bool = True,
        draw_connections: bool = True
    ) -> np.ndarray:
        """
        Draw pose skeleton on a video frame.

        Args:
            frame: Video frame
            pose_data: Pose data for this frame
            draw_landmarks: Whether to draw landmark points
            draw_connections: Whether to draw connections between landmarks

        Returns:
            Frame with pose overlay
        """
        if pose_data is None:
            return frame

        frame_with_pose = frame.copy()
        height, width = frame.shape[:2]

        keypoints = pose_data['keypoints']

        # Draw connections
        if draw_connections:
            connections = self.mp_pose.POSE_CONNECTIONS

            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]

                start_name = self.mp_pose.PoseLandmark(start_idx).name
                end_name = self.mp_pose.PoseLandmark(end_idx).name

                if start_name in keypoints and end_name in keypoints:
                    start_point = (keypoints[start_name]['x_px'], keypoints[start_name]['y_px'])
                    end_point = (keypoints[end_name]['x_px'], keypoints[end_name]['y_px'])

                    # Color based on visibility
                    visibility = min(
                        keypoints[start_name]['visibility'],
                        keypoints[end_name]['visibility']
                    )
                    color = (0, int(255 * visibility), 0)

                    cv2.line(frame_with_pose, start_point, end_point, color, 2)

        # Draw landmarks
        if draw_landmarks:
            for landmark_name, landmark in keypoints.items():
                point = (landmark['x_px'], landmark['y_px'])
                visibility = landmark['visibility']

                # Color based on visibility
                color = (0, 0, int(255 * visibility))

                cv2.circle(frame_with_pose, point, 4, color, -1)

        return frame_with_pose

    def create_shot_comparison_video(
        self,
        video_path: str,
        shots: List[Shot],
        output_path: str,
        max_shots_per_type: int = 2
    ):
        """
        Create side-by-side video comparison of different shot types.

        Args:
            video_path: Path to original video
            shots: List of Shot objects
            output_path: Path to save output video
            max_shots_per_type: Maximum shots to include per type
        """
        # Select representative shots
        selected_shots = {}

        for shot_type in ['bandeja', 'vibora', 'smash']:
            type_shots = [s for s in shots if s.shot_type == shot_type]
            selected_shots[shot_type] = type_shots[:max_shots_per_type]

        # For simplicity, just save first frame of each shot type
        cap = cv2.VideoCapture(video_path)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, shot_type in enumerate(['bandeja', 'vibora', 'smash']):
            if shot_type in selected_shots and selected_shots[shot_type]:
                shot = selected_shots[shot_type][0]

                # Get middle frame of shot
                middle_frame = shot.start_frame + (shot.end_frame - shot.start_frame) // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
                ret, frame = cap.read()

                if ret:
                    # Find corresponding pose
                    pose_idx = middle_frame - shot.frames[0]
                    if 0 <= pose_idx < len(shot.pose_sequence):
                        pose_data = shot.pose_sequence[pose_idx]
                        frame = self.draw_pose_on_frame(frame, pose_data)

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    axes[idx].imshow(frame_rgb)
                    axes[idx].set_title(f"{shot_type.upper()}\nFrame {middle_frame}")
                    axes[idx].axis('off')

        cap.release()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Shot comparison saved to {output_path}")

    def plot_feature_distributions(
        self,
        features_df: pd.DataFrame,
        output_path: str,
        top_n_features: int = 6
    ):
        """
        Plot feature distributions grouped by shot type.

        Args:
            features_df: DataFrame with features
            output_path: Path to save plot
            top_n_features: Number of top features to plot
        """
        # Select numeric features (exclude shot_type)
        numeric_features = features_df.select_dtypes(include=[np.number]).columns.tolist()

        # Calculate variance for each feature to find most discriminative
        feature_variance = {}
        for feature in numeric_features:
            by_type = features_df.groupby('shot_type')[feature].mean()
            feature_variance[feature] = by_type.var()

        # Select top features by variance
        top_features = sorted(feature_variance.items(), key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in top_features[:top_n_features]]

        # Create subplots
        n_cols = 3
        n_rows = (len(top_features) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for idx, feature in enumerate(top_features):
            ax = axes[idx]

            # Create violin plot
            shot_types = features_df['shot_type'].unique()
            data_by_type = [
                features_df[features_df['shot_type'] == st][feature].values
                for st in shot_types
            ]

            positions = range(len(shot_types))
            parts = ax.violinplot(data_by_type, positions=positions, showmeans=True)

            # Color by shot type
            colors = {'bandeja': '#FF6B6B', 'vibora': '#4ECDC4', 'smash': '#45B7D1'}
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors.get(shot_types[i], 'gray'))
                pc.set_alpha(0.7)

            ax.set_xticks(positions)
            ax.set_xticklabels(shot_types)
            ax.set_ylabel(feature)
            ax.set_title(f"{feature}\n(variance: {feature_variance[feature]:.4f})")
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(top_features), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Feature distributions saved to {output_path}")

    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        labels: List[str],
        output_path: str,
        title: str = "Confusion Matrix"
    ):
        """
        Create confusion matrix heatmap.

        Args:
            confusion_matrix: Confusion matrix
            labels: Class labels
            output_path: Path to save plot
            title: Plot title
        """
        plt.figure(figsize=(8, 6))

        # Create heatmap
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Count'}
        )

        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Confusion matrix saved to {output_path}")

    def plot_feature_importance(
        self,
        random_forest_model,
        feature_names: List[str],
        output_path: str,
        top_n: int = 10
    ):
        """
        Plot feature importance from RandomForest.

        Args:
            random_forest_model: Trained RandomForest model
            feature_names: List of feature names
            output_path: Path to save plot
            top_n: Number of top features to show
        """
        # Get feature importances
        importances = random_forest_model.feature_importances_

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)

        # Create plot
        plt.figure(figsize=(10, 6))

        bars = plt.barh(
            range(len(importance_df)),
            importance_df['importance'],
            color='steelblue'
        )

        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importance', fontsize=12)
        plt.title('Top Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                f'{width:.3f}',
                ha='left',
                va='center',
                fontsize=9
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Feature importance saved to {output_path}")

    def create_summary_visualization(
        self,
        results: Dict,
        features_df: pd.DataFrame,
        output_path: str
    ):
        """
        Create a comprehensive summary visualization.

        Args:
            results: Classification results from all models
            features_df: Features DataFrame
            output_path: Path to save plot
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle('Padel Shot Classification - Results Summary',
                    fontsize=16, fontweight='bold')

        # 1. Accuracy comparison
        ax1 = fig.add_subplot(gs[0, :])
        model_names = []
        accuracies = []

        for model_name, result in results.items():
            if result is not None:
                model_names.append(model_name.replace('_', ' ').title())
                accuracies.append(result['accuracy'] * 100)

        bars = ax1.bar(model_names, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
        ax1.set_ylim([0, 100])
        ax1.axhline(y=33.33, color='red', linestyle='--', label='Random Baseline')
        ax1.legend()

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10)

        # 2-4. Confusion matrices
        cm_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
        labels = sorted(features_df['shot_type'].unique())

        for idx, (model_name, ax) in enumerate(zip(['rule_based', 'random_forest', 'lstm'], cm_axes)):
            if model_name in results and results[model_name] is not None:
                cm = results[model_name]['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                          xticklabels=labels, yticklabels=labels,
                          ax=ax, cbar=False)
                ax.set_title(model_name.replace('_', ' ').title(), fontsize=11, fontweight='bold')
                ax.set_ylabel('True' if idx == 0 else '')
                ax.set_xlabel('Predicted')

        # 5. Shot type distribution
        ax5 = fig.add_subplot(gs[2, 0])
        shot_counts = features_df['shot_type'].value_counts()
        colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        ax5.pie(shot_counts.values, labels=shot_counts.index, autopct='%1.1f%%',
               colors=colors_pie, startangle=90)
        ax5.set_title('Shot Type Distribution', fontsize=11, fontweight='bold')

        # 6. Feature correlation heatmap (top features)
        ax6 = fig.add_subplot(gs[2, 1:])
        numeric_features = features_df.select_dtypes(include=[np.number]).columns[:8]
        correlation = features_df[numeric_features].corr()
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, ax=ax6, cbar_kws={'label': 'Correlation'})
        ax6.set_title('Feature Correlation Matrix', fontsize=11, fontweight='bold')

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Summary visualization saved to {output_path}")


def create_all_visualizations(
    video_path: str,
    shots: List[Shot],
    features_df: pd.DataFrame,
    results: Dict,
    random_forest_model,
    output_dir: str = "results/visualizations"
):
    """
    Create all visualizations at once.

    Args:
        video_path: Path to video file
        shots: List of Shot objects
        features_df: Features DataFrame
        results: Classification results
        random_forest_model: Trained RandomForest model
        output_dir: Output directory
    """
    visualizer = PoseVisualizer(output_dir)

    print("\n=== Creating Visualizations ===\n")

    # 1. Shot comparison
    visualizer.create_shot_comparison_video(
        video_path,
        shots,
        f"{output_dir}/shot_comparison.png"
    )

    # 2. Feature distributions
    visualizer.plot_feature_distributions(
        features_df,
        f"{output_dir}/feature_distributions.png"
    )

    # 3. Confusion matrices
    labels = sorted(features_df['shot_type'].unique())

    for model_name, result in results.items():
        if result is not None:
            visualizer.plot_confusion_matrix(
                result['confusion_matrix'],
                labels,
                f"{output_dir}/confusion_matrix_{model_name}.png",
                title=f"Confusion Matrix - {model_name.replace('_', ' ').title()}"
            )

    # 4. Feature importance
    feature_names = [col for col in features_df.columns if col != 'shot_type']
    visualizer.plot_feature_importance(
        random_forest_model,
        feature_names,
        f"{output_dir}/feature_importance.png"
    )

    # 5. Summary visualization
    visualizer.create_summary_visualization(
        results,
        features_df,
        f"{output_dir}/summary.png"
    )

    print(f"\n✓ All visualizations saved to {output_dir}/")


if __name__ == "__main__":
    print("This module is meant to be imported, not run directly")
    print("Use main.py to run the full pipeline with visualizations")
