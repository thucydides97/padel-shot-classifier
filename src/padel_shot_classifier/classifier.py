"""
Classification Module

Implements three approaches for classifying padel shots:
1. Rule-based classifier using thresholds
2. RandomForest on feature vectors
3. LSTM on raw pose sequences
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import LeaveOneOut
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path

from .shot_segmenter import Shot


class RuleBasedClassifier:
    """Simple rule-based classifier using feature thresholds."""

    def __init__(self):
        """Initialize rule-based classifier."""
        self.rules = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        Learn threshold rules from training data.

        Args:
            X: Feature matrix
            y: Labels
        """
        # Calculate median values for each shot type
        self.rules = {}

        for label in np.unique(y):
            mask = y == label
            self.rules[label] = {
                'wrist_velocity_max': X.loc[mask, 'wrist_velocity_max'].median(),
                'elbow_angle_min': X.loc[mask, 'elbow_angle_min'].median(),
                'contact_height_relative': X.loc[mask, 'contact_height_relative'].median(),
            }

        print("✓ Rule-based classifier trained")
        print("  Rules learned:")
        for label, rules in self.rules.items():
            print(f"    {label}: {rules}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using rule-based approach.

        Args:
            X: Feature matrix

        Returns:
            Predicted labels
        """
        predictions = []

        for _, row in X.iterrows():
            # Calculate distance to each rule set
            distances = {}
            for label, rules in self.rules.items():
                dist = 0
                for feature, threshold in rules.items():
                    if feature in row:
                        dist += abs(row[feature] - threshold)
                distances[label] = dist

            # Predict the label with minimum distance
            predictions.append(min(distances, key=distances.get))

        return np.array(predictions)

    def save(self, path: str):
        """Save model to file."""
        with open(path, 'wb') as f:
            pickle.dump(self.rules, f)

    def load(self, path: str):
        """Load model from file."""
        with open(path, 'rb') as f:
            self.rules = pickle.load(f)


class LSTMClassifier(nn.Module):
    """LSTM classifier for pose sequences."""

    def __init__(self, input_size: int, hidden_size: int, num_classes: int, num_layers: int = 2):
        """
        Initialize LSTM classifier.

        Args:
            input_size: Number of features per timestep
            hidden_size: Number of hidden units
            num_classes: Number of output classes
            num_layers: Number of LSTM layers
        """
        super(LSTMClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """Forward pass."""
        # LSTM output
        lstm_out, _ = self.lstm(x)

        # Use output from last timestep
        last_output = lstm_out[:, -1, :]

        # Fully connected layer
        out = self.fc(last_output)

        return out


class ShotClassifier:
    """Main classifier combining all three approaches."""

    def __init__(self):
        """Initialize classifier."""
        self.rule_based = RuleBasedClassifier()
        self.random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        self.lstm_model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        self.feature_columns = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_features(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare features for classification.

        Args:
            features_df: DataFrame with features and shot_type column

        Returns:
            Tuple of (X, y) - features and labels
        """
        # Separate features from labels
        y = features_df['shot_type'].values
        X = features_df.drop(columns=['shot_type'])

        # Store feature columns
        if self.feature_columns is None:
            self.feature_columns = X.columns.tolist()

        return X, y

    def prepare_sequences(self, shots: List[Shot]) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Prepare pose sequences for LSTM.

        Args:
            shots: List of Shot objects

        Returns:
            Tuple of (sequences, labels)
        """
        sequences = []
        labels = []

        for shot in shots:
            # Extract pose keypoints into sequence
            sequence = []

            for pose in shot.pose_sequence:
                if pose is None:
                    # Use zeros for missing poses
                    sequence.append(np.zeros(33 * 3))  # 33 landmarks * 3 coords
                else:
                    # Flatten keypoints
                    keypoints = pose['keypoints']
                    coords = []
                    for landmark_name in sorted(keypoints.keys()):
                        landmark = keypoints[landmark_name]
                        coords.extend([landmark['x'], landmark['y'], landmark['z']])
                    sequence.append(np.array(coords))

            sequences.append(np.array(sequence))
            labels.append(shot.shot_type)

        return sequences, np.array(labels)

    def train_all(
        self,
        features_df: pd.DataFrame,
        shots: List[Shot],
        use_lstm: bool = True
    ) -> Dict[str, Dict]:
        """
        Train all three classifiers.

        Args:
            features_df: DataFrame with features
            shots: List of Shot objects
            use_lstm: Whether to train LSTM (can be slow)

        Returns:
            Dictionary with results for each classifier
        """
        print("\n=== Training Classifiers ===\n")

        # Prepare data
        X, y = self.prepare_features(features_df)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns
        )

        results = {}

        # 1. Rule-based classifier
        print("1. Training Rule-Based Classifier...")
        results['rule_based'] = self._evaluate_with_loo(
            self.rule_based, X, y, 'Rule-Based'
        )

        # 2. Random Forest
        print("\n2. Training Random Forest...")
        results['random_forest'] = self._evaluate_with_loo(
            self.random_forest, X_scaled, y, 'Random Forest'
        )

        # 3. LSTM
        if use_lstm:
            print("\n3. Training LSTM...")
            sequences, seq_labels = self.prepare_sequences(shots)
            results['lstm'] = self._train_lstm(sequences, seq_labels)
        else:
            print("\n3. Skipping LSTM training")
            results['lstm'] = None

        return results

    def _evaluate_with_loo(
        self,
        model,
        X: pd.DataFrame,
        y: np.ndarray,
        model_name: str
    ) -> Dict:
        """
        Evaluate model using leave-one-out cross-validation.

        Args:
            model: Classifier model
            X: Features
            y: Labels
            model_name: Name for logging

        Returns:
            Dictionary with evaluation metrics
        """
        loo = LeaveOneOut()
        predictions = []
        actuals = []

        for train_idx, test_idx in loo.split(X):
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train and predict
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            predictions.extend(pred)
            actuals.extend(y_test)

        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        accuracy = accuracy_score(actuals, predictions)
        conf_matrix = confusion_matrix(actuals, predictions)
        report = classification_report(actuals, predictions, output_dict=True)

        print(f"  {model_name} Accuracy: {accuracy:.2%}")
        print(f"  Confusion Matrix:\n{conf_matrix}")

        # Train final model on all data
        model.fit(X, y)

        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': report,
            'predictions': predictions,
            'actuals': actuals
        }

    def _train_lstm(
        self,
        sequences: List[np.ndarray],
        labels: np.ndarray,
        epochs: int = 50,
        batch_size: int = 4
    ) -> Dict:
        """
        Train LSTM classifier.

        Args:
            sequences: List of pose sequences
            labels: Shot type labels
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Dictionary with evaluation metrics
        """
        # Encode labels
        labels_encoded = self.label_encoder.transform(labels)

        # Pad sequences to same length
        max_len = max(len(seq) for seq in sequences)
        padded_sequences = []

        for seq in sequences:
            if len(seq) < max_len:
                padding = np.zeros((max_len - len(seq), seq.shape[1]))
                padded_seq = np.vstack([seq, padding])
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)

        sequences_array = np.array(padded_sequences)

        # Initialize model
        input_size = sequences_array.shape[2]
        num_classes = len(np.unique(labels_encoded))

        self.lstm_model = LSTMClassifier(
            input_size=input_size,
            hidden_size=64,
            num_classes=num_classes,
            num_layers=2
        ).to(self.device)

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)

        # Leave-one-out cross-validation
        loo = LeaveOneOut()
        predictions = []
        actuals = []

        for train_idx, test_idx in loo.split(sequences_array):
            X_train = torch.FloatTensor(sequences_array[train_idx]).to(self.device)
            y_train = torch.LongTensor(labels_encoded[train_idx]).to(self.device)
            X_test = torch.FloatTensor(sequences_array[test_idx]).to(self.device)

            # Train on fold
            self.lstm_model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = self.lstm_model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()

            # Predict
            self.lstm_model.eval()
            with torch.no_grad():
                output = self.lstm_model(X_test)
                pred = output.argmax(dim=1).cpu().numpy()
                predictions.extend(pred)
                actuals.extend(labels_encoded[test_idx])

        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Decode labels back to original
        predictions_decoded = self.label_encoder.inverse_transform(predictions)
        actuals_decoded = self.label_encoder.inverse_transform(actuals)

        accuracy = accuracy_score(actuals_decoded, predictions_decoded)
        conf_matrix = confusion_matrix(actuals_decoded, predictions_decoded)
        report = classification_report(actuals_decoded, predictions_decoded, output_dict=True)

        print(f"  LSTM Accuracy: {accuracy:.2%}")
        print(f"  Confusion Matrix:\n{conf_matrix}")

        # Train final model on all data
        X_all = torch.FloatTensor(sequences_array).to(self.device)
        y_all = torch.LongTensor(labels_encoded).to(self.device)

        self.lstm_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.lstm_model(X_all)
            loss = criterion(outputs, y_all)
            loss.backward()
            optimizer.step()

        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': report,
            'predictions': predictions_decoded,
            'actuals': actuals_decoded
        }

    def save_models(self, output_dir: str):
        """
        Save all trained models.

        Args:
            output_dir: Directory to save models
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save rule-based
        self.rule_based.save(f"{output_dir}/rule_based.pkl")

        # Save random forest
        with open(f"{output_dir}/random_forest.pkl", 'wb') as f:
            pickle.dump(self.random_forest, f)

        # Save LSTM
        if self.lstm_model is not None:
            torch.save(self.lstm_model.state_dict(), f"{output_dir}/lstm.pth")

        # Save preprocessing objects
        with open(f"{output_dir}/label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)

        with open(f"{output_dir}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)

        print(f"\n✓ Models saved to {output_dir}/")


if __name__ == "__main__":
    import argparse
    from .pose_extractor import PoseExtractor
    from .shot_segmenter import segment_shots_from_csv
    from .feature_calculator import calculate_features

    parser = argparse.ArgumentParser(description="Train shot classifiers")
    parser.add_argument("--labels", required=True, help="Path to labels CSV")
    parser.add_argument("--poses", required=True, help="Path to pose data JSON")
    parser.add_argument("--features", help="Path to features CSV (optional)")
    parser.add_argument("--output", default="results/models",
                       help="Directory to save models")
    parser.add_argument("--no-lstm", action="store_true",
                       help="Skip LSTM training")

    args = parser.parse_args()

    # Load data
    pose_data = PoseExtractor.load_pose_data(args.poses)
    shots = segment_shots_from_csv(args.labels, pose_data)

    # Calculate or load features
    if args.features:
        features_df = pd.read_csv(args.features)
    else:
        features_df = calculate_features(shots)

    # Train classifiers
    classifier = ShotClassifier()
    results = classifier.train_all(features_df, shots, use_lstm=not args.no_lstm)

    # Save models
    classifier.save_models(args.output)

    print("\n✓ Training complete!")
