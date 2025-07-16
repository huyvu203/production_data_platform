"""
Train a RandomForestClassifier for repeat purchase prediction.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
from typing import Tuple


def load_training_data() -> pd.DataFrame:
    """Load the training features data."""
    features_path = Path(__file__).parent.parent / "features" / "training_features.csv"
    
    if not features_path.exists():
        raise FileNotFoundError(
            f"Training features not found at {features_path}. "
            "Please run features/compute_features.py first."
        )
    
    df = pd.read_csv(features_path)
    return df


def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Prepare features and target for training.
    
    Returns:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature column names
    """
    # Identify feature columns (exclude user_id, product_id, will_repurchase)
    feature_cols = [col for col in df.columns 
                   if col not in ['user_id', 'product_id', 'product_id_x', 'product_id_y', 'will_repurchase']]
    
    # Handle any remaining categorical columns by encoding them
    df_features = df[feature_cols].copy()
    
    # Convert any object/string columns to numeric
    for col in df_features.columns:
        if df_features[col].dtype == 'object':
            print(f"Converting categorical column '{col}' to numeric...")
            # Use label encoding for categorical features
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df_features[col] = le.fit_transform(df_features[col].astype(str))
    
    X = df_features.values
    y = df['will_repurchase'].values
    
    # Handle any remaining NaN values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y, df_features.columns.tolist()


def train_model(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier.
    """
    # Configure the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',  # Handle class imbalance
        n_jobs=-1
    )
    
    print("Training Random Forest model...")
    model.fit(X, y)
    
    return model


def evaluate_model(model: RandomForestClassifier, X: np.ndarray, y: np.ndarray, 
                  feature_names: list) -> None:
    """
    Evaluate the trained model and display metrics.
    """
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Retrain on training split for fair evaluation
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # AUC score
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\nAUC Score: {auc_score:.4f}")
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print(f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance
    print("\nTop 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))


def save_model(model: RandomForestClassifier, feature_names: list) -> None:
    """
    Save the trained model and feature names.
    """
    model_dir = Path(__file__).parent
    
    # Save the model
    model_path = model_dir / "model.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save feature names for consistency during inference
    feature_names_path = model_dir / "feature_names.pkl"
    joblib.dump(feature_names, feature_names_path)
    print(f"Feature names saved to: {feature_names_path}")
    
    # Save model metadata
    metadata = {
        'model_type': 'RandomForestClassifier',
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'training_date': pd.Timestamp.now().isoformat(),
        'model_params': model.get_params()
    }
    
    import json
    metadata_path = model_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to: {metadata_path}")


def main():
    """Main training pipeline."""
    print("Starting model training pipeline...")
    
    # Load data
    df = load_training_data()
    print(f"Loaded training data: {len(df)} samples")
    
    # Prepare data
    X, y, feature_names = prepare_data(df)
    
    # Train model
    model = train_model(X, y)
    
    # Evaluate model
    evaluate_model(model, X, y, feature_names)
    
    # Save model
    save_model(model, feature_names)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print("Model ready for deployment. You can now:")
    print("1. Set up Feast feature store")
    print("2. Start the prediction API server")
    print("3. Send real-time prediction requests")


if __name__ == "__main__":
    main()
