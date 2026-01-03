"""
Machine Learning Predictor for FlowFix AI
Predicts task duration and delay probability using historical data
"""
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import execute_query

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_training_data():
    """Load and prepare data for ML training"""
    print("📊 Loading training data...")
    
    query = """
    SELECT 
        task_id,
        assignee,
        status,
        priority,
        project,
        actual_duration,
        is_delayed,
        bottleneck_type
    FROM tasks
    WHERE actual_duration IS NOT NULL
    """
    
    df = execute_query(query)
    print(f"✅ Loaded {len(df)} completed tasks")
    
    return df


def engineer_features(df):
    """Create features for ML models"""
    print("🔧 Engineering features...")
    
    # Encode categorical variables
    label_encoders = {}
    
    categorical_cols = ['assignee', 'priority', 'project', 'status']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Create assignee statistics features
    assignee_stats = df.groupby('assignee')['actual_duration'].agg([
        'mean', 'std', 'count', 'median'
    ]).reset_index()
    assignee_stats.columns = ['assignee', 'assignee_avg_duration', 
                               'assignee_std_duration', 'assignee_task_count',
                               'assignee_median_duration']
    
    df = df.merge(assignee_stats, on='assignee', how='left')
    
    # Create project statistics features
    project_stats = df.groupby('project')['actual_duration'].agg([
        'mean', 'count'
    ]).reset_index()
    project_stats.columns = ['project', 'project_avg_duration', 'project_task_count']
    
    df = df.merge(project_stats, on='project', how='left')
    
    # Fill NaN values
    df['assignee_std_duration'] = df['assignee_std_duration'].fillna(0)
    
    print("✅ Feature engineering complete")
    
    return df, label_encoders


def train_duration_predictor(df, label_encoders):
    """Train regression model to predict task duration"""
    print("\n" + "="*60)
    print("🤖 Training Duration Prediction Model")
    print("="*60 + "\n")
    
    # Select features
    feature_cols = [
        'assignee_encoded',
        'priority_encoded',
        'project_encoded',
        'assignee_avg_duration',
        'assignee_std_duration',
        'assignee_task_count',
        'project_avg_duration',
        'project_task_count'
    ]
    
    X = df[feature_cols].copy()
    y = df['actual_duration'].copy()
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples\n")
    
    # Train model
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print("\n📊 Model Performance:")
    print(f"   Train MAE: {train_mae:.2f} days")
    print(f"   Test MAE: {test_mae:.2f} days")
    print(f"   Train R²: {train_r2:.3f}")
    print(f"   Test R²: {test_r2:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📈 Top 5 Important Features:")
    for idx, row in feature_importance.head().iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Save model
    model_path = os.path.join(MODEL_DIR, 'duration_predictor.pkl')
    joblib.dump({
        'model': model,
        'feature_cols': feature_cols,
        'label_encoders': label_encoders,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2
    }, model_path)
    
    print(f"\n✅ Model saved to {model_path}")
    
    return model, feature_cols


def train_delay_classifier(df, label_encoders):
    """Train classification model to predict if task will be delayed"""
    print("\n" + "="*60)
    print("🤖 Training Delay Classification Model")
    print("="*60 + "\n")
    
    # Select features
    feature_cols = [
        'assignee_encoded',
        'priority_encoded',
        'project_encoded',
        'assignee_avg_duration',
        'assignee_std_duration',
        'assignee_task_count',
        'project_avg_duration',
        'project_task_count'
    ]
    
    X = df[feature_cols].copy()
    y = df['is_delayed'].copy()
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    
    # Check class distribution
    print(f"Class distribution:")
    print(f"   Not Delayed: {(y == 0).sum()}")
    print(f"   Delayed: {(y == 1).sum()}\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples\n")
    
    # Train model
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print("\n📊 Model Performance:")
    print(f"   Train Accuracy: {train_acc:.3f}")
    print(f"   Test Accuracy: {test_acc:.3f}")
    
    print("\n📋 Classification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, 
                                target_names=['Not Delayed', 'Delayed']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📈 Top 5 Important Features:")
    for idx, row in feature_importance.head().iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Save model
    model_path = os.path.join(MODEL_DIR, 'delay_classifier.pkl')
    joblib.dump({
        'model': model,
        'feature_cols': feature_cols,
        'label_encoders': label_encoders,
        'train_acc': train_acc,
        'test_acc': test_acc
    }, model_path)
    
    print(f"\n✅ Model saved to {model_path}")
    
    return model, feature_cols


def predict_new_task(task_data, model_type='duration'):
    """
    Make prediction for a new task
    
    Args:
        task_data: dict with keys: assignee, priority, project
        model_type: 'duration' or 'delay'
    """
    # Load model
    model_path = os.path.join(MODEL_DIR, f'{model_type}_predictor.pkl' 
                              if model_type == 'duration' 
                              else f'{model_type}_classifier.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Please train the model first.")
    
    model_data = joblib.load(model_path)
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    label_encoders = model_data['label_encoders']
    
    # Prepare features (simplified version - in production, would need full feature engineering)
    # This is a placeholder for demonstration
    
    return None


def train_all_models():
    """Train all ML models"""
    print("\n" + "="*60)
    print("🚀 STARTING ML MODEL TRAINING")
    print("="*60 + "\n")
    
    # Load data
    df = load_training_data()
    
    if len(df) < 50:
        print("⚠️  Warning: Limited training data. Results may not be reliable.")
    
    # Engineer features
    df, label_encoders = engineer_features(df)
    
    # Train duration predictor
    duration_model, duration_features = train_duration_predictor(df, label_encoders)
    
    # Train delay classifier
    delay_model, delay_features = train_delay_classifier(df, label_encoders)
    
    print("\n" + "="*60)
    print("✅ ALL MODELS TRAINED SUCCESSFULLY")
    print("="*60 + "\n")
    
    return {
        'duration_model': duration_model,
        'delay_model': delay_model,
        'label_encoders': label_encoders
    }


if __name__ == "__main__":
    models = train_all_models()
    
    print("\n📦 Models saved in 'models/' directory:")
    print("   - duration_predictor.pkl")
    print("   - delay_classifier.pkl")
