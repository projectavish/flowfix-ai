"""
Machine Learning Predictor for FlowFix AI - Production Grade
Predicts task duration and delay probability with SHAP, versioning, and logging
"""
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os
import logging
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
from sqlalchemy import text
from utils import get_engine, execute_query

# Try to import SHAP for explainability
try:
    import shap
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available. Install with: pip install shap")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_DIR = "models"
SHAP_DIR = os.path.join(MODEL_DIR, "shap_plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SHAP_DIR, exist_ok=True)

# Model version - increment when making significant changes
MODEL_VERSION = "1.1"


def load_training_data():
    """Load and prepare data for ML training"""
    logger.info("📊 Loading training data...")
    
    query = """
    SELECT 
        task_id,
        task_name,
        assignee,
        status,
        priority,
        project,
        actual_duration,
        is_delayed,
        bottleneck_type,
        reassignment_count,
        comments,
        start_date,
        end_date
    FROM tasks
    WHERE actual_duration IS NOT NULL
    """
    
    df = execute_query(query)
    logger.info(f"✅ Loaded {len(df)} completed tasks")
    
    return df


def engineer_features(df):
    """
    Create advanced features for ML models
    
    New features:
    - task_name_length: Length of task name (complexity indicator)
    - comment_length: Length of comments field
    - days_in_project: Ordinal encoding by project duration
    - priority_ordinal: Ordinal encoding (High=3, Medium=2, Low=1)
    - has_bottleneck: Binary flag if task had bottleneck
    - reassignment_flag: Binary flag if task was reassigned
    """
    logger.info("🔧 Engineering features...")
    
    df = df.copy()
    
    # Text-based features
    df['task_name_length'] = df['task_name'].fillna('').astype(str).apply(len)
    df['comment_length'] = df['comments'].fillna('').astype(str).apply(len)
    df['has_comment'] = (df['comment_length'] > 0).astype(int)
    
    # Bottleneck features
    df['has_bottleneck'] = df['bottleneck_type'].notna().astype(int)
    
    # Reassignment features
    df['reassignment_count'] = df['reassignment_count'].fillna(0).astype(int)
    df['reassignment_flag'] = (df['reassignment_count'] > 0).astype(int)
    
    # Priority ordinal encoding
    priority_map = {'High': 3, 'Medium': 2, 'Low': 1}
    df['priority_ordinal'] = df['priority'].map(priority_map).fillna(1).astype(int)
    
    # Date features (if available)
    if 'start_date' in df.columns and 'end_date' in df.columns:
        try:
            df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
            df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
            df['date_range'] = (df['end_date'] - df['start_date']).dt.days
            df['date_range'] = df['date_range'].fillna(0).clip(lower=0)
        except:
            df['date_range'] = 0
    else:
        df['date_range'] = 0
    
    # Encode categorical variables
    label_encoders = {}
    
    categorical_cols = ['assignee', 'project', 'status']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Create assignee statistics features
    assignee_stats = df.groupby('assignee').agg({
        'actual_duration': ['mean', 'std', 'count', 'median'],
        'is_delayed': 'mean'
    }).reset_index()
    assignee_stats.columns = [
        'assignee', 'assignee_avg_duration', 'assignee_std_duration', 
        'assignee_task_count', 'assignee_median_duration', 'assignee_delay_rate'
    ]
    
    df = df.merge(assignee_stats, on='assignee', how='left')
    
    # Create project statistics features
    project_stats = df.groupby('project').agg({
        'actual_duration': ['mean', 'count'],
        'is_delayed': 'mean'
    }).reset_index()
    project_stats.columns = [
        'project', 'project_avg_duration', 'project_task_count', 'project_delay_rate'
    ]
    
    df = df.merge(project_stats, on='project', how='left')
    
    # Fill NaN values
    df['assignee_std_duration'] = df['assignee_std_duration'].fillna(0)
    df['assignee_delay_rate'] = df['assignee_delay_rate'].fillna(0)
    df['project_delay_rate'] = df['project_delay_rate'].fillna(0)
    
    logger.info("✅ Feature engineering complete")
    
    return df, label_encoders


def log_training_run(model_type, metrics, feature_importance, version=MODEL_VERSION):
    """Log training run to database"""
    engine = get_engine()
    
    query = text("""
        INSERT INTO ml_training_log 
        (model_type, model_version, metrics_json, feature_importance_json)
        VALUES (:model_type, :version, :metrics, :features)
    """)
    
    try:
        with engine.connect() as conn:
            conn.execute(query, {
                'model_type': model_type,
                'version': version,
                'metrics': json.dumps(metrics),
                'features': json.dumps(feature_importance)
            })
            conn.commit()
        logger.info(f"✅ Logged training run for {model_type}")
    except Exception as e:
        logger.error(f"Failed to log training run: {e}")


def create_shap_visualizations(model, X_train, X_test, feature_cols, model_type):
    """Create SHAP visualizations for model explainability"""
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available, skipping visualizations")
        return
    
    try:
        logger.info(f"🎨 Generating SHAP visualizations for {model_type}...")
        
        # Create explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values (use smaller sample for speed)
        sample_size = min(100, len(X_test))
        X_sample = X_test.sample(n=sample_size, random_state=42) if len(X_test) > sample_size else X_test
        shap_values = explainer.shap_values(X_sample)
        
        # For binary classification, use positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False)
        summary_path = os.path.join(SHAP_DIR, f'{model_type}_summary.png')
        plt.savefig(summary_path, bbox_inches='tight', dpi=100)
        plt.close()
        logger.info(f"   Saved summary plot: {summary_path}")
        
        # Feature importance bar plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, 
                         plot_type="bar", show=False)
        bar_path = os.path.join(SHAP_DIR, f'{model_type}_importance.png')
        plt.savefig(bar_path, bbox_inches='tight', dpi=100)
        plt.close()
        logger.info(f"   Saved importance plot: {bar_path}")
        
        logger.info("✅ SHAP visualizations created")
        
    except Exception as e:
        logger.error(f"Error creating SHAP visualizations: {e}")


def find_optimal_threshold(y_true, y_proba):
    """
    Find optimal classification threshold using Youden's J statistic
    
    Returns: optimal_threshold, metrics_dict
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    # Youden's J statistic = sensitivity + specificity - 1
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold, {
        'optimal_threshold': float(optimal_threshold),
        'tpr_at_threshold': float(tpr[optimal_idx]),
        'fpr_at_threshold': float(fpr[optimal_idx]),
        'j_score': float(j_scores[optimal_idx])
    }


def train_duration_predictor(df, label_encoders):
    """Train regression model to predict task duration"""
    logger.info("\n" + "="*60)
    logger.info("🤖 Training Duration Prediction Model")
    logger.info("="*60 + "\n")
    
    # Select features (enhanced)
    feature_cols = [
        'assignee_encoded',
        'priority_ordinal',
        'project_encoded',
        'status_encoded',
        'assignee_avg_duration',
        'assignee_std_duration',
        'assignee_task_count',
        'assignee_delay_rate',
        'project_avg_duration',
        'project_task_count',
        'project_delay_rate',
        'task_name_length',
        'comment_length',
        'has_comment',
        'has_bottleneck',
        'reassignment_count',
        'reassignment_flag',
        'date_range'
    ]
    
    X = df[feature_cols].copy()
    y = df['actual_duration'].copy()
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples\n")
    
    # Train model
    logger.info("Training Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_mae = -cv_scores.mean()
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    logger.info("\n📊 Model Performance:")
    logger.info(f"   Train MAE: {train_mae:.2f} days")
    logger.info(f"   Test MAE: {test_mae:.2f} days")
    logger.info(f"   CV MAE: {cv_mae:.2f} days")
    logger.info(f"   Train RMSE: {train_rmse:.2f} days")
    logger.info(f"   Test RMSE: {test_rmse:.2f} days")
    logger.info(f"   Train R²: {train_r2:.3f}")
    logger.info(f"   Test R²: {test_r2:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\n📈 Top 10 Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Create SHAP visualizations
    create_shap_visualizations(model, X_train, X_test, feature_cols, 'duration')
    
    # Prepare metrics for logging
    metrics = {
        'train_mae': float(train_mae),
        'test_mae': float(test_mae),
        'cv_mae': float(cv_mae),
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
    
    feature_importance_dict = feature_importance.set_index('feature')['importance'].to_dict()
    feature_importance_dict = {k: float(v) for k, v in feature_importance_dict.items()}
    
    # Log to database
    log_training_run('duration_predictor', metrics, feature_importance_dict)
    
    # Save model with version and timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'duration_predictor_v{MODEL_VERSION}_{timestamp}.pkl'
    model_path = os.path.join(MODEL_DIR, model_filename)
    
    joblib.dump({
        'model': model,
        'feature_cols': feature_cols,
        'label_encoders': label_encoders,
        'metrics': metrics,
        'version': MODEL_VERSION,
        'trained_at': datetime.now().isoformat()
    }, model_path)
    
    # Also save as latest
    latest_path = os.path.join(MODEL_DIR, 'duration_predictor_latest.pkl')
    joblib.dump({
        'model': model,
        'feature_cols': feature_cols,
        'label_encoders': label_encoders,
        'metrics': metrics,
        'version': MODEL_VERSION,
        'trained_at': datetime.now().isoformat()
    }, latest_path)
    
    logger.info(f"\n✅ Model saved to {model_path}")
    logger.info(f"✅ Latest model saved to {latest_path}")
    
    return model, feature_cols


def train_delay_classifier(df, label_encoders):
    """Train classification model to predict if task will be delayed"""
    logger.info("\n" + "="*60)
    logger.info("🤖 Training Delay Classification Model")
    logger.info("="*60 + "\n")
    
    # Select features (enhanced)
    feature_cols = [
        'assignee_encoded',
        'priority_ordinal',
        'project_encoded',
        'status_encoded',
        'assignee_avg_duration',
        'assignee_std_duration',
        'assignee_task_count',
        'assignee_delay_rate',
        'project_avg_duration',
        'project_task_count',
        'project_delay_rate',
        'task_name_length',
        'comment_length',
        'has_comment',
        'has_bottleneck',
        'reassignment_count',
        'reassignment_flag',
        'date_range'
    ]
    
    X = df[feature_cols].copy()
    y = df['is_delayed'].copy()
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    
    # Check class distribution
    logger.info(f"Class distribution:")
    logger.info(f"   Not Delayed: {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")
    logger.info(f"   Delayed: {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples\n")
    
    # Train model
    logger.info("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                scoring='accuracy', n_jobs=-1)
    cv_acc = cv_scores.mean()
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:, 1]
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(y_test, y_proba_test)
    except:
        roc_auc = 0.5
    
    # Find optimal threshold
    optimal_threshold, threshold_metrics = find_optimal_threshold(y_test, y_proba_test)
    
    logger.info("\n📊 Model Performance:")
    logger.info(f"   Train Accuracy: {train_acc:.3f}")
    logger.info(f"   Test Accuracy: {test_acc:.3f}")
    logger.info(f"   CV Accuracy: {cv_acc:.3f}")
    logger.info(f"   ROC AUC: {roc_auc:.3f}")
    logger.info(f"   Optimal Threshold: {optimal_threshold:.3f}")
    
    logger.info("\n📋 Classification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, 
                                target_names=['Not Delayed', 'Delayed']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\n📈 Top 10 Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Create SHAP visualizations
    create_shap_visualizations(model, X_train, X_test, feature_cols, 'delay')
    
    # Prepare metrics for logging
    metrics = {
        'train_acc': float(train_acc),
        'test_acc': float(test_acc),
        'cv_acc': float(cv_acc),
        'roc_auc': float(roc_auc),
        'optimal_threshold': float(optimal_threshold),
        'n_train': len(X_train),
        'n_test': len(X_test),
        **threshold_metrics
    }
    
    feature_importance_dict = feature_importance.set_index('feature')['importance'].to_dict()
    feature_importance_dict = {k: float(v) for k, v in feature_importance_dict.items()}
    
    # Log to database
    log_training_run('delay_classifier', metrics, feature_importance_dict)
    
    # Save model with version and timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'delay_classifier_v{MODEL_VERSION}_{timestamp}.pkl'
    model_path = os.path.join(MODEL_DIR, model_filename)
    
    joblib.dump({
        'model': model,
        'feature_cols': feature_cols,
        'label_encoders': label_encoders,
        'metrics': metrics,
        'optimal_threshold': optimal_threshold,
        'version': MODEL_VERSION,
        'trained_at': datetime.now().isoformat()
    }, model_path)
    
    # Also save as latest
    latest_path = os.path.join(MODEL_DIR, 'delay_classifier_latest.pkl')
    joblib.dump({
        'model': model,
        'feature_cols': feature_cols,
        'label_encoders': label_encoders,
        'metrics': metrics,
        'optimal_threshold': optimal_threshold,
        'version': MODEL_VERSION,
        'trained_at': datetime.now().isoformat()
    }, latest_path)
    
    logger.info(f"\n✅ Model saved to {model_path}")
    logger.info(f"✅ Latest model saved to {latest_path}")
    
    return model, feature_cols


def save_prediction_to_db(task_id, model_type, prediction, confidence=None):
    """Save prediction to ml_predictions table"""
    engine = get_engine()
    
    query = text("""
        INSERT INTO ml_predictions 
        (task_id, model_type, prediction_value, confidence_score, model_version)
        VALUES (:task_id, :model_type, :prediction, :confidence, :version)
    """)
    
    try:
        with engine.connect() as conn:
            conn.execute(query, {
                'task_id': task_id,
                'model_type': model_type,
                'prediction': float(prediction),
                'confidence': float(confidence) if confidence else None,
                'version': MODEL_VERSION
            })
            conn.commit()
        logger.info(f"✅ Saved prediction for {task_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to save prediction: {e}")
        return False


def predict_new_task(task_data, model_type='duration', save_to_db=True):
    """
    Make prediction for a new task
    
    Args:
        task_data: dict with keys matching required features
        model_type: 'duration' or 'delay'
        save_to_db: whether to save prediction to database
        
    Returns:
        prediction value and metadata
    """
    # Load latest model
    model_filename = f'{model_type}_{"predictor" if model_type == "duration" else "classifier"}_latest.pkl'
    model_path = os.path.join(MODEL_DIR, model_filename)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}. Please train the model first using train_all_models()"
        )
    
    model_data = joblib.load(model_path)
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    
    # Prepare feature vector
    # This would need full feature engineering in production
    # For now, return a simple placeholder
    
    logger.warning("predict_new_task() needs full implementation with feature engineering")
    return None


def train_all_models():
    """Train all ML models with full production features"""
    print("\n" + "="*60)
    print("🚀 STARTING ML MODEL TRAINING - Production Grade")
    print(f"   Version: {MODEL_VERSION}")
    print("="*60 + "\n")
    
    # Load data
    df = load_training_data()
    
    if len(df) < 50:
        logger.warning("⚠️  Warning: Limited training data. Results may not be reliable.")
    
    # Engineer features
    df, label_encoders = engineer_features(df)
    
    # Validation and export
    logger.info("\n📊 Data Validation:")
    logger.info(f"   Total samples: {len(df)}")
    logger.info(f"   Features engineered: {len([c for c in df.columns if '_encoded' in c or 'avg_' in c or 'std_' in c])}")
    logger.info(f"   Missing values: {df.isnull().sum().sum()}")
    
    # Export feature set for analysis
    feature_export_path = os.path.join(MODEL_DIR, f'feature_set_v{MODEL_VERSION}.csv')
    df.to_csv(feature_export_path, index=False)
    logger.info(f"   Exported feature set: {feature_export_path}")
    
    # Train duration predictor
    duration_model, duration_features = train_duration_predictor(df, label_encoders)
    
    # Train delay classifier
    delay_model, delay_features = train_delay_classifier(df, label_encoders)
    
    print("\n" + "="*60)
    print("✅ ALL MODELS TRAINED SUCCESSFULLY")
    print("="*60)
    print(f"\n📦 Models saved in '{MODEL_DIR}/' directory:")
    print(f"   - duration_predictor_v{MODEL_VERSION}_*.pkl")
    print(f"   - delay_classifier_v{MODEL_VERSION}_*.pkl")
    print(f"   - duration_predictor_latest.pkl")
    print(f"   - delay_classifier_latest.pkl")
    if SHAP_AVAILABLE:
        print(f"\n🎨 SHAP visualizations saved in '{SHAP_DIR}/' directory")
    print("\n")
    
    return {
        'duration_model': duration_model,
        'delay_model': delay_model,
        'label_encoders': label_encoders
    }


if __name__ == "__main__":
    import sys
    
    # Check for verbose flag
    if '--verbose' in sys.argv:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Train models
    models = train_all_models()
