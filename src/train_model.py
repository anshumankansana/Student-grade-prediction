import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import os
from typing import Dict, Tuple, Any

class GradePredictor:
    def __init__(self):
        """Initialize the grade prediction model."""
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = [
            'hours_studied_per_week',
            'attendance_percentage', 
            'previous_grade',
            'sleep_hours_per_night',
            'has_tutor',
            'study_group_participation',
            'assignments_completion_percentage',
            'extracurricular_hours_per_week'
        ]
        self.metrics = {}
        self.feature_importance = {}
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load student data from CSV file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        data = pd.read_csv(filepath)
        print(f"Loaded {len(data)} records from {filepath}")
        return data
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target variables."""
        X = data[self.feature_names].values
        y = data['final_grade'].values
        
        # Handle any missing values
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train the model and return performance metrics."""
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        print("Training the model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, 
                                   cv=5, scoring='r2')
        
        # Store metrics
        self.metrics = {
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'cv_r2_mean': float(cv_scores.mean()),
            'cv_r2_std': float(cv_scores.std())
        }
        
        # Store feature importance
        importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
        self.feature_importance = {k: float(v) for k, v in importance_dict.items()}
        
        # Print results
        print(f"\nModel Performance:")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Cross-validation R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        print(f"\nFeature Importance:")
        for feature, importance in sorted(self.feature_importance.items(), 
                                        key=lambda x: x[1], reverse=True):
            print(f"{feature}: {importance:.4f}")
        
        return self.metrics
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        return np.clip(predictions, 0, 100)  # Ensure grades are in valid range
    
    def save_model(self, model_dir: str = "model"):
        """Save the trained model and scaler."""
        os.makedirs(model_dir, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance
        }
        
        model_path = os.path.join(model_dir, "grade_predictor.pkl")
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
        
        # Save metrics as JSON for easy access
        metrics_path = os.path.join(model_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump({
                'metrics': self.metrics,
                'feature_importance': self.feature_importance
            }, f, indent=2)
        print(f"Metrics saved to {metrics_path}")
    
    @classmethod
    def load_model(cls, model_path: str = "model/grade_predictor.pkl"):
        """Load a trained model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        
        predictor = cls()
        predictor.model = model_data['model']
        predictor.scaler = model_data['scaler']
        predictor.feature_names = model_data['feature_names']
        predictor.metrics = model_data['metrics']
        predictor.feature_importance = model_data['feature_importance']
        
        return predictor

def main():
    """Train and save the grade prediction model."""
    print("Starting model training...")
    
    # Initialize predictor
    predictor = GradePredictor()
    
    # Load data
    try:
        data = predictor.load_data("data/student_data.csv")
    except FileNotFoundError:
        print("Data file not found. Please run data_generator.py first.")
        return
    
    # Prepare data
    X, y = predictor.prepare_data(data)
    
    # Train model
    metrics = predictor.train(X, y)
    
    # Save model
    predictor.save_model()
    
    print("\nModel training completed successfully!")
    print(f"Final test R²: {metrics['test_r2']:.4f}")
    print(f"Final test RMSE: {metrics['test_rmse']:.4f}")

if __name__ == "__main__":
    main()