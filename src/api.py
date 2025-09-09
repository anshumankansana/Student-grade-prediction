from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional
from src.train_model import GradePredictor

app = FastAPI(
    title="Student Grade Prediction API",
    description="API for predicting student final grades based on study habits and performance metrics",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class StudentFeatures(BaseModel):
    """Model for student feature input."""
    hours_studied_per_week: float = Field(..., ge=0, le=50, description="Hours studied per week")
    attendance_percentage: float = Field(..., ge=0, le=100, description="Attendance percentage")
    previous_grade: float = Field(..., ge=0, le=100, description="Previous grade")
    sleep_hours_per_night: float = Field(..., ge=0, le=24, description="Sleep hours per night")
    has_tutor: bool = Field(..., description="Whether student has a tutor")
    study_group_participation: bool = Field(..., description="Study group participation")
    assignments_completion_percentage: float = Field(..., ge=0, le=100, description="Assignment completion percentage")
    extracurricular_hours_per_week: float = Field(..., ge=0, le=50, description="Extracurricular hours per week")

class PredictionResponse(BaseModel):
    """Model for prediction response."""
    predicted_grade: float
    confidence_interval: Dict[str, float]
    features_used: Dict[str, Any]

class MetricsResponse(BaseModel):
    """Model for metrics response."""
    model_metrics: Dict[str, float]
    feature_importance: Dict[str, float]

# Global model instance
predictor = None

def load_model():
    """Load the trained model."""
    global predictor
    try:
        predictor = GradePredictor.load_model("model/grade_predictor.pkl")
        print("Model loaded successfully")
    except FileNotFoundError:
        print("Warning: Model not found. Please train the model first.")
        predictor = None

# Load model on startup
load_model()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page."""
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found</h1><p>Please ensure static/index.html exists.</p>")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_status = "loaded" if predictor is not None else "not loaded"
    return {"status": "healthy", "model_status": model_status}

@app.get("/sample-data")
async def get_sample_data() -> List[Dict[str, Any]]:
    """Get sample student data."""
    try:
        sample_data = pd.read_csv("data/sample_data.csv")
        return sample_data.head(20).to_dict('records')
    except FileNotFoundError:
        # Return hardcoded sample if file doesn't exist
        return [
            {
                "hours_studied_per_week": 20.5,
                "attendance_percentage": 85.0,
                "previous_grade": 78.5,
                "sleep_hours_per_night": 7.0,
                "has_tutor": True,
                "study_group_participation": False,
                "assignments_completion_percentage": 92.0,
                "extracurricular_hours_per_week": 5.0,
                "final_grade": 82.3
            }
        ]

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get model performance metrics and feature importance."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return MetricsResponse(
        model_metrics=predictor.metrics,
        feature_importance=predictor.feature_importance
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_grade(features: StudentFeatures):
    """Predict student final grade based on input features."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to numpy array
        feature_array = np.array([[
            features.hours_studied_per_week,
            features.attendance_percentage,
            features.previous_grade,
            features.sleep_hours_per_night,
            int(features.has_tutor),
            int(features.study_group_participation),
            features.assignments_completion_percentage,
            features.extracurricular_hours_per_week
        ]])
        
        # Make prediction
        prediction = predictor.predict(feature_array)[0]
        
        # Calculate confidence interval (simplified)
        uncertainty = 5.0  # Based on typical RMSE
        confidence_interval = {
            "lower": max(0, prediction - uncertainty),
            "upper": min(100, prediction + uncertainty)
        }
        
        return PredictionResponse(
            predicted_grade=round(prediction, 2),
            confidence_interval=confidence_interval,
            features_used=features.dict()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/data-distribution")
async def get_data_distribution():
    """Get data distribution for visualization."""
    try:
        data = pd.read_csv("data/student_data.csv")
        
        # Calculate distributions
        grade_distribution = data['final_grade'].value_counts(bins=10).sort_index().to_dict()
        
        feature_stats = {}
        for feature in data.columns:
            if feature != 'final_grade':
                feature_stats[feature] = {
                    'mean': float(data[feature].mean()),
                    'std': float(data[feature].std()),
                    'min': float(data[feature].min()),
                    'max': float(data[feature].max())
                }
        
        return {
            "grade_distribution": grade_distribution,
            "feature_statistics": feature_stats,
            "total_records": len(data)
        }
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Data file not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)