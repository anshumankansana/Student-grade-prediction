import pandas as pd
import numpy as np
import random
from typing import List, Dict
import os

class StudentDataGenerator:
    def __init__(self, seed: int = 42):
        """Initialize the data generator with a random seed for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_student_data(self, n_students: int = 1000) -> pd.DataFrame:
        """Generate synthetic student data with realistic correlations."""
        
        # Base features
        hours_studied = np.random.normal(15, 5, n_students)
        hours_studied = np.clip(hours_studied, 2, 40)
        
        attendance = np.random.beta(8, 2, n_students) * 100
        attendance = np.clip(attendance, 40, 100)
        
        sleep_hours = np.random.normal(7, 1.5, n_students)
        sleep_hours = np.clip(sleep_hours, 4, 12)
        
        previous_grades = np.random.normal(75, 12, n_students)
        previous_grades = np.clip(previous_grades, 30, 100)
        
        # Binary features with realistic probabilities
        has_tutor = np.random.choice([0, 1], n_students, p=[0.7, 0.3])
        study_group = np.random.choice([0, 1], n_students, p=[0.6, 0.4])
        
        # Assignment completion with correlation to other factors
        base_completion = np.random.beta(7, 2, n_students) * 100
        completion_bonus = (hours_studied - 10) * 2 + (attendance - 70) * 0.5
        assignments_completion = base_completion + completion_bonus
        assignments_completion = np.clip(assignments_completion, 20, 100)
        
        extracurricular_hours = np.random.exponential(3, n_students)
        extracurricular_hours = np.clip(extracurricular_hours, 0, 20)
        
        # Calculate final grade with realistic correlations
        final_grade = self._calculate_realistic_grade(
            hours_studied, attendance, sleep_hours, previous_grades,
            has_tutor, study_group, assignments_completion, extracurricular_hours
        )
        
        # Create DataFrame
        data = pd.DataFrame({
            'hours_studied_per_week': np.round(hours_studied, 1),
            'attendance_percentage': np.round(attendance, 1),
            'previous_grade': np.round(previous_grades, 1),
            'sleep_hours_per_night': np.round(sleep_hours, 1),
            'has_tutor': has_tutor,
            'study_group_participation': study_group,
            'assignments_completion_percentage': np.round(assignments_completion, 1),
            'extracurricular_hours_per_week': np.round(extracurricular_hours, 1),
            'final_grade': np.round(final_grade, 1)
        })
        
        return data
    
    def _calculate_realistic_grade(self, hours_studied, attendance, sleep_hours, 
                                 previous_grades, has_tutor, study_group, 
                                 assignments_completion, extracurricular_hours):
        """Calculate final grade based on input features with realistic relationships."""
        
        # Base grade from previous performance
        base_grade = previous_grades * 0.4
        
        # Study time impact (diminishing returns)
        study_impact = np.minimum(hours_studied * 1.2, 25)
        
        # Attendance impact
        attendance_impact = (attendance - 50) * 0.3
        
        # Sleep impact (optimal around 7-8 hours)
        optimal_sleep = 7.5
        sleep_penalty = np.abs(sleep_hours - optimal_sleep) * -2
        
        # Tutor bonus
        tutor_bonus = has_tutor * 5
        
        # Study group bonus
        group_bonus = study_group * 3
        
        # Assignment completion impact
        assignment_impact = (assignments_completion - 50) * 0.4
        
        # Extracurricular balance (too much hurts, moderate amount helps)
        extra_impact = np.where(extracurricular_hours < 8, 
                               extracurricular_hours * 0.5,
                               8 * 0.5 - (extracurricular_hours - 8) * 1.0)
        
        # Combine all factors
        final_grade = (base_grade + study_impact + attendance_impact + 
                      sleep_penalty + tutor_bonus + group_bonus + 
                      assignment_impact + extra_impact)
        
        # Add some random noise
        noise = np.random.normal(0, 3, len(final_grade))
        final_grade += noise
        
        # Ensure grades are within realistic bounds
        final_grade = np.clip(final_grade, 0, 100)
        
        return final_grade
    
    def save_data(self, data: pd.DataFrame, filepath: str):
        """Save the generated data to a CSV file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        print(f"Generated {len(data)} student records")
        print(f"Final grade statistics:")
        print(data['final_grade'].describe())

def main():
    """Generate and save student data."""
    generator = StudentDataGenerator()
    
    # Generate training data
    train_data = generator.generate_student_data(n_students=1200)
    generator.save_data(train_data, "data/student_data.csv")
    
    # Generate sample data for API
    sample_data = generator.generate_student_data(n_students=100)
    generator.save_data(sample_data, "data/sample_data.csv")

if __name__ == "__main__":
    main()