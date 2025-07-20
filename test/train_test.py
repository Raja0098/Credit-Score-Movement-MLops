import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pytest

# Import the functions from your training script
from src.train import load_data, train_model

# Test Case 1: Check if the data file loads correctly
def test_data_loading():
    """
    Tests that the data loading function returns a pandas DataFrame.
    """
    dummy_data = "age,monthly_income,credit_score_t-1,total_credit_limit\n30,5000,700,50000"
    with open("dummy_data.csv", "w") as f:
        f.write(dummy_data)
    
    df = load_data("dummy_data.csv")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

# Test Case 2: Check if the model training function works
def test_model_training():
    """
    Tests that the model training function returns a trained model object.
    """
    data = {
        'age': [25, 30, 35],
        'monthly_income': [5000, 6000, 7000],
        'credit_score_t-1': [700, 650, 720],
        'total_credit_limit': [50000, 40000, 60000]
    }
    df = pd.DataFrame(data)
    
    features = ['age', 'monthly_income', 'credit_score_t-1']
    target = 'total_credit_limit'
    
    # FIX: Pass the missing arguments to the function call
    model = train_model(df, features, target, n_estimators=10, random_state=42)
    
    # Assert that the output is a scikit-learn classifier
    assert isinstance(model, RandomForestClassifier)
