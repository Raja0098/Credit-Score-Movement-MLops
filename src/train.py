import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_data(file_path):
    """Loads data from a CSV file."""
    return pd.read_csv(file_path)

def train_model(df, features, target):
    """Trains a RandomForestClassifier model."""
    X = df[features]
    y = df[target]
    
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# This main block runs only when you execute 'python src/train.py'
if __name__ == "__main__":
    
    # Use your actual data file name
    df = load_data('data/credit_data.csv') 
    
    # Your updated features and target
    features = ['age', 'monthly_income', 'credit_score_t-1']
    target = 'target_credit_score_movement'

    print("Training model...")
    model = train_model(df, features, target)

    joblib.dump(model, 'model.joblib')
    print("Model trained and saved as model.joblib")
