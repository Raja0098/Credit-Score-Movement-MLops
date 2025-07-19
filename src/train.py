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

# The main block below only runs when you execute 'python src/train.py'
# It does NOT run when the test file imports from it
if __name__ == "__main__":
    print("Loading data...")
    # Use your actual data file name if it's different
    df = load_data('data/simulated_data.csv') 
    
    features = ['age', 'monthly_income', 'credit_score_t-1']
    target = 'total_credit_limit' # Replace with your actual target column

    print("Training model...")
    model = train_model(df, features, target)

    joblib.dump(model, 'model.joblib')
    print("Model trained and saved as model.joblib")
