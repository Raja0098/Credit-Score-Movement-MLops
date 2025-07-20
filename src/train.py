import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn
import argparse # Import the library

def load_data(file_path):
    """Loads data from a CSV file."""
    return pd.read_csv(file_path)

def train_model(df, features, target, n_estimators, random_state):
    """Trains a model and logs results with MLflow."""
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        # Log parameters that were passed in
        params = {"n_estimators": n_estimators, "random_state": random_state}
        mlflow.log_params(params)

        # Train the model with the specified parameters
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)

        # Evaluate and log metrics
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)
        print(f"Accuracy: {acc:.4f}")

        # Log the model artifact
        mlflow.sklearn.log_model(model, "random-forest-model")
    return model

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the forest")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    args = parser.parse_args()

    df = load_data('data/credit_data.csv')
    features = ['age', 'monthly_income', 'credit_score_t-1']
    target = 'target_credit_score_movement'

    print("Training model and logging with MLflow...")
    # Pass the arguments to the training function
    model = train_model(df, features, target, n_estimators=args.n_estimators, random_state=args.random_state)
    
    joblib.dump(model, 'model.joblib')
    print("Model training complete.")