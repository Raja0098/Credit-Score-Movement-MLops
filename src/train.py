import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv('data/credit_data.csv')

# Simple feature selection for this PoC
features = ['age', 'monthly_income', 'credit_score_t-1']
target = 'target_credit_score_movement' # Replace with your actual target column

X = df[features]
y = df[target]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model artifact
joblib.dump(model, 'model.joblib')
print("Model trained and saved to model.joblib")