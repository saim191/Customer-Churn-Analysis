import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('data/retail_customers_churn.csv').fillna("")

# Features used for model
feature_columns = [
    'Gender', 'Income', 'Age', 'Country',
    'Total_Amount', 'Feedback', 'Order_Status', 'Ratings'
]

X = df[feature_columns].copy()
y = (df['Churn'] == "Yes").astype(int)

# Encode all categorical columns with LabelEncoder and save encoders
categorical = ['Gender', 'Income', 'Country', 'Feedback', 'Order_Status']
encoders = {}
for col in categorical:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Convert numerics and fill missing/blank
X['Total_Amount'] = pd.to_numeric(X['Total_Amount'], errors='coerce').fillna(0)
X['Ratings'] = pd.to_numeric(X['Ratings'], errors='coerce').fillna(0)
X['Age'] = pd.to_numeric(X['Age'], errors='coerce').fillna(0)

# Final nan clean for any leftovers
X = X.fillna(0)

# Split for validation (optional step)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders for dashboard/app use
joblib.dump(model, 'model/churn_model.pkl')
joblib.dump(encoders, 'model/encoders.pkl')

print("Training complete. Model and label encoders saved.")

# Validation accuracy check (optional)
score = model.score(X_test, y_test)
print(f"Validation accuracy: {score:.3f}")
