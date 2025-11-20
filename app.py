from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Optional, good for API usage

model = joblib.load('model/churn_model.pkl')
CUSTOMERS = pd.read_csv('data/retail_customers_churn.csv').fillna("")

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "ok", "message": "Retail Churn API is running!"})

@app.route('/dashboard', methods=['GET'])
def dashboard():
    total = len(CUSTOMERS)
    churn_rate = 100 * (CUSTOMERS['Churn'] == 'Yes').mean()
    at_risk = CUSTOMERS[CUSTOMERS['Churn'] == 'Yes']
    total_revenue = CUSTOMERS['Total_Amount'].sum()
    dashboard_data = {
        "total_customers": total,
        "churn_rate": round(churn_rate, 1),
        "at_risk_customers": len(at_risk),
        "total_revenue": round(total_revenue, 2)
    }
    return jsonify(dashboard_data)

@app.route('/customers', methods=['GET'])
def customers():
    return jsonify([
        {
            "name": row['Name'],
            "email": row['Email'],
            "churn_status": row['Churn'],
            "risk_score": 85 if row['Churn'] == 'Yes' else 20
        }
        for _, row in CUSTOMERS.iterrows()
    ])

@app.route('/predict-churn', methods=['POST'])
def predict_churn():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    for col in ['Gender', 'Income', 'Country']:
        if col in input_df:
            input_df[col] = pd.factorize([input_df[col][0]])[0]
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0, 1]
    return jsonify({
        "prediction": int(pred),
        "probability_of_churn": float(prob)
    })

if __name__ == '__main__':
    app.run(debug=True)
