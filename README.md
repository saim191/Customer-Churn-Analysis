# Customer Churn Analysis 

This repository contains an interactive Streamlit dashboard for predicting and visualizing customer churn, built with Python, machine learning, and modern data visualization libraries.

## 🚀 Features

- Browse Customers: View and filter customer details and churn status.
- Churn Analytics: Interactive pie and bar charts showing churn breakdown and risk by country.
- Predict Churn: Enter new customer details and get instant churn risk predictions.
- Top High-Risk Customers: See the top churned customers and reasons for leaving.
- Churn Explanation: Rule-based and data-driven reasons for customer churn are displayed.

## 🗂️ Project Structure

customer_churn_analysis/
│
├── data/
│ └── sample_customers.csv # Sample dataset with demo customers
│
├── model/
│ └── churn_model.pkl, encoders.pkl (generated; not included in repo—see usage)
│
├── scripts/
│ └── train_model.py # Model training and encoding script
│
├── app_streamlit.py # Streamlit dashboard app
│
├── requirements.txt # Python dependencies
│
└── README.md # Project documentation (this file)


## 📊 Sample Dataset

- `sample_customers.csv` contains 10 demo customers with all required fields, including actual churn reasons for "Yes" records.
- For full use or production, replace with your own customer dataset.

## 🛠️ How to Run

**Requirements:**  
- Python 3.7+
- Install requirements:

**Steps:**

1. Place your customer CSV file (or use `sample_customers.csv`) in the `data` folder.
2. Run the training script to build the model and encoders:
This generates `model/churn_model.pkl` and `model/encoders.pkl`.
3. Launch the dashboard:
4. Open the dashboard at `http://localhost:8501`

## ⚡ How It Works

- All categorical columns are label-encoded for robust ML predictions.
- Dashboard navigation uses side menu and modern Streamlit component styling.
- Predictions use exactly the same encoders as training for trustworthy results.
- Churn explanations use both user input, ML patterns, and (if present) the actual reason from the data.

## 📝 Customizing

- To use your own data, replace the sample CSV; ensure columns match.
- Add actual churn reasons to your data for best experience.
- Retrain the model whenever your data changes.

## 🧑‍💻 For Contributors & Reviewers

- Sample data is provided for demonstration—please do not upload or use real confidential customer data.
- Model and encoder files are not included due to GitHub file size policy; you can recreate them as described above.
- Suggestions and issues welcome!



