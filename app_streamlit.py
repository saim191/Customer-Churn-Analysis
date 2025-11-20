import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import altair as alt
from streamlit_option_menu import option_menu

# --- Risk Explanation Function (for fallback only) ---
def get_risk_reasons(row):
    reasons = []
    if row['Ratings'] < 4:
        reasons.append(f"Low rating ({row['Ratings']})")
    if isinstance(row['Feedback'], str) and (row['Feedback'] in ["Bad", "Poor"] or "bad" in row['Feedback'].lower()):
        reasons.append("Negative feedback")
    if row['Total_Amount'] < 500:
        reasons.append(f"Low spending ({int(row['Total_Amount'])})")
    if isinstance(row['Order_Status'], str) and row['Order_Status'].lower() in ["returned", "pending", "cancelled"]:
        reasons.append(f"Problem order status ({row['Order_Status']})")
    if row.get('Income', '') == 'Low':
        reasons.append("Low income group")
    if not reasons:
        fallback = (f"Ratings: {row['Ratings']}, Feedback: {row['Feedback']}, "
                    f"Amount: {row['Total_Amount']}, Status: {row['Order_Status']}, Income: {row['Income']}")
        return "No specific risk—profile: " + fallback
    return ", ".join(reasons)

# --- Data and Model ---
df = pd.read_csv('data/retail_customers_churn_withreasons.csv').fillna("")
df['Total_Amount'] = pd.to_numeric(df['Total_Amount'], errors='coerce').fillna(0)
feature_columns = [
    'Gender', 'Income', 'Age', 'Country',
    'Total_Amount', 'Feedback', 'Order_Status', 'Ratings'
]
df['ChurnBool'] = (df['Churn'] == 'Yes').astype(int)
model = joblib.load('model/churn_model.pkl')
encoders = joblib.load('model/encoders.pkl')

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

# --- Sidebar Menu ---
with st.sidebar:
    menu = option_menu(
        "Menu", 
        ["Browse Customers", "Churn Analytics", "Predict Churn", "Top High-Risk Customers"],
        icons=['people','bar-chart','activity','star-fill'],
        menu_icon="cast", default_index=0,
        styles={
            "container": {"padding": "6px", "background-color": "#181818"},
            "icon": {"color": "#FFC94D", "font-size": "22px"}, 
            "nav-link": {"font-size": "17px", "text-align": "left", "margin":"4px", "--hover-color": "#222"},
            "nav-link-selected": {"background-color": "#FFC94D", "color":"#181818"},
        }
    )

# --- KPI Cards ---
st.markdown("<h1 style='text-align:center;'>Customer Churn Dashboard</h1>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
total_customers = len(df)
churn_rate = round(100 * (df['Churn'] == 'Yes').mean(), 1)
at_risk_customers = len(df[df['Churn'] == 'Yes'])
total_revenue = round(df['Total_Amount'].sum(), 2)
col1.metric("Total Customers", f"{total_customers}")
col2.metric("Churn Rate", f"{churn_rate} %")
col3.metric("At Risk", f"{at_risk_customers}")
col4.metric("Total Revenue", f"${int(total_revenue):,}")

# --- 1. Browse Customers ---
if menu == "Browse Customers":
    st.header("Browse Customers")
    show_cols = ['Name', 'Email', 'Churn', 'Total_Amount']
    risk_select = st.selectbox(
        "Filter by Risk Level:",
        ("All", "High Risk (Churned)", "Low Risk (Retained)")
    )
    if risk_select == "High Risk (Churned)":
        customer_view = df[df['Churn'] == 'Yes']
    elif risk_select == "Low Risk (Retained)":
        customer_view = df[df['Churn'] == 'No']
    else:
        customer_view = df
    st.dataframe(customer_view[show_cols].head(30), height=400)

# --- 2. Churn Analytics ---
elif menu == "Churn Analytics":
    st.header("Churn Analytics")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Churn Distribution Pie Chart")
        fig_pie = px.pie(df, names='Churn', title='Churn vs. Not Churn', color='Churn')
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_b:
        st.subheader("Churn Rate by Country (Bar)")
        bar = alt.Chart(df).mark_bar().encode(
            x=alt.X('Country:N', sort='-y'),
            y=alt.Y('mean(ChurnBool):Q', title='Churn Rate'),
            tooltip=['Country', alt.Tooltip('mean(ChurnBool):Q', title='Churn Rate')]
        ).properties(width=300)
        st.altair_chart(bar, use_container_width=True)

# --- 3. Predict Churn (with correct encoding) ---
elif menu == "Predict Churn":
    st.header("Predict Churn Risk")
    with st.form("Predict"):
        gender = st.selectbox("Gender", sorted(df['Gender'].unique()))
        income = st.selectbox("Income", sorted(df['Income'].unique()))
        age = st.number_input("Age", min_value=0, max_value=120, value=35)
        country = st.selectbox("Country", sorted(df['Country'].unique()))
        amount = st.number_input("Total Amount", min_value=0, value=1000)
        feedback = st.selectbox("Feedback", sorted(df['Feedback'].unique()))
        order_status = st.selectbox("Order Status", sorted(df['Order_Status'].unique()))
        ratings = st.number_input("Ratings", min_value=0.0, max_value=10.0, value=5.0)
        submit = st.form_submit_button("Predict Risk")
        if submit:
            X_input = {
                'Gender': gender,
                'Income': income,
                'Age': float(age),
                'Country': country,
                'Total_Amount': float(amount),
                'Feedback': feedback,
                'Order_Status': order_status,
                'Ratings': float(ratings)
            }
            # Use same encoders as training!
            for col in ['Gender', 'Income', 'Country', 'Feedback', 'Order_Status']:
                X_input[col] = encoders[col].transform([X_input[col]])[0]
            # Arrange and predict
            X_pred = pd.DataFrame([[X_input[col] for col in feature_columns]], columns=feature_columns)
            pred = model.predict(X_pred)[0]
            proba = model.predict_proba(X_pred)[0, 1]
            st.success(f"Predicted Churn: {'YES ⚠️' if pred==1 else 'NO ✅'} — Risk Score: {int(proba*100)}%")
            # Rule-based reason for this prediction
            dummy_row = pd.Series(X_input)
            dummy_row['Ratings'] = ratings
            dummy_row['Feedback'] = feedback
            dummy_row['Total_Amount'] = amount
            dummy_row['Order_Status'] = order_status
            dummy_row['Income'] = income
            exp = get_risk_reasons(dummy_row)
            st.info(f"Reason: {exp}")

# --- 4. Top High-Risk Customers (actual reason or fallback) ---
elif menu == "Top High-Risk Customers":
    st.header("Top 10 High-Risk Customers (by Total Amount)")
    top_risk = df[df['Churn'] == 'Yes'].sort_values('Total_Amount', ascending=False).head(10)
    for i, row in top_risk.iterrows():
        reason = row['Churn_Reason'] if 'Churn_Reason' in row and isinstance(row['Churn_Reason'], str) and row['Churn_Reason'].strip() else get_risk_reasons(row)
        st.markdown(
            f"""
            <div style="border:1.4px solid #c63; background:#222; color:#eee; box-shadow:2px 2px 12px #0002; padding:13px; margin:7px 0 13px 0; border-radius:10px;">
            <span style="font-weight:bold;">{row['Name']}</span>
            <span style="background:#f8c963; color:#150c00; border-radius:5px; border: 1px solid #c63; padding:4px 9px 4px 9px; margin-left:16px; font-family:monospace; font-size:1.08em;">
            {row['Email']}
            </span>
            <span style="float:right; font-size:1.11em;">💸 ${int(row['Total_Amount']):,}</span>
            <br>
            <span style="color:#fd6060; font-weight:bold; font-size:1.1em;">Risk: ⚠️ HIGH</span>
            <br>
            <span style="color:#e6b324;">Reason: {reason}</span>
            </div>
            """, unsafe_allow_html=True
        )
