import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Page config
st.set_page_config(
    page_title="Powergrid Overrun Demo",
    page_icon="⚡",
    layout="wide"
)

# Title and description
st.title("⚡ Powergrid Overrun Analysis")
st.markdown("""
This dashboard analyzes and predicts power grid overrun scenarios using historical data and machine learning.
""")

# Generate synthetic data
@st.cache_data
def generate_synthetic_data():
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2025-09-29', freq='D')
    n_samples = len(dates)
    
    data = pd.DataFrame({
        'Date': dates,
        'Power_Consumption': np.random.normal(1000, 200, n_samples) + \
                           np.sin(np.linspace(0, 4*np.pi, n_samples)) * 100,
        'Temperature': np.random.normal(25, 5, n_samples),
        'Peak_Hours': np.random.randint(0, 24, n_samples),
        'Grid_Load': np.random.uniform(60, 95, n_samples)
    })
    
    # Add some overrun scenarios
    data['Overrun'] = (data['Power_Consumption'] > 1200) & \
                     (data['Grid_Load'] > 85)
    
    return data

# Load or generate data
data = generate_synthetic_data()

# Sidebar controls
st.sidebar.header("Analysis Parameters")
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(data['Date'].min(), data['Date'].max())
)

# Filter data based on date range
mask = (data['Date'].dt.date >= date_range[0]) & \
       (data['Date'].dt.date <= date_range[1])
filtered_data = data.loc[mask]

# Main dashboard
col1, col2 = st.columns(2)

with col1:
    st.subheader("Power Consumption Over Time")
    fig = px.line(filtered_data, x='Date', y='Power_Consumption',
                  color='Overrun', title='Daily Power Consumption')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Grid Load vs Power Consumption")
    fig = px.scatter(filtered_data, x='Grid_Load', y='Power_Consumption',
                    color='Overrun', title='Grid Load Analysis')
    st.plotly_chart(fig, use_container_width=True)

# Predictive Analytics
st.header("Overrun Prediction")

# Prepare features for prediction
features = ['Temperature', 'Peak_Hours', 'Grid_Load']
X = filtered_data[features]
y = filtered_data['Power_Consumption']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Interactive prediction
st.subheader("Predict Power Consumption")
col1, col2, col3 = st.columns(3)

with col1:
    temp = st.number_input("Temperature (°C)", min_value=15.0, max_value=40.0, value=25.0)
with col2:
    peak = st.number_input("Peak Hours", min_value=0, max_value=24, value=12)
with col3:
    load = st.number_input("Grid Load (%)", min_value=60.0, max_value=100.0, value=75.0)

if st.button("Predict"):
    prediction = model.predict([[temp, peak, load]])[0]
    is_overrun = prediction > 1200 and load > 85
    
    st.metric(
        label="Predicted Power Consumption",
        value=f"{prediction:.2f} kW",
        delta="Overrun Risk!" if is_overrun else "Normal",
        delta_color="inverse"
    )

# Historical overrun analysis
st.header("Historical Analysis")
overrun_days = filtered_data[filtered_data['Overrun']]['Date'].dt.date.tolist()
if overrun_days:
    st.warning(f"Found {len(overrun_days)} days with potential overrun scenarios")
    st.write("Recent overrun dates:", overrun_days[-5:])
else:
    st.success("No overrun scenarios detected in the selected date range")