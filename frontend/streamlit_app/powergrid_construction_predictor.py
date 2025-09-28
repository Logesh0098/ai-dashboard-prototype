import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import io
import xlrd
import openpyxl

# Page config
st.set_page_config(
    page_title="Power Grid Construction Predictor",
    page_icon="🏗️",
    layout="wide"
)

# Title and description
st.title("🏗️ Power Grid Construction Project Predictor")
st.markdown("""
This dashboard helps predict construction costs and timelines for power grid projects using historical data and machine learning.
Upload your historical project data or use our sample dataset for predictions.
""")

# Data loading functions
def load_csv(file):
    return pd.read_csv(file)

def load_excel(file):
    return pd.read_excel(file)

def validate_data(df):
    """Validate if the uploaded data has the required columns and correct data types"""
    required_columns = {
        'Project_Start_Date': 'datetime64[ns]',
        'Line_Length_KM': 'float64',
        'Voltage_Level_KV': 'int64',
        'Terrain_Difficulty': 'object',
        'Number_of_Towers': 'int64',
        'Right_of_Way_Cost': 'float64',
        'Material_Cost': 'float64',
        'Total_Cost': 'float64',
        'Project_Duration': 'int64'
    }
    
    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.write("Required columns and their data types:")
        st.code("\n".join([f"{col}: {dtype}" for col, dtype in required_columns.items()]))
        return False
    
    # Check data types and try to convert if needed
    type_issues = []
    for col, expected_type in required_columns.items():
        try:
            if col == 'Project_Start_Date':
                df[col] = pd.to_datetime(df[col])
            elif expected_type == 'int64':
                df[col] = df[col].astype('int64')
            elif expected_type == 'float64':
                df[col] = df[col].astype('float64')
            elif col == 'Terrain_Difficulty':
                if not all(df[col].isin(['Low', 'Medium', 'High'])):
                    type_issues.append(f"{col} must contain only 'Low', 'Medium', or 'High' values")
        except Exception as e:
            type_issues.append(f"Could not convert {col} to {expected_type}: {str(e)}")
    
    if type_issues:
        st.error("Data type issues found:")
        for issue in type_issues:
            st.write(f"- {issue}")
        return False
    
    return True

# Generate synthetic project data
@st.cache_data
def generate_synthetic_data():
    np.random.seed(42)
    n_samples = 100
    
    # Project parameters that affect cost and time
    data = pd.DataFrame({
        'Project_Start_Date': pd.date_range(start='2020-01-01', periods=n_samples, freq='ME'),  # Changed M to ME
        'Line_Length_KM': np.random.uniform(10, 100, n_samples).astype('float64'),
        'Voltage_Level_KV': np.random.choice([33, 66, 110, 220, 400], n_samples).astype('int64'),
        'Terrain_Difficulty': np.random.choice(['Low', 'Medium', 'High'], n_samples),  # Keep as object type
        'Number_of_Towers': np.random.randint(20, 200, n_samples).astype('int64'),
        'Right_of_Way_Cost': np.random.uniform(1000000, 5000000, n_samples).astype('float64'),
        'Material_Cost': np.random.uniform(5000000, 20000000, n_samples).astype('float64')
    })
    
    # Calculate terrain multiplier after DataFrame creation
    terrain_multiplier = {'Low': 1.0, 'Medium': 1.3, 'High': 1.6}
    data['Terrain_Multiplier'] = data['Terrain_Difficulty'].map(terrain_multiplier).astype('float64')
    
    # Calculate synthetic project costs
    base_cost_per_km = 2000000  # Base cost per kilometer
    
    # Total project cost calculation
    data['Total_Cost'] = (
        data['Line_Length_KM'] * base_cost_per_km * data['Terrain_Multiplier'] +
        data['Right_of_Way_Cost'] +
        data['Material_Cost'] +
        data['Number_of_Towers'] * 500000  # Cost per tower
    )
    
    # Project duration calculation (in months)
    data['Project_Duration'] = (
        3 + # Base planning time
        data['Line_Length_KM'] * 0.2 + # Time per km
        data['Number_of_Towers'] * 0.1 + # Time per tower
        np.random.normal(0, 2, n_samples) # Random variations
    ).astype(int)
    
    return data

# Load or generate data
data = generate_synthetic_data()

# Sidebar - Data Upload and Parameters
st.sidebar.header("Data Source")
data_source = st.sidebar.radio("Select Data Source", ["Upload Data", "Use Sample Data"])

if data_source == "Upload Data":
    st.sidebar.header("Upload Project Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your project data file",
        type=["csv", "xlsx", "xls"],
        help="Upload a CSV or Excel file with historical project data"
    )
    
    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split(".")[-1]
            if file_extension in ["xlsx", "xls"]:
                data = load_excel(uploaded_file)
            else:
                data = load_csv(uploaded_file)
            
            if validate_data(data):
                st.sidebar.success("Data loaded successfully!")
                st.sidebar.write(f"Number of projects: {len(data)}")
                
                # Display data summary
                with st.expander("View Data Summary"):
                    st.write("Data Preview:")
                    st.dataframe(data.head())
                    
                    st.write("Statistical Summary:")
                    st.dataframe(data.describe())
                    
                    st.write("Data Types:")
                    st.dataframe(pd.DataFrame(data.dtypes, columns=['Data Type']))
            else:
                data = generate_synthetic_data()
                st.sidebar.warning("Using sample data instead.")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")
            data = generate_synthetic_data()
            st.sidebar.warning("Using sample data instead.")
    else:
        data = generate_synthetic_data()
        st.sidebar.info("Upload a file or continue with sample data.")
else:
    data = generate_synthetic_data()
    st.sidebar.info("Using sample data for demonstration.")

# Download sample template
if st.sidebar.button("Download Sample Template"):
    sample_df = generate_synthetic_data().head()
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        sample_df.to_excel(writer, index=False, sheet_name='Template')
    
    st.sidebar.download_button(
        label="Download Excel Template",
        data=buffer.getvalue(),
        file_name="project_data_template.xlsx",
        mime="application/vnd.ms-excel"
    )

st.sidebar.header("Project Parameters")

# Input parameters for prediction
with st.sidebar:
    st.subheader("Enter Project Details")
    line_length = st.number_input("Line Length (KM)", min_value=10.0, max_value=100.0, value=50.0)
    voltage = st.selectbox("Voltage Level (KV)", options=[33, 66, 110, 220, 400])
    terrain = st.selectbox("Terrain Difficulty", options=['Low', 'Medium', 'High'])
    towers = st.number_input("Number of Towers", min_value=20, max_value=200, value=100)
    row_cost = st.number_input("Right of Way Cost (₹)", min_value=1000000, max_value=5000000, value=2000000)
    material_cost = st.number_input("Material Cost (₹)", min_value=5000000, max_value=20000000, value=10000000)

# Main dashboard
col1, col2 = st.columns(2)

with col1:
    st.subheader("Cost Analysis")
    fig_cost = px.scatter(data, 
                         x='Line_Length_KM', 
                         y='Total_Cost',
                         color='Terrain_Difficulty',
                         size='Number_of_Towers',
                         hover_data=['Voltage_Level_KV'],
                         title='Project Costs vs Line Length')
    st.plotly_chart(fig_cost, use_container_width=True)

with col2:
    st.subheader("Duration Analysis")
    fig_duration = px.scatter(data,
                            x='Number_of_Towers',
                            y='Project_Duration',
                            color='Terrain_Difficulty',
                            size='Line_Length_KM',
                            hover_data=['Voltage_Level_KV'],
                            title='Project Duration vs Number of Towers')
    st.plotly_chart(fig_duration, use_container_width=True)

# Prepare data for prediction
X_cost = data[['Line_Length_KM', 'Number_of_Towers', 'Right_of_Way_Cost', 'Material_Cost']]
y_cost = data['Total_Cost']
X_duration = data[['Line_Length_KM', 'Number_of_Towers']]
y_duration = data['Project_Duration']

# Train models
cost_model = RandomForestRegressor(n_estimators=100, random_state=42)
duration_model = RandomForestRegressor(n_estimators=100, random_state=42)

cost_model.fit(X_cost, y_cost)
duration_model.fit(X_duration, y_duration)

# Make predictions
if st.button("Calculate Predictions"):
    # Prepare input data
    cost_input = np.array([[line_length, towers, row_cost, material_cost]])
    duration_input = np.array([[line_length, towers]])
    
    # Get predictions
    predicted_cost = cost_model.predict(cost_input)[0]
    predicted_duration = duration_model.predict(duration_input)[0]
    
    # Display predictions
    st.header("Project Predictions")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Estimated Project Cost",
            value=f"₹{predicted_cost:,.2f}",
            delta=f"±{predicted_cost * 0.1:,.2f}"
        )
        
    with col2:
        st.metric(
            label="Estimated Duration",
            value=f"{predicted_duration:.0f} months",
            delta=f"±{max(2, predicted_duration * 0.1):.0f} months"
        )
    
    # Additional insights
    st.subheader("Project Insights")
    
    # Cost breakdown
    cost_breakdown = {
        'Line Construction': line_length * 2000000,
        'Tower Installation': towers * 500000,
        'Right of Way': row_cost,
        'Materials': material_cost
    }
    
    fig_breakdown = px.pie(
        values=list(cost_breakdown.values()),
        names=list(cost_breakdown.keys()),
        title='Estimated Cost Breakdown'
    )
    st.plotly_chart(fig_breakdown, use_container_width=True)
    
    # Risk factors
    st.subheader("Risk Factors")
    risk_level = "Medium"
    if terrain == 'High' and line_length > 70:
        risk_level = "High"
    elif terrain == 'Low' and line_length < 30:
        risk_level = "Low"
        
    st.warning(f"""
    Project Risk Level: {risk_level}
    - Terrain Difficulty: {terrain}
    - Project Scale: {line_length:.1f} KM
    - Infrastructure: {towers} towers
    """)