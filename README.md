# Power Grid Construction Cost Predictor

A machine learning-powered dashboard for predicting construction costs and timelines for power grid projects.

## Features

- Cost and timeline prediction based on historical data
- Interactive data visualization
- Support for custom data upload (CSV, Excel)
- Detailed project risk assessment
- Cost breakdown analysis

## Demo

![Dashboard Demo](docs/images/dashboard_demo.png)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/power-grid-predictor.git
cd power-grid-predictor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run frontend/streamlit_app/powergrid_construction_predictor.py
```

2. Open your browser and navigate to http://localhost:8501

3. Either:
   - Use the sample data provided
   - Upload your own data (CSV/Excel)
   - Input custom project parameters

## Data Format

Required columns for custom data:

| Column Name | Data Type | Description |
|------------|-----------|-------------|
| Project_Start_Date | datetime64[ns] | Project start date |
| Line_Length_KM | float64 | Length of power line in kilometers |
| Voltage_Level_KV | int64 | Voltage level in kilovolts |
| Terrain_Difficulty | object | Low/Medium/High |
| Number_of_Towers | int64 | Number of transmission towers |
| Right_of_Way_Cost | float64 | Right of way acquisition cost |
| Material_Cost | float64 | Total material cost |
| Total_Cost | float64 | Total project cost |
| Project_Duration | int64 | Project duration in months |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - feel free to use this project for your own purposes.

## Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/power-grid-predictor](https://github.com/yourusername/power-grid-predictor)