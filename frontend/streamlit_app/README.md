# Powergrid Overrun Demo

A Streamlit dashboard for visualizing and predicting power grid overrun scenarios.

## Features

- Interactive data visualization
- Predictive analytics for overrun scenarios
- Time series analysis of power consumption
- Plotly-powered interactive charts

## Quick Start

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run powergrid_overrun_demo.py
```

## Deployment Options

### Streamlit Community Cloud

1. Push this code to a public GitHub repository
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy instantly!

### Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose Streamlit as the SDK
3. Push your code or use the web interface
4. Your app will be deployed automatically

## Requirements

See `requirements.txt` for the complete list of dependencies:

- streamlit
- pandas
- numpy
- scikit-learn
- plotly

## Data Format

The app expects input data with the following columns:

- Timestamp
- Power consumption values
- Other relevant metrics

## Contributing

Feel free to open issues or submit pull requests for improvements!
