# AI Dashboard Prototype

This is a full-stack AI dashboard application with a FastAPI backend for serving ML models and a React frontend for visualization.

## Project Structure

```
ai-dashboard-prototype/
│
├── backend/                      # FastAPI backend
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── models/              # ML model loading
│   │   ├── routes/              # API endpoints
│   │   ├── services/            # Business logic
│   │   └── utils/               # Helper functions
│   └── requirements.txt         # Python dependencies
│
└── frontend/                    # React frontend
    └── react_app/
        ├── src/
        │   ├── components/      # React components
        │   ├── App.js
        │   └── index.js
        ├── package.json
        └── vite.config.js
```

## Setup Instructions

### Backend Setup

1. Create a Python virtual environment:

   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. Install Node.js dependencies:

   ```bash
   cd frontend/react_app
   npm install
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

The frontend will be available at `http://localhost:5173`

## API Endpoints

- `GET /`: Welcome message
- `POST /predict`: Make predictions
  - Request body: `{ "features": [float, float, ...] }`
  - Response: `{ "prediction": float, "confidence": float }`

## Frontend Features

- Input form for feature submission
- Real-time prediction display
- Confidence score visualization
- Error handling and user feedback

## Technologies Used

- Backend:
  - FastAPI
  - Scikit-learn
  - NumPy
  - Pydantic
- Frontend:
  - React
  - Vite
  - Material-UI
  - Chart.js
  - Axios
