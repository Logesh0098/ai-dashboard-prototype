import React, { useState } from 'react';
import { Container, Paper, Typography, TextField, Button, Box } from '@mui/material';
import { Line } from 'react-chartjs-2';

const Dashboard = () => {
  const [inputFeatures, setInputFeatures] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState('');

  const handlePredict = async () => {
    try {
      const features = inputFeatures.split(',').map(Number);
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ features }),
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const result = await response.json();
      setPrediction(result);
      setError('');
    } catch (err) {
      setError('Error making prediction: ' + err.message);
      setPrediction(null);
    }
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          AI Dashboard
        </Typography>
        
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Make Prediction
          </Typography>
          <TextField
            fullWidth
            label="Input Features (comma-separated)"
            value={inputFeatures}
            onChange={(e) => setInputFeatures(e.target.value)}
            margin="normal"
          />
          <Button
            variant="contained"
            color="primary"
            onClick={handlePredict}
            sx={{ mt: 2 }}
          >
            Predict
          </Button>
        </Paper>

        {error && (
          <Paper sx={{ p: 3, mb: 3, bgcolor: 'error.light' }}>
            <Typography color="error">{error}</Typography>
          </Paper>
        )}

        {prediction && (
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Prediction Results
            </Typography>
            <Typography>
              Prediction: {prediction.prediction}
            </Typography>
            <Typography>
              Confidence: {(prediction.confidence * 100).toFixed(2)}%
            </Typography>
          </Paper>
        )}
      </Box>
    </Container>
  );
};

export default Dashboard;