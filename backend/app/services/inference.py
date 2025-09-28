from app.models.model_loader import ModelLoader
from app.utils.preprocess import preprocess_input
import numpy as np

class InferenceService:
    def __init__(self):
        self.model_loader = ModelLoader()
        
    def predict(self, features: list[float]) -> dict:
        """
        Make predictions using the loaded model
        
        Args:
            features: List of input features
            
        Returns:
            dict: Prediction results with confidence
        """
        # Preprocess input
        processed_features = preprocess_input(features)
        
        # Get model and make prediction
        model = self.model_loader.get_model()
        prediction = model.predict([processed_features])[0]
        
        # Get prediction probability/confidence if available
        confidence = 0.0
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba([processed_features])[0]
            confidence = float(np.max(proba))
            
        return {
            "prediction": float(prediction),
            "confidence": confidence
        }