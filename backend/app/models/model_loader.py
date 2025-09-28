import joblib
import os

class ModelLoader:
    def __init__(self):
        self.model = None
        self.model_path = os.getenv("MODEL_PATH", "model.joblib")
    
    def load_model(self):
        """Load the pre-trained model"""
        if not self.model:
            try:
                self.model = joblib.load(self.model_path)
            except Exception as e:
                raise Exception(f"Failed to load model: {str(e)}")
        return self.model

    def get_model(self):
        """Get the loaded model instance"""
        if not self.model:
            self.load_model()
        return self.model