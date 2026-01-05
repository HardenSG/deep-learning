import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from pathlib import Path
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelPredictor:
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"预测器初始化完成，设备: {self.device}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            predictions = predictions.cpu().numpy().flatten()
        
        return predictions
    
    def predict_single(self, X: np.ndarray) -> float:
        if len(X.shape) == 2:
            X = np.expand_dims(X, axis=0)
        
        prediction = self.predict(X)
        return float(prediction[0])
    
    def predict_with_confidence(self, X: np.ndarray, n_iterations: int = 10) -> Dict[str, float]:
        self.model.train()
        
        predictions = []
        for _ in range(n_iterations):
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                pred = self.model(X_tensor)
                predictions.append(pred.cpu().numpy())
        
        self.model.eval()
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        return {
            "prediction": float(mean_pred),
            "std": float(std_pred),
            "confidence_interval_lower": float(mean_pred - 1.96 * std_pred),
            "confidence_interval_upper": float(mean_pred + 1.96 * std_pred)
        }
    
    def load_model(self, model_path: str) -> None:
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        logger.info(f"模型加载自: {model_path}")
    
    def batch_predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        self.model.eval()
        
        predictions = []
        num_samples = len(X)
        
        for i in range(0, num_samples, batch_size):
            batch_X = X[i:i + batch_size]
            batch_pred = self.predict(batch_X)
            predictions.extend(batch_pred)
        
        return np.array(predictions)
