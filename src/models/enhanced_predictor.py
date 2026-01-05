import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from pathlib import Path
from .lstm_model import LSTMModel
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EnhancedPredictor:
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"增强预测器初始化完成，设备: {self.device}")
    
    def load_model(self, model_path: str) -> None:
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        logger.info(f"模型加载自: {model_path}")
    
    def predict_with_probability(self, X: np.ndarray) -> Dict:
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            raw_prediction = self.model(X_tensor)
            raw_value = raw_prediction.cpu().numpy().flatten()[0]
            
            probability = self._calculate_probability(raw_value)
            
            confidence_level = self._get_confidence_level(probability)
            
            recommendation = self._generate_recommendation(
                probability, 
                raw_value, 
                confidence_level
            )
            
            return {
                "predicted_return": float(raw_value),
                "up_probability": float(probability),
                "down_probability": float(1 - probability),
                "direction": "上涨" if raw_value > 0 else "下跌",
                "confidence_level": confidence_level,
                "recommendation": recommendation
            }
    
    def _calculate_probability(self, return_value: float) -> float:
        probability = 1 / (1 + np.exp(-return_value * 100))
        
        return probability
    
    def _get_confidence_level(self, probability: float) -> str:
        if probability >= 0.7 or probability <= 0.3:
            return "高"
        elif probability >= 0.6 or probability <= 0.4:
            return "中"
        else:
            return "低"
    
    def _generate_recommendation(
        self, 
        probability: float, 
        return_value: float,
        confidence_level: str
    ) -> Dict:
        if probability >= 0.65 and return_value > 0.01:
            action = "强烈建议买入"
            reason = f"上涨概率{probability*100:.1f}%，预期收益{return_value*100:.2f}%"
            risk_level = "低" if confidence_level == "高" else "中"
        elif probability >= 0.55 and return_value > 0.005:
            action = "建议买入"
            reason = f"上涨概率{probability*100:.1f}%，预期收益{return_value*100:.2f}%"
            risk_level = "中"
        elif probability >= 0.45 and probability <= 0.55:
            action = "观望"
            reason = f"涨跌概率接近，市场方向不明确"
            risk_level = "高"
        elif probability <= 0.45 and return_value < -0.005:
            action = "不建议买入"
            reason = f"下跌概率{(1-probability)*100:.1f}%，预期下跌{abs(return_value)*100:.2f}%"
            risk_level = "中"
        elif probability <= 0.35 and return_value < -0.01:
            action = "建议卖出/观望"
            reason = f"下跌概率{(1-probability)*100:.1f}%，预期下跌{abs(return_value)*100:.2f}%"
            risk_level = "低" if confidence_level == "高" else "中"
        else:
            action = "观望"
            reason = "市场信号不明确，建议等待更好的时机"
            risk_level = "中"
        
        return {
            "action": action,
            "reason": reason,
            "risk_level": risk_level,
            "confidence": confidence_level
        }
    
    def predict_with_monte_carlo(
        self, 
        X: np.ndarray, 
        n_simulations: int = 100
    ) -> Dict:
        self.model.train()
        
        predictions = []
        for _ in range(n_simulations):
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                pred = self.model(X_tensor)
                predictions.append(pred.cpu().numpy().flatten()[0])
        
        self.model.eval()
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        up_count = np.sum(predictions > 0)
        up_probability = up_count / n_simulations
        
        percentile_5 = np.percentile(predictions, 5)
        percentile_95 = np.percentile(predictions, 95)
        
        return {
            "mean_return": float(mean_pred),
            "std_return": float(std_pred),
            "up_probability": float(up_probability),
            "down_probability": float(1 - up_probability),
            "confidence_interval_5": float(percentile_5),
            "confidence_interval_95": float(percentile_95),
            "direction": "上涨" if mean_pred > 0 else "下跌",
            "simulations": n_simulations
        }
    
    def get_detailed_analysis(
        self, 
        X: np.ndarray,
        current_price: float,
        use_monte_carlo: bool = False
    ) -> Dict:
        if use_monte_carlo:
            basic_pred = self.predict_with_probability(X)
            mc_pred = self.predict_with_monte_carlo(X, n_simulations=100)
            
            predicted_return = mc_pred["mean_return"]
            up_probability = mc_pred["up_probability"]
        else:
            basic_pred = self.predict_with_probability(X)
            predicted_return = basic_pred["predicted_return"]
            up_probability = basic_pred["up_probability"]
        
        predicted_price = current_price * (1 + predicted_return)
        price_change = predicted_price - current_price
        
        result = {
            "current_price": float(current_price),
            "predicted_price": float(predicted_price),
            "price_change": float(price_change),
            "price_change_pct": float(predicted_return * 100),
            "up_probability_pct": float(up_probability * 100),
            "down_probability_pct": float((1 - up_probability) * 100),
            "direction": "上涨" if predicted_return > 0 else "下跌",
            "confidence_level": basic_pred["confidence_level"],
            "recommendation": basic_pred["recommendation"]
        }
        
        if use_monte_carlo:
            result["monte_carlo"] = {
                "mean_return": mc_pred["mean_return"],
                "std_return": mc_pred["std_return"],
                "confidence_interval_5_pct": mc_pred["confidence_interval_5"] * 100,
                "confidence_interval_95_pct": mc_pred["confidence_interval_95"] * 100,
                "simulations": mc_pred["simulations"]
            }
        
        return result
