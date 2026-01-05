import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
from pathlib import Path
from .technical_indicators import TechnicalIndicators
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FeatureBuilder:
    def __init__(self, config: dict):
        self.config = config
        self.window_size = config.get("window_size", 60)
        self.normalization = config.get("normalization", "minmax")
        self.scaler = None
        self.feature_columns = None
        
        logger.info(f"特征构建器初始化: window_size={self.window_size}, normalization={self.normalization}")
    
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = TechnicalIndicators.calculate_all_indicators(df, self.config)
        
        df = df.dropna()
        
        logger.info(f"特征构建完成: {len(df)} 条记录, {len(df.columns)} 个特征")
        return df
    
    def prepare_training_data(
        self, 
        df: pd.DataFrame,
        target_column: str = "close",
        prediction_horizon: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        df = self.build_features(df)
        
        exclude_columns = ["trade_date", "stock_code", "id", "created_at"]
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        self.feature_columns = feature_columns
        
        df_features = df[feature_columns].values
        
        if self.scaler is None:
            if self.normalization == "minmax":
                self.scaler = MinMaxScaler()
            else:
                self.scaler = StandardScaler()
            
            df_features = self.scaler.fit_transform(df_features)
        else:
            df_features = self.scaler.transform(df_features)
        
        X, y = [], []
        
        for i in range(len(df_features) - self.window_size - prediction_horizon + 1):
            X.append(df_features[i:i + self.window_size])
            
            future_price = df[target_column].iloc[i + self.window_size + prediction_horizon - 1]
            current_price = df[target_column].iloc[i + self.window_size - 1]
            
            target_return = (future_price - current_price) / current_price
            y.append(target_return)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"训练数据准备完成: X shape={X.shape}, y shape={y.shape}")
        
        return X, y, feature_columns
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        total_samples = len(X)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        logger.info(f"数据集划分: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def save_scaler(self, save_path: str) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "feature_columns": self.feature_columns,
                "window_size": self.window_size,
                "normalization": self.normalization
            }, f)
        
        logger.info(f"Scaler保存至: {save_path}")
    
    def load_scaler(self, load_path: str) -> None:
        with open(load_path, "rb") as f:
            data = pickle.load(f)
            self.scaler = data["scaler"]
            self.feature_columns = data["feature_columns"]
            self.window_size = data["window_size"]
            self.normalization = data["normalization"]
        
        logger.info(f"Scaler加载自: {load_path}")
    
    def prepare_prediction_data(self, df: pd.DataFrame) -> np.ndarray:
        df = self.build_features(df)
        
        if df.empty:
            raise ValueError("数据为空")
        
        if self.feature_columns is None:
            raise ValueError("特征列未定义，请先训练模型或加载scaler")
        
        df_features = df[self.feature_columns].values
        
        if self.scaler is None:
            raise ValueError("Scaler未初始化，请先训练模型或加载scaler")
        
        df_features = self.scaler.transform(df_features)
        
        if len(df_features) < self.window_size:
            raise ValueError(f"数据不足，需要至少 {self.window_size} 条记录")
        
        X = df_features[-self.window_size:]
        X = np.expand_dims(X, axis=0)
        
        return X
