import pandas as pd
import numpy as np
from typing import List, Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TechnicalIndicators:
    @staticmethod
    def calculate_ma(df: pd.DataFrame, periods: List[int] = [5, 10, 20, 30, 60]) -> pd.DataFrame:
        for period in periods:
            df[f"ma_{period}"] = df["close"].rolling(window=period).mean()
        return df
    
    @staticmethod
    def calculate_ema(df: pd.DataFrame, periods: List[int] = [12, 26]) -> pd.DataFrame:
        for period in periods:
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
        return df
    
    @staticmethod
    def calculate_macd(
        df: pd.DataFrame, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> pd.DataFrame:
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        return df
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        delta = df["close"].diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        return df
    
    @staticmethod
    def calculate_kdj(df: pd.DataFrame, period: int = 9) -> pd.DataFrame:
        low_min = df["low"].rolling(window=period).min()
        high_max = df["high"].rolling(window=period).max()
        
        rsv = (df["close"] - low_min) / (high_max - low_min) * 100
        
        df["kdj_k"] = rsv.ewm(com=2, adjust=False).mean()
        df["kdj_d"] = df["kdj_k"].ewm(com=2, adjust=False).mean()
        df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]
        
        return df
    
    @staticmethod
    def calculate_bollinger_bands(
        df: pd.DataFrame, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        df["bb_middle"] = df["close"].rolling(window=period).mean()
        std = df["close"].rolling(window=period).std()
        
        df["bb_upper"] = df["bb_middle"] + (std * std_dev)
        df["bb_lower"] = df["bb_middle"] - (std * std_dev)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        
        return df
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(window=period).mean()
        
        return df
    
    @staticmethod
    def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
        obv = [0]
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i-1]:
                obv.append(obv[-1] + df["volume"].iloc[i])
            elif df["close"].iloc[i] < df["close"].iloc[i-1]:
                obv.append(obv[-1] - df["volume"].iloc[i])
            else:
                obv.append(obv[-1])
        
        df["obv"] = obv
        return df
    
    @staticmethod
    def calculate_volume_ratio(df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
        df["volume_ma"] = df["volume"].rolling(window=period).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]
        return df
    
    @staticmethod
    def calculate_price_change(df: pd.DataFrame, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        for period in periods:
            df[f"price_change_{period}d"] = df["close"].pct_change(periods=period)
        return df
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
        logger.info("开始计算技术指标...")
        
        df = df.copy()
        
        tech_config = config.get("technical_indicators", {})
        
        df = TechnicalIndicators.calculate_ma(df, tech_config.get("ma_periods", [5, 10, 20, 30, 60]))
        df = TechnicalIndicators.calculate_ema(df, tech_config.get("ema_periods", [12, 26]))
        df = TechnicalIndicators.calculate_macd(
            df,
            tech_config.get("macd_fast", 12),
            tech_config.get("macd_slow", 26),
            tech_config.get("macd_signal", 9)
        )
        df = TechnicalIndicators.calculate_rsi(df, tech_config.get("rsi_period", 14))
        df = TechnicalIndicators.calculate_kdj(df, tech_config.get("kdj_period", 9))
        df = TechnicalIndicators.calculate_bollinger_bands(
            df,
            tech_config.get("bollinger_period", 20),
            tech_config.get("bollinger_std", 2)
        )
        df = TechnicalIndicators.calculate_atr(df, tech_config.get("atr_period", 14))
        df = TechnicalIndicators.calculate_obv(df)
        df = TechnicalIndicators.calculate_volume_ratio(df)
        df = TechnicalIndicators.calculate_price_change(df)
        
        logger.info(f"技术指标计算完成，共 {len(df.columns)} 个特征")
        
        return df
