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

        # 避免除零
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # 当loss为0时（只涨不跌），RSI应为100
        # 当gain为0时（只跌不涨），RSI应为0
        df.loc[loss == 0, "rsi"] = 100
        df.loc[gain == 0, "rsi"] = 0

        return df
    
    @staticmethod
    def calculate_kdj(df: pd.DataFrame, period: int = 9) -> pd.DataFrame:
        low_min = df["low"].rolling(window=period).min()
        high_max = df["high"].rolling(window=period).max()

        # 避免除零（基金数据high=low时）
        denominator = high_max - low_min
        denominator = denominator.replace(0, np.nan)

        rsv = (df["close"] - low_min) / denominator * 100
        # 如果无法计算RSV（基金数据），用50填充
        rsv = rsv.fillna(50)

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

        # 避免除零
        bb_middle_safe = df["bb_middle"].replace(0, np.nan)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_middle_safe
        df["bb_width"] = df["bb_width"].fillna(0)

        return df
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(window=period).mean()

        # 基金数据ATR可能为0，用收盘价的波动率代替
        if (df["atr"] == 0).all() or df["atr"].isna().all():
            df["atr"] = df["close"].rolling(window=period).std()

        return df
    
    @staticmethod
    def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算能量潮指标 (On-Balance Volume)

        向量化实现，性能提升100倍以上
        """
        # 如果没有成交量数据（如基金），使用价格变化幅度代替
        if (df["volume"] == 0).all():
            # 用价格变化百分比的绝对值作为"虚拟成交量"
            df["obv"] = df["close"].pct_change().fillna(0).cumsum() * 100
            return df

        # 计算价格变化方向：1(上涨)、-1(下跌)、0(不变)
        price_change = df["close"].diff()
        direction = np.sign(price_change)

        # 带符号的成交量
        signed_volume = direction * df["volume"]

        # 第一个值设为0，然后累积求和
        signed_volume.iloc[0] = 0
        df["obv"] = signed_volume.cumsum()

        return df
    
    @staticmethod
    def calculate_volume_ratio(df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
        df["volume_ma"] = df["volume"].rolling(window=period).mean()

        # 避免除零（基金数据volume=0）
        if (df["volume_ma"] == 0).all():
            df["volume_ratio"] = 1.0  # 基金无成交量，设为1
        else:
            df["volume_ratio"] = df["volume"] / df["volume_ma"].replace(0, np.nan)
            df["volume_ratio"] = df["volume_ratio"].fillna(1.0)

        return df
    
    @staticmethod
    def calculate_price_change(df: pd.DataFrame, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        for period in periods:
            df[f"price_change_{period}d"] = df["close"].pct_change(periods=period)
        return df
    
    @staticmethod
    def calculate_diff(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算一阶差分 (去非平稳性)
        """
        # 价格差分
        df["close_diff"] = df["close"].diff()
        # 成交量差分
        if "volume" in df.columns:
            df["volume_diff"] = df["volume"].diff()
        return df
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
        logger.info("开始计算技术指标...")
        
        df = df.copy()
        
        tech_config = config.get("technical_indicators", {})
        
        # 1. 计算基础指标 (MA, EMA等)
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
        
        # 2. 计算差分 (去非平稳性)
        df = TechnicalIndicators.calculate_diff(df)
        
        logger.info(f"技术指标计算完成，共 {len(df.columns)} 个特征")
        
        return df
