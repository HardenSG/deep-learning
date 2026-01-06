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
        data_len = len(df)
        logger.info(f"开始构建特征，原始数据: {data_len} 条")
        
        adjusted_config = self._adjust_config_for_data_size(df, self.config)
        
        df = TechnicalIndicators.calculate_all_indicators(df, adjusted_config)
        
        before_dropna = len(df)
        df = df.dropna()
        after_dropna = len(df)
        
        dropped_rows = before_dropna - after_dropna
        if dropped_rows > 0:
            logger.warning(f"dropna() 删除了 {dropped_rows} 行数据 ({dropped_rows/before_dropna*100:.1f}%)")
        
        if len(df) == 0:
            raise ValueError(
                f"特征构建后数据为空！原始数据{data_len}条，计算指标后{before_dropna}条，dropna后{after_dropna}条。"
                f"可能原因：数据量太少，无法计算所需的技术指标。建议至少采集100条以上的历史数据。"
            )
        
        logger.info(f"特征构建完成: {len(df)} 条记录, {len(df.columns)} 个特征")
        return df
    
    def _adjust_config_for_data_size(self, df: pd.DataFrame, config: dict) -> dict:
        """
        根据数据量动态调整技术指标参数
        
        Args:
            df: 原始数据
            config: 原始配置
            
        Returns:
            调整后的配置
        """
        data_len = len(df)
        adjusted_config = config.copy()
        
        if "technical_indicators" not in adjusted_config:
            adjusted_config["technical_indicators"] = {}
        
        tech_config = adjusted_config["technical_indicators"]
        
        if data_len < 150:
            logger.warning(f"数据量较少({data_len}条)，自动调整技术指标参数以避免数据全部被删除")
            
            max_period = max(5, min(10, data_len // 4))
            
            original_ma = tech_config.get("ma_periods", [5, 10, 20, 30, 60])
            adjusted_ma = [p for p in original_ma if p <= max_period]
            if not adjusted_ma:
                adjusted_ma = [5]
            tech_config["ma_periods"] = adjusted_ma
            logger.info(f"MA周期: {original_ma} -> {adjusted_ma}")
            
            original_ema = tech_config.get("ema_periods", [12, 26])
            adjusted_ema = [p for p in original_ema if p <= max_period]
            if not adjusted_ema:
                adjusted_ema = [5]
            tech_config["ema_periods"] = adjusted_ema
            logger.info(f"EMA周期: {original_ema} -> {adjusted_ema}")
            
            original_bollinger = tech_config.get("bollinger_period", 20)
            adjusted_bollinger = min(original_bollinger, max_period)
            if adjusted_bollinger != original_bollinger:
                tech_config["bollinger_period"] = adjusted_bollinger
                logger.info(f"布林带周期: {original_bollinger} -> {adjusted_bollinger}")
            
            original_rsi = tech_config.get("rsi_period", 14)
            adjusted_rsi = min(original_rsi, max_period)
            if adjusted_rsi != original_rsi:
                tech_config["rsi_period"] = adjusted_rsi
                logger.info(f"RSI周期: {original_rsi} -> {adjusted_rsi}")
            
            original_atr = tech_config.get("atr_period", 14)
            adjusted_atr = min(original_atr, max_period)
            if adjusted_atr != original_atr:
                tech_config["atr_period"] = adjusted_atr
                logger.info(f"ATR周期: {original_atr} -> {adjusted_atr}")
            
            original_kdj = tech_config.get("kdj_period", 9)
            adjusted_kdj = min(original_kdj, max_period)
            if adjusted_kdj != original_kdj:
                tech_config["kdj_period"] = adjusted_kdj
                logger.info(f"KDJ周期: {original_kdj} -> {adjusted_kdj}")
            
            original_macd_fast = tech_config.get("macd_fast", 12)
            adjusted_macd_fast = min(original_macd_fast, max_period)
            if adjusted_macd_fast != original_macd_fast:
                tech_config["macd_fast"] = adjusted_macd_fast
                logger.info(f"MACD快线: {original_macd_fast} -> {adjusted_macd_fast}")
            
            original_macd_slow = tech_config.get("macd_slow", 26)
            adjusted_macd_slow = min(original_macd_slow, max_period)
            if adjusted_macd_slow != original_macd_slow:
                tech_config["macd_slow"] = adjusted_macd_slow
                logger.info(f"MACD慢线: {original_macd_slow} -> {adjusted_macd_slow}")
            
            original_macd_signal = tech_config.get("macd_signal", 9)
            adjusted_macd_signal = min(original_macd_signal, max_period)
            if adjusted_macd_signal != original_macd_signal:
                tech_config["macd_signal"] = adjusted_macd_signal
                logger.info(f"MACD信号线: {original_macd_signal} -> {adjusted_macd_signal}")
        
        return adjusted_config
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_column: str = "close",
        prediction_horizon: int = 5,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备训练数据

        Args:
            df: 原始数据
            target_column: 目标列名
            prediction_horizon: 预测时间跨度
            fit_scaler: 是否在当前数据上fit scaler（仅训练集应该为True）

        Returns:
            X, y, feature_columns
        """
        df = self.build_features(df)

        exclude_columns = ["trade_date", "stock_code", "id", "created_at"]
        feature_columns = [col for col in df.columns if col not in exclude_columns]

        self.feature_columns = feature_columns

        df_features = df[feature_columns].values

        # 关键修复：只在训练集上fit scaler
        if fit_scaler:
            if self.normalization == "minmax":
                self.scaler = MinMaxScaler()
            else:
                self.scaler = StandardScaler()

            df_features = self.scaler.fit_transform(df_features)
            logger.info(f"Scaler已在当前数据集上fit（应仅用于训练集）")
        else:
            if self.scaler is None:
                raise ValueError("Scaler未初始化，请先在训练集上fit")
            df_features = self.scaler.transform(df_features)
            logger.info(f"使用已有scaler进行transform（用于验证/测试集）")

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
        """
        时序数据划分（已弃用，建议使用 prepare_train_val_test_data）

        该方法假设数据已经在整个数据集上进行了归一化，可能存在数据泄露。
        请使用 prepare_train_val_test_data 方法以正确的方式处理数据。
        """
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

    def prepare_train_val_test_data(
        self,
        df: pd.DataFrame,
        target_column: str = "close",
        prediction_horizon: int = 5,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        正确的时序数据划分方式（避免数据泄露）

        先按时间划分原始数据，再分别进行特征工程和归一化

        Args:
            df: 原始数据（按时间排序）
            target_column: 目标列名
            prediction_horizon: 预测时间跨度
            train_ratio: 训练集比例
            val_ratio: 验证集比例

        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test, feature_columns
        """
        total_len = len(df)
        
        original_window_size = self.window_size
        adjusted_window_size = min(self.window_size, max(10, total_len // 5))
        
        if adjusted_window_size != original_window_size:
            logger.warning(f"数据量较少({total_len}条)，window_size从{original_window_size}调整为{adjusted_window_size}")
            self.window_size = adjusted_window_size
        
        required_min_len = self.window_size + prediction_horizon

        if total_len < required_min_len:
            raise ValueError(f"数据长度不足，至少需要 {required_min_len} 条记录，当前只有 {total_len} 条")

        train_end = int(total_len * train_ratio)
        val_end = int(total_len * (train_ratio + val_ratio))

        # 确保每个数据集都有足够的数据
        train_end = max(train_end, required_min_len)
        val_end = max(val_end, train_end + required_min_len)

        # 按时间分割原始数据
        df_train = df.iloc[:train_end].copy()
        df_val = df.iloc[:val_end].copy()  # 验证集需要包含训练集数据以构建窗口
        df_test = df.copy()  # 测试集需要包含所有数据以构建窗口

        logger.info(f"原始数据划分: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

        # 在训练集上fit scaler
        X_train, y_train, feature_columns = self.prepare_training_data(
            df_train,
            target_column=target_column,
            prediction_horizon=prediction_horizon,
            fit_scaler=True
        )

        # 在验证集上使用训练好的scaler
        X_val_full, y_val_full, _ = self.prepare_training_data(
            df_val,
            target_column=target_column,
            prediction_horizon=prediction_horizon,
            fit_scaler=False
        )

        # 在测试集上使用训练好的scaler
        X_test_full, y_test_full, _ = self.prepare_training_data(
            df_test,
            target_column=target_column,
            prediction_horizon=prediction_horizon,
            fit_scaler=False
        )

        # 从完整数据中提取对应的验证集和测试集部分
        train_samples = len(X_train)
        val_samples_start = train_samples
        val_samples_end = int(len(df_val) * (train_ratio + val_ratio) / train_ratio) - self.window_size - prediction_horizon + 1

        X_val = X_val_full[val_samples_start:] if val_samples_start < len(X_val_full) else X_val_full[-1:]
        y_val = y_val_full[val_samples_start:] if val_samples_start < len(y_val_full) else y_val_full[-1:]

        test_samples_start = val_end - self.window_size
        X_test = X_test_full[test_samples_start:] if test_samples_start < len(X_test_full) else X_test_full[-1:]
        y_test = y_test_full[test_samples_start:] if test_samples_start < len(y_test_full) else y_test_full[-1:]

        logger.info(f"最终数据集: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        logger.info("✅ 数据泄露已修复：scaler仅在训练集上fit")

        return X_train, y_train, X_val, y_val, X_test, y_test, feature_columns
    
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
