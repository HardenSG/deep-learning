import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
from pathlib import Path
from .technical_indicators import TechnicalIndicators
from ..utils.logger import get_logger

logger = get_logger(__name__)


# 多因子相关的特征列定义
FUND_FLOW_FEATURES = [
    'main_net_inflow', 'main_net_inflow_pct',
    'super_large_net_inflow', 'large_net_inflow',
    'medium_net_inflow', 'small_net_inflow'
]

NORTH_FLOW_FEATURES = ['north_net_buy']

SENTIMENT_FEATURES = ['sentiment_mean', 'sentiment_std', 'news_count']

MACRO_FEATURES = ['lpr_1y', 'lpr_5y', 'cpi', 'm2_yoy']

ALL_MULTI_FACTOR_FEATURES = (
    FUND_FLOW_FEATURES + NORTH_FLOW_FEATURES +
    SENTIMENT_FEATURES + MACRO_FEATURES
)


class FeatureBuilder:
    def __init__(self, config: dict):
        self.config = config
        self.window_size = config.get("window_size", 60)
        self.normalization = config.get("normalization", "minmax")
        self.scaler = None
        self.y_scaler = None  # 目标值归一化
        self.feature_columns = None

        # 多因子配置
        self.use_multi_factors = config.get("use_multi_factors", False)
        self.multi_factor_config = config.get("multi_factors", {})

        logger.info(f"特征构建器初始化: window_size={self.window_size}, "
                   f"normalization={self.normalization}, use_multi_factors={self.use_multi_factors}")
    
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

    def build_features_with_multi_factors(
        self,
        df: pd.DataFrame,
        factors: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        构建包含多因子的特征集

        Args:
            df: 价格数据（包含基础OHLCV）
            factors: 多因子数据字典，由MultiFactorCollector.collect_all_factors()返回

        Returns:
            DataFrame: 包含技术指标和多因子的完整特征集
        """
        data_len = len(df)
        logger.info(f"开始构建多因子特征，原始数据: {data_len} 条")

        # 1. 先合并多因子数据（如果提供）
        if factors and self.use_multi_factors:
            df = self._merge_multi_factors(df, factors)
            logger.info(f"多因子合并后: {len(df)} 条记录, {len(df.columns)} 列")

        # 2. 计算技术指标
        adjusted_config = self._adjust_config_for_data_size(df, self.config)
        df = TechnicalIndicators.calculate_all_indicators(df, adjusted_config)

        # 3. 计算多因子衍生特征
        if self.use_multi_factors:
            df = self._calculate_multi_factor_derived_features(df)

        # 4. 处理缺失值
        before_dropna = len(df)
        df = df.dropna()
        after_dropna = len(df)

        dropped_rows = before_dropna - after_dropna
        if dropped_rows > 0:
            logger.warning(f"dropna() 删除了 {dropped_rows} 行数据 ({dropped_rows/before_dropna*100:.1f}%)")

        if len(df) == 0:
            raise ValueError(
                f"特征构建后数据为空！原始数据{data_len}条，dropna后{after_dropna}条。"
                f"建议检查多因子数据是否有效。"
            )

        logger.info(f"多因子特征构建完成: {len(df)} 条记录, {len(df.columns)} 个特征")
        return df

    def _merge_multi_factors(
        self,
        df: pd.DataFrame,
        factors: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        合并多因子数据到价格数据

        Args:
            df: 价格数据
            factors: 多因子数据字典

        Returns:
            DataFrame: 合并后的数据
        """
        result = df.copy()
        result['trade_date'] = pd.to_datetime(result['trade_date'])

        # 1. 合并资金流向因子
        if 'fund_flow' in factors and not factors['fund_flow'].empty:
            fund_flow = factors['fund_flow'].copy()
            fund_flow['trade_date'] = pd.to_datetime(fund_flow['trade_date'])

            fund_cols = ['trade_date'] + [c for c in FUND_FLOW_FEATURES if c in fund_flow.columns]
            fund_flow = fund_flow[fund_cols]

            result = result.merge(fund_flow, on='trade_date', how='left')
            logger.debug("合并资金流向因子完成")

        # 2. 合并北向资金因子
        if 'north_flow' in factors and not factors['north_flow'].empty:
            north_flow = factors['north_flow'].copy()
            north_flow['trade_date'] = pd.to_datetime(north_flow['trade_date'])

            north_cols = ['trade_date', 'net_buy_amount']
            north_flow = north_flow[north_cols].rename(
                columns={'net_buy_amount': 'north_net_buy'}
            )

            result = result.merge(north_flow, on='trade_date', how='left')
            logger.debug("合并北向资金因子完成")

        # 3. 合并舆情因子
        if 'sentiment' in factors and not factors['sentiment'].empty:
            sentiment = factors['sentiment'].copy()
            sentiment['trade_date'] = pd.to_datetime(sentiment['trade_date'])

            sentiment_cols = ['trade_date'] + [c for c in SENTIMENT_FEATURES if c in sentiment.columns]
            sentiment = sentiment[sentiment_cols]

            result = result.merge(sentiment, on='trade_date', how='left')
            logger.debug("合并舆情因子完成")

        # 4. 合并LPR因子（月度数据，前向填充）
        if 'lpr' in factors and not factors['lpr'].empty:
            lpr = factors['lpr'].copy()
            lpr['trade_date'] = pd.to_datetime(lpr['trade_date'])

            result = result.merge(lpr, on='trade_date', how='left')
            result['lpr_1y'] = result['lpr_1y'].ffill()
            result['lpr_5y'] = result['lpr_5y'].ffill()
            logger.debug("合并LPR因子完成")

        # 5. 合并CPI因子
        if 'cpi' in factors and not factors['cpi'].empty:
            cpi = factors['cpi'].copy()
            cpi['trade_date'] = pd.to_datetime(cpi['trade_date'])

            result = result.merge(cpi, on='trade_date', how='left')
            result['cpi'] = result['cpi'].ffill()
            logger.debug("合并CPI因子完成")

        # 6. 合并M2因子
        if 'm2' in factors and not factors['m2'].empty:
            m2 = factors['m2'].copy()
            m2['trade_date'] = pd.to_datetime(m2['trade_date'])

            result = result.merge(m2, on='trade_date', how='left')
            result['m2_yoy'] = result['m2_yoy'].ffill()
            logger.debug("合并M2因子完成")

        # 填充缺失的多因子列为0（对于日度数据）
        for col in FUND_FLOW_FEATURES + NORTH_FLOW_FEATURES + SENTIMENT_FEATURES:
            if col in result.columns:
                result[col] = result[col].fillna(0)

        return result

    def _calculate_multi_factor_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算多因子衍生特征

        Args:
            df: 包含多因子的数据

        Returns:
            DataFrame: 添加衍生特征后的数据
        """
        result = df.copy()

        # 1. 资金流向衍生特征
        if 'main_net_inflow' in result.columns:
            # 主力资金流向变化率
            result['main_flow_change'] = result['main_net_inflow'].pct_change()

            # 主力资金流向5日均值
            result['main_flow_ma5'] = result['main_net_inflow'].rolling(5).mean()

            # 资金流向动量（主力-散户）
            if 'small_net_inflow' in result.columns:
                result['fund_flow_momentum'] = (
                    result['main_net_inflow'] - result['small_net_inflow']
                )

            logger.debug("计算资金流向衍生特征完成")

        # 2. 北向资金衍生特征
        if 'north_net_buy' in result.columns:
            # 北向资金变化率
            result['north_flow_change'] = result['north_net_buy'].pct_change()

            # 北向资金5日均值
            result['north_flow_ma5'] = result['north_net_buy'].rolling(5).mean()

            # 北向资金累计（10日）
            result['north_flow_sum10'] = result['north_net_buy'].rolling(10).sum()

            logger.debug("计算北向资金衍生特征完成")

        # 3. 舆情衍生特征
        if 'sentiment_mean' in result.columns:
            # 情绪变化
            result['sentiment_change'] = result['sentiment_mean'].diff()

            # 情绪5日均值
            result['sentiment_ma5'] = result['sentiment_mean'].rolling(5).mean()

            # 新闻热度变化
            if 'news_count' in result.columns:
                result['news_count_change'] = result['news_count'].pct_change()

            logger.debug("计算舆情衍生特征完成")

        # 4. 宏观因子衍生特征
        if 'lpr_1y' in result.columns:
            # LPR变化
            result['lpr_1y_change'] = result['lpr_1y'].diff()

        if 'cpi' in result.columns:
            # CPI变化
            result['cpi_change'] = result['cpi'].diff()

        # 5. 组合因子
        # 资金情绪综合因子
        if 'main_net_inflow_pct' in result.columns and 'sentiment_mean' in result.columns:
            result['fund_sentiment_score'] = (
                result['main_net_inflow_pct'] * 0.7 +
                result['sentiment_mean'] * 30  # 情绪归一化后约在[-1,1]，放大
            )

        # 填充NaN
        derived_cols = [
            'main_flow_change', 'main_flow_ma5', 'fund_flow_momentum',
            'north_flow_change', 'north_flow_ma5', 'north_flow_sum10',
            'sentiment_change', 'sentiment_ma5', 'news_count_change',
            'lpr_1y_change', 'cpi_change', 'fund_sentiment_score'
        ]

        for col in derived_cols:
            if col in result.columns:
                result[col] = result[col].fillna(0)

        return result
    
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
        
        # 检查是否只使用平稳特征
        if self.config.get("use_stationary", False):
            logger.info("启用平稳性处理：移除价格相关的非平稳特征，仅保留收益率、差分和技术指标")
            # 基础非平稳列
            non_stationary_cols = ["open", "high", "low", "close", "adj_close", "volume"]
            exclude_columns.extend(non_stationary_cols)
            
            # 移除价格相关的指标 (MA, EMA, BB等)
            for col in df.columns:
                if col.startswith("ma_") or col.startswith("ema_"):
                    exclude_columns.append(col)
                if col.startswith("bb_") and "width" not in col:  # 保留bb_width (相对值)
                    exclude_columns.append(col)
        
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
    
    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        """
        将归一化的预测值还原为真实数值
        """
        if self.y_scaler is None:
            return y
        
        # 确保输入维度正确
        y_reshaped = y.reshape(-1, 1)
        y_inv = self.y_scaler.inverse_transform(y_reshaped)
        return y_inv.flatten()
    
    def save_scaler(self, save_path: str) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "y_scaler": self.y_scaler,
                "feature_columns": self.feature_columns,
                "window_size": self.window_size,
                "normalization": self.normalization,
                "config": self.config  # 保存训练时的配置
            }, f)

        logger.info(f"Scaler保存至: {save_path}")

    def load_scaler(self, load_path: str) -> None:
        with open(load_path, "rb") as f:
            data = pickle.load(f)
            self.scaler = data["scaler"]
            self.y_scaler = data.get("y_scaler")
            self.feature_columns = data["feature_columns"]
            self.window_size = data["window_size"]
            self.normalization = data["normalization"]
            # 加载训练时的配置，确保预测时使用相同的特征
            if "config" in data:
                self.config = data["config"]

        logger.info(f"Scaler加载自: {load_path}")

    def prepare_prediction_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        准备预测数据

        注意：此方法会使用与训练时相同的配置来构建特征，
        确保特征列与训练时一致。
        """
        # 使用固定配置构建特征，不做动态调整
        df = self._build_features_for_prediction(df)

        if df.empty:
            raise ValueError("数据为空")

        if self.feature_columns is None:
            raise ValueError("特征列未定义，请先训练模型或加载scaler")

        # 检查特征列是否存在
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            available_cols = list(df.columns)
            raise ValueError(
                f"缺少特征列: {missing_cols}\n"
                f"可用列: {available_cols}\n"
                f"请确保有足够的历史数据来计算所需的技术指标"
            )

        df_features = df[self.feature_columns].values

        if self.scaler is None:
            raise ValueError("Scaler未初始化，请先训练模型或加载scaler")

        df_features = self.scaler.transform(df_features)

        if len(df_features) < self.window_size:
            raise ValueError(f"数据不足，需要至少 {self.window_size} 条记录")

        X = df_features[-self.window_size:]
        X = np.expand_dims(X, axis=0)

        return X

    def _build_features_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        为预测构建特征，不做数据量相关的动态调整
        使用与训练时相同的配置
        """
        data_len = len(df)
        logger.info(f"为预测构建特征，原始数据: {data_len} 条")

        # 从feature_columns推断所需的技术指标配置
        # 这样可以兼容旧的scaler文件（config可能为空）
        config_for_indicators = self._infer_tech_config_from_features()

        df = TechnicalIndicators.calculate_all_indicators(df, config_for_indicators)

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

        logger.info(f"预测特征构建完成: {len(df)} 条记录, {len(df.columns)} 个特征")
        return df

    def _infer_tech_config_from_features(self) -> dict:
        """
        从feature_columns推断技术指标配置

        用于兼容旧的scaler文件（可能没有保存完整的config）
        """
        # 如果config中已有technical_indicators且不为空，直接使用
        tech_config = self.config.get("technical_indicators", {})
        if tech_config and tech_config.get("ma_periods"):
            return self.config

        # 从feature_columns推断配置
        if not self.feature_columns:
            return self.config

        inferred_config = {"technical_indicators": {}}

        # 推断MA周期
        ma_periods = []
        for col in self.feature_columns:
            if col.startswith("ma_") and col[3:].isdigit():
                ma_periods.append(int(col[3:]))
        if ma_periods:
            inferred_config["technical_indicators"]["ma_periods"] = sorted(ma_periods)
        else:
            # 如果feature_columns中没有ma_*列，使用最小配置避免计算不必要的指标
            inferred_config["technical_indicators"]["ma_periods"] = []

        # 推断EMA周期
        ema_periods = []
        for col in self.feature_columns:
            if col.startswith("ema_") and col[4:].isdigit():
                ema_periods.append(int(col[4:]))
        if ema_periods:
            inferred_config["technical_indicators"]["ema_periods"] = sorted(ema_periods)
        else:
            inferred_config["technical_indicators"]["ema_periods"] = []

        # 推断其他指标（使用默认值，因为这些通常不会变化）
        # RSI
        if "rsi" in self.feature_columns:
            inferred_config["technical_indicators"]["rsi_period"] = 14

        # KDJ
        if any(col.startswith("kdj_") for col in self.feature_columns):
            inferred_config["technical_indicators"]["kdj_period"] = 9

        # Bollinger Bands
        if any(col.startswith("bb_") for col in self.feature_columns):
            inferred_config["technical_indicators"]["bollinger_period"] = 20
            inferred_config["technical_indicators"]["bollinger_std"] = 2

        # ATR
        if "atr" in self.feature_columns:
            inferred_config["technical_indicators"]["atr_period"] = 14

        # MACD（使用标准参数）
        if "macd" in self.feature_columns:
            inferred_config["technical_indicators"]["macd_fast"] = 12
            inferred_config["technical_indicators"]["macd_slow"] = 26
            inferred_config["technical_indicators"]["macd_signal"] = 9

        logger.info(f"从feature_columns推断技术指标配置: {inferred_config['technical_indicators']}")

        return inferred_config
