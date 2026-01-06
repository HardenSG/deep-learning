import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path
from datetime import datetime, timedelta
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ImprovedPredictor:
    """
    改进版预测器 - 提供完整的预测结果和风险评估

    改进点:
    1. 完整的返回格式（包含时间戳、元数据、风险指标）
    2. 更准确的概率估计
    3. 详细的风险评估（波动率、VaR、止损止盈）
    4. 模型性能指标
    5. 数据质量检查
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # 模型元数据
        self.model_metadata = {}

        logger.info(f"改进版预测器初始化完成，设备: {self.device}")

    def load_model(self, model_path: str) -> None:
        """加载模型并提取元数据"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # 提取模型元数据
        self.model_metadata = {
            "model_path": model_path,
            "epoch": checkpoint.get("epoch", "unknown"),
            "train_loss": checkpoint.get("train_losses", [])[-1] if checkpoint.get("train_losses") else None,
            "val_loss": checkpoint.get("val_loss", None),
            "model_info": checkpoint.get("model_info", {}),
            "trained_at": checkpoint.get("trained_at", "unknown")
        }

        logger.info(f"模型加载成功: {model_path}")
        logger.info(f"模型验证损失: {self.model_metadata.get('val_loss', 'N/A')}")

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_iterations: int = 50
    ) -> Dict:
        """
        使用 Dropout Monte Carlo 估计预测不确定性

        比简单的 sigmoid 更准确
        """
        self.model.train()  # 开启dropout

        predictions = []
        for _ in range(n_iterations):
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                pred = self.model(X_tensor)
                predictions.append(pred.cpu().numpy().flatten()[0])

        self.model.eval()

        predictions = np.array(predictions)

        # 统计分析
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        median_pred = np.median(predictions)

        # 置信区间
        ci_5 = np.percentile(predictions, 5)
        ci_25 = np.percentile(predictions, 25)
        ci_75 = np.percentile(predictions, 75)
        ci_95 = np.percentile(predictions, 95)

        # 方向概率（更准确的估计）
        up_count = np.sum(predictions > 0)
        up_probability = up_count / n_iterations

        # 极端情况概率
        large_up_count = np.sum(predictions > 0.05)  # 大涨（>5%）
        large_down_count = np.sum(predictions < -0.05)  # 大跌（<-5%）

        return {
            "mean_return": float(mean_pred),
            "median_return": float(median_pred),
            "std_return": float(std_pred),
            "up_probability": float(up_probability),
            "down_probability": float(1 - up_probability),
            "large_up_probability": float(large_up_count / n_iterations),
            "large_down_probability": float(large_down_count / n_iterations),
            "confidence_intervals": {
                "ci_5": float(ci_5),
                "ci_25": float(ci_25),
                "ci_75": float(ci_75),
                "ci_95": float(ci_95)
            },
            "direction": "上涨" if mean_pred > 0 else "下跌",
            "n_simulations": n_iterations
        }

    def calculate_risk_metrics(
        self,
        predictions: Dict,
        current_price: float
    ) -> Dict:
        """
        计算风险指标

        包括：波动率、VaR、预期收益风险比等
        """
        mean_return = predictions["mean_return"]
        std_return = predictions["std_return"]

        # 年化波动率（假设预测5天，252个交易日/年）
        annualized_volatility = std_return * np.sqrt(252 / 5)

        # Value at Risk (95% 置信水平)
        var_95 = predictions["confidence_intervals"]["ci_5"]

        # 夏普比率估计（假设无风险利率3%）
        risk_free_rate_daily = 0.03 / 252 * 5
        sharpe_ratio = (mean_return - risk_free_rate_daily) / std_return if std_return > 0 else 0

        # 预期收益风险比
        reward_risk_ratio = abs(mean_return / std_return) if std_return > 0 else 0

        # 最大可能损失（以当前价格计算）
        max_loss_amount = current_price * var_95
        max_loss_pct = var_95 * 100

        # 风险等级评估
        if std_return < 0.02:
            volatility_level = "低"
        elif std_return < 0.05:
            volatility_level = "中"
        else:
            volatility_level = "高"

        return {
            "volatility": {
                "daily": float(std_return),
                "annualized": float(annualized_volatility),
                "level": volatility_level
            },
            "value_at_risk": {
                "var_95_pct": float(max_loss_pct),
                "var_95_amount": float(max_loss_amount),
                "description": f"95%置信度下最大损失约{abs(max_loss_pct):.2f}%"
            },
            "ratios": {
                "sharpe_ratio": float(sharpe_ratio),
                "reward_risk_ratio": float(reward_risk_ratio)
            }
        }

    def generate_trading_signals(
        self,
        predictions: Dict,
        risk_metrics: Dict,
        current_price: float
    ) -> Dict:
        """
        生成交易信号和建议

        包括：操作建议、止损止盈价格、仓位建议等
        """
        mean_return = predictions["mean_return"]
        up_prob = predictions["up_probability"]
        std_return = predictions["std_return"]

        # 预测价格
        predicted_price = current_price * (1 + mean_return)

        # 置信度评估
        if std_return < 0.01:
            confidence = "很高"
        elif std_return < 0.03:
            confidence = "高"
        elif std_return < 0.05:
            confidence = "中"
        else:
            confidence = "低"

        # 交易信号生成
        if up_prob >= 0.7 and mean_return > 0.02:
            action = "强烈买入"
            reason = f"上涨概率{up_prob*100:.1f}%，预期收益{mean_return*100:.2f}%"
            position_pct = 70  # 建议仓位70%
        elif up_prob >= 0.6 and mean_return > 0.01:
            action = "买入"
            reason = f"上涨概率{up_prob*100:.1f}%，预期收益{mean_return*100:.2f}%"
            position_pct = 50
        elif up_prob >= 0.55 and mean_return > 0.005:
            action = "小幅买入"
            reason = f"上涨概率{up_prob*100:.1f}%，适合小仓位试探"
            position_pct = 30
        elif up_prob >= 0.45:
            action = "观望"
            reason = "涨跌概率接近，建议等待更明确信号"
            position_pct = 0
        elif up_prob >= 0.4 and mean_return > -0.01:
            action = "减仓观望"
            reason = f"下跌概率{(1-up_prob)*100:.1f}%，建议减少持仓"
            position_pct = 20
        else:
            action = "卖出/空仓"
            reason = f"下跌概率{(1-up_prob)*100:.1f}%，预期下跌{abs(mean_return)*100:.2f}%"
            position_pct = 0

        # 止损止盈价格
        stop_loss_pct = -0.03  # 默认止损3%
        take_profit_pct = mean_return + std_return * 2  # 预测值 + 2倍标准差

        stop_loss_price = current_price * (1 + stop_loss_pct)
        take_profit_price = current_price * (1 + take_profit_pct)

        # 风险评级
        risk_score = 0
        if risk_metrics["volatility"]["level"] == "高":
            risk_score += 3
        elif risk_metrics["volatility"]["level"] == "中":
            risk_score += 2
        else:
            risk_score += 1

        if confidence == "低":
            risk_score += 3
        elif confidence == "中":
            risk_score += 2
        else:
            risk_score += 1

        if risk_score >= 5:
            overall_risk = "高"
        elif risk_score >= 3:
            overall_risk = "中"
        else:
            overall_risk = "低"

        return {
            "action": action,
            "reason": reason,
            "confidence": confidence,
            "position": {
                "suggested_pct": position_pct,
                "description": f"建议仓位{position_pct}%"
            },
            "stop_loss": {
                "price": float(stop_loss_price),
                "pct": stop_loss_pct * 100,
                "description": f"建议止损价: ¥{stop_loss_price:.2f} ({stop_loss_pct*100:.1f}%)"
            },
            "take_profit": {
                "price": float(take_profit_price),
                "pct": take_profit_pct * 100,
                "description": f"建议止盈价: ¥{take_profit_price:.2f} ({take_profit_pct*100:.1f}%)"
            },
            "risk_assessment": {
                "overall_risk": overall_risk,
                "risk_score": risk_score,
                "warning": self._get_risk_warning(overall_risk, action)
            }
        }

    def _get_risk_warning(self, risk_level: str, action: str) -> Optional[str]:
        """生成风险警告"""
        if risk_level == "高" and action in ["强烈买入", "买入"]:
            return "⚠️ 高风险：波动较大，建议控制仓位并设置止损"
        elif risk_level == "高":
            return "⚠️ 高风险：市场不确定性大，谨慎操作"
        elif risk_level == "中" and action in ["强烈买入"]:
            return "⚠️ 中等风险：建议分批建仓"
        return None

    def get_comprehensive_prediction(
        self,
        X: np.ndarray,
        current_price: float,
        stock_code: str,
        n_simulations: int = 50
    ) -> Dict:
        """
        获取完整的预测结果

        包括：预测值、概率、风险指标、交易信号、元数据等
        """
        # 1. 预测和不确定性估计
        predictions = self.predict_with_uncertainty(X, n_simulations)

        # 2. 风险指标
        risk_metrics = self.calculate_risk_metrics(predictions, current_price)

        # 3. 交易信号
        trading_signals = self.generate_trading_signals(predictions, risk_metrics, current_price)

        # 4. 价格预测
        mean_return = predictions["mean_return"]
        predicted_price = current_price * (1 + mean_return)
        price_change = predicted_price - current_price

        # 5. 元数据
        now = datetime.now()
        prediction_valid_until = now + timedelta(days=5)  # 预测5天有效

        # 6. 数据质量检查
        data_quality = self._check_data_quality(X)

        # 组装完整结果
        result = {
            "metadata": {
                "stock_code": stock_code,
                "prediction_time": now.isoformat(),
                "prediction_valid_until": prediction_valid_until.isoformat(),
                "model_version": self.model_metadata.get("epoch", "unknown"),
                "model_val_loss": self.model_metadata.get("val_loss"),
                "n_simulations": n_simulations,
                "data_quality": data_quality
            },
            "price": {
                "current": float(current_price),
                "predicted": float(predicted_price),
                "change_amount": float(price_change),
                "change_pct": float(mean_return * 100)
            },
            "probability": {
                "up": float(predictions["up_probability"] * 100),
                "down": float(predictions["down_probability"] * 100),
                "large_up": float(predictions["large_up_probability"] * 100),
                "large_down": float(predictions["large_down_probability"] * 100),
                "direction": predictions["direction"]
            },
            "uncertainty": {
                "mean_return_pct": float(predictions["mean_return"] * 100),
                "median_return_pct": float(predictions["median_return"] * 100),
                "std_return_pct": float(predictions["std_return"] * 100),
                "confidence_intervals": {
                    "ci_5_pct": float(predictions["confidence_intervals"]["ci_5"] * 100),
                    "ci_25_pct": float(predictions["confidence_intervals"]["ci_25"] * 100),
                    "ci_75_pct": float(predictions["confidence_intervals"]["ci_75"] * 100),
                    "ci_95_pct": float(predictions["confidence_intervals"]["ci_95"] * 100)
                }
            },
            "risk_metrics": risk_metrics,
            "trading_signals": trading_signals,
            "disclaimer": {
                "warning": "本预测仅供参考，不构成投资建议。股市有风险，投资需谨慎。",
                "model_limitations": [
                    "模型基于历史数据训练，无法预测突发事件",
                    "预测准确性受市场环境影响",
                    "建议结合基本面分析综合判断"
                ]
            }
        }

        return result

    def _check_data_quality(self, X: np.ndarray) -> Dict:
        """检查输入数据质量"""
        # 检查异常值
        has_nan = np.isnan(X).any()
        has_inf = np.isinf(X).any()

        # 检查数据范围
        data_min = float(np.min(X))
        data_max = float(np.max(X))
        data_mean = float(np.mean(X))

        # 质量评分
        quality_score = 100
        issues = []

        if has_nan:
            quality_score -= 50
            issues.append("包含NaN值")

        if has_inf:
            quality_score -= 50
            issues.append("包含无穷值")

        if data_max > 10 or data_min < -10:
            quality_score -= 20
            issues.append("数据范围异常，可能未正确归一化")

        quality_level = "优" if quality_score >= 90 else "良" if quality_score >= 70 else "差"

        return {
            "score": quality_score,
            "level": quality_level,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "data_range": {
                "min": data_min,
                "max": data_max,
                "mean": data_mean
            },
            "issues": issues if issues else None
        }
