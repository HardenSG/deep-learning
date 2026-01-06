import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DirectionLoss(nn.Module):
    """
    方向损失：惩罚预测方向错误的情况

    对于涨跌方向的预测，比具体数值更重要
    """
    def __init__(self):
        super(DirectionLoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测值 (batch_size,)
            target: 真实值 (batch_size,)

        Returns:
            方向损失（0-1之间，0表示方向完全正确，1表示完全错误）
        """
        # 计算方向是否一致：同号为正确，异号为错误
        direction_correct = (pred * target) > 0

        # 方向错误的比例
        direction_loss = 1.0 - direction_correct.float().mean()

        return direction_loss


class WeightedDirectionLoss(nn.Module):
    """
    加权方向损失：对大幅度涨跌的方向预测错误给予更大惩罚

    如果预测大涨实际大跌，这比预测小涨实际小跌更严重
    """
    def __init__(self):
        super(WeightedDirectionLoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测值 (batch_size,)
            target: 真实值 (batch_size,)

        Returns:
            加权方向损失
        """
        # 方向是否一致
        direction_correct = (pred * target) > 0

        # 使用真实值的绝对值作为权重（幅度越大，权重越大）
        weights = torch.abs(target) + 1e-8  # 避免除零

        # 方向错误时权重为正，正确时权重为负
        weighted_errors = torch.where(
            direction_correct,
            torch.zeros_like(weights),
            weights
        )

        # 归一化损失
        loss = weighted_errors.mean() / (weights.mean() + 1e-8)

        return loss


class HybridLoss(nn.Module):
    """
    混合损失函数：MSE + 方向损失

    结合了数值预测的准确性和方向预测的正确性

    Args:
        alpha: MSE损失的权重
        beta: 方向损失的权重
        use_weighted_direction: 是否使用加权方向损失
    """
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        use_weighted_direction: bool = True
    ):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()

        if use_weighted_direction:
            self.direction_loss = WeightedDirectionLoss()
        else:
            self.direction_loss = DirectionLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测值 (batch_size,)
            target: 真实值 (batch_size,)

        Returns:
            混合损失
        """
        mse = self.mse_loss(pred, target)
        direction = self.direction_loss(pred, target)

        total_loss = self.alpha * mse + self.beta * direction

        return total_loss

    def get_components(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        返回各个损失分量，用于监控
        """
        with torch.no_grad():
            mse = self.mse_loss(pred, target)
            direction = self.direction_loss(pred, target)
            total = self.alpha * mse + self.beta * direction

        return {
            "total_loss": total.item(),
            "mse_loss": mse.item(),
            "direction_loss": direction.item()
        }


class QuantileLoss(nn.Module):
    """
    分位数损失：对极端值给予不对称惩罚

    可用于预测不同置信度下的收益率

    Args:
        quantile: 分位数（0-1之间），0.5为中位数
    """
    def __init__(self, quantile: float = 0.5):
        super(QuantileLoss, self).__init__()
        assert 0 < quantile < 1, "quantile必须在0-1之间"
        self.quantile = quantile

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测值 (batch_size,)
            target: 真实值 (batch_size,)

        Returns:
            分位数损失
        """
        errors = target - pred
        loss = torch.max(
            self.quantile * errors,
            (self.quantile - 1) * errors
        )
        return loss.mean()


class ExtremeLoss(nn.Module):
    """
    极端值损失：对大幅涨跌的预测错误给予额外惩罚

    在金融市场中，错过大涨或未能避免大跌的代价很高

    Args:
        threshold: 定义"极端值"的阈值（如0.05表示±5%）
        extreme_weight: 极端值的额外权重
    """
    def __init__(self, threshold: float = 0.05, extreme_weight: float = 2.0):
        super(ExtremeLoss, self).__init__()
        self.threshold = threshold
        self.extreme_weight = extreme_weight
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测值 (batch_size,)
            target: 真实值 (batch_size,)

        Returns:
            加权后的MSE损失
        """
        mse = self.mse_loss(pred, target)

        # 识别极端值
        is_extreme = torch.abs(target) > self.threshold

        # 对极端值应用额外权重
        weights = torch.where(
            is_extreme,
            torch.ones_like(mse) * self.extreme_weight,
            torch.ones_like(mse)
        )

        weighted_mse = (mse * weights).mean()

        return weighted_mse


class AdvancedHybridLoss(nn.Module):
    """
    高级混合损失：MSE + 方向损失 + 极端值损失

    全面考虑数值准确性、方向正确性和极端情况

    Args:
        alpha: MSE损失权重
        beta: 方向损失权重
        gamma: 极端值损失权重
        extreme_threshold: 极端值阈值
    """
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        extreme_threshold: float = 0.05
    ):
        super(AdvancedHybridLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.mse_loss = nn.MSELoss()
        self.direction_loss = WeightedDirectionLoss()
        self.extreme_loss = ExtremeLoss(threshold=extreme_threshold)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测值 (batch_size,)
            target: 真实值 (batch_size,)

        Returns:
            高级混合损失
        """
        mse = self.mse_loss(pred, target)
        direction = self.direction_loss(pred, target)
        extreme = self.extreme_loss(pred, target)

        total_loss = self.alpha * mse + self.beta * direction + self.gamma * extreme

        return total_loss

    def get_components(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        返回各个损失分量，用于监控
        """
        with torch.no_grad():
            mse = self.mse_loss(pred, target)
            direction = self.direction_loss(pred, target)
            extreme = self.extreme_loss(pred, target)
            total = self.alpha * mse + self.beta * direction + self.gamma * extreme

        return {
            "total_loss": total.item(),
            "mse_loss": mse.item(),
            "direction_loss": direction.item(),
            "extreme_loss": extreme.item()
        }
