# -*- coding: utf-8 -*-
"""
改进版API接口 - 添加到 main.py 中使用
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path


class ImprovedPredictionResponse(BaseModel):
    """改进版预测响应模型"""

    # 元数据
    metadata: Dict = Field(..., description="预测元数据（时间、模型版本、数据质量等）")

    # 价格信息
    price: Dict = Field(..., description="价格信息（当前价、预测价、涨跌幅）")

    # 概率信息
    probability: Dict = Field(..., description="概率信息（上涨/下跌概率）")

    # 不确定性
    uncertainty: Dict = Field(..., description="预测不确定性（标准差、置信区间）")

    # 风险指标
    risk_metrics: Dict = Field(..., description="风险指标（波动率、VaR、夏普比率）")

    # 交易信号
    trading_signals: Dict = Field(..., description="交易信号（操作建议、止损止盈）")

    # 免责声明
    disclaimer: Dict = Field(..., description="免责声明和模型局限性")

    class Config:
        schema_extra = {
            "example": {
                "metadata": {
                    "stock_code": "600519",
                    "prediction_time": "2026-01-07T01:30:00",
                    "prediction_valid_until": "2026-01-12T01:30:00",
                    "model_version": "epoch_50",
                    "model_val_loss": 0.0012,
                    "n_simulations": 50,
                    "data_quality": {
                        "score": 100,
                        "level": "优",
                        "has_nan": False,
                        "has_inf": False
                    }
                },
                "price": {
                    "current": 1580.50,
                    "predicted": 1618.32,
                    "change_amount": 37.82,
                    "change_pct": 2.39
                },
                "probability": {
                    "up": 72.5,
                    "down": 27.5,
                    "large_up": 15.0,
                    "large_down": 5.0,
                    "direction": "上涨"
                },
                "uncertainty": {
                    "mean_return_pct": 2.39,
                    "median_return_pct": 2.31,
                    "std_return_pct": 1.25,
                    "confidence_intervals": {
                        "ci_5_pct": -0.5,
                        "ci_25_pct": 1.2,
                        "ci_75_pct": 3.5,
                        "ci_95_pct": 5.1
                    }
                },
                "risk_metrics": {
                    "volatility": {
                        "daily": 0.0125,
                        "annualized": 0.089,
                        "level": "低"
                    },
                    "value_at_risk": {
                        "var_95_pct": -0.5,
                        "var_95_amount": -7.9,
                        "description": "95%置信度下最大损失约0.50%"
                    },
                    "ratios": {
                        "sharpe_ratio": 1.85,
                        "reward_risk_ratio": 1.91
                    }
                },
                "trading_signals": {
                    "action": "买入",
                    "reason": "上涨概率72.5%，预期收益2.39%",
                    "confidence": "高",
                    "position": {
                        "suggested_pct": 50,
                        "description": "建议仓位50%"
                    },
                    "stop_loss": {
                        "price": 1533.09,
                        "pct": -3.0,
                        "description": "建议止损价: ¥1533.09 (-3.0%)"
                    },
                    "take_profit": {
                        "price": 1660.25,
                        "pct": 5.04,
                        "description": "建议止盈价: ¥1660.25 (5.0%)"
                    },
                    "risk_assessment": {
                        "overall_risk": "低",
                        "risk_score": 2,
                        "warning": None
                    }
                },
                "disclaimer": {
                    "warning": "本预测仅供参考，不构成投资建议。股市有风险，投资需谨慎。",
                    "model_limitations": [
                        "模型基于历史数据训练，无法预测突发事件",
                        "预测准确性受市场环境影响",
                        "建议结合基本面分析综合判断"
                    ]
                }
            }
        }


class ErrorResponse(BaseModel):
    """统一错误响应"""
    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误信息")
    details: Optional[Dict] = Field(None, description="详细信息")
    timestamp: str = Field(..., description="错误时间")
    error_code: str = Field(..., description="错误代码")

    class Config:
        schema_extra = {
            "example": {
                "error": "ModelNotFoundError",
                "message": "模型文件不存在，请先训练模型",
                "details": {
                    "stock_code": "600519",
                    "model_path": "data/models/600519_model.pth"
                },
                "timestamp": "2026-01-07T01:30:00",
                "error_code": "E404_MODEL"
            }
        }


# 错误代码定义
class ErrorCodes:
    """API错误代码"""
    MODEL_NOT_FOUND = "E404_MODEL"
    SCALER_NOT_FOUND = "E404_SCALER"
    DATA_NOT_FOUND = "E404_DATA"
    INVALID_STOCK_CODE = "E400_INVALID_STOCK"
    DATA_QUALITY_ERROR = "E400_DATA_QUALITY"
    PREDICTION_ERROR = "E500_PREDICTION"
    INTERNAL_ERROR = "E500_INTERNAL"


def create_error_response(
    error_type: str,
    message: str,
    error_code: str,
    details: Optional[Dict] = None
) -> Dict:
    """创建统一的错误响应"""
    return {
        "error": error_type,
        "message": message,
        "details": details,
        "timestamp": datetime.now().isoformat(),
        "error_code": error_code
    }


# 改进版API端点示例
"""
@app.post("/api/v2/predict/comprehensive", response_model=ImprovedPredictionResponse)
async def predict_comprehensive(request: PredictRequest):
    '''
    改进版综合预测接口

    返回完整的预测结果，包括：
    - 价格预测（当前价、预测价、涨跌幅）
    - 概率分析（上涨/下跌/大涨/大跌概率）
    - 不确定性估计（标准差、置信区间）
    - 风险指标（波动率、VaR、夏普比率）
    - 交易信号（操作建议、止损止盈、仓位建议）
    - 元数据（预测时间、模型版本、数据质量）
    '''
    try:
        stock_code = request.stock_code
        model_path = Path(config.model.get("model_save_path", "data/models")) / f"{stock_code}_model.pth"
        scaler_path = Path(config.model.get("model_save_path", "data/models")) / f"{stock_code}_scaler.pkl"

        # 1. 验证文件存在性
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail=create_error_response(
                    "ModelNotFoundError",
                    f"模型文件不存在: {model_path}，请先训练模型",
                    ErrorCodes.MODEL_NOT_FOUND,
                    {"stock_code": stock_code, "model_path": str(model_path)}
                )
            )

        if not scaler_path.exists():
            raise HTTPException(
                status_code=404,
                detail=create_error_response(
                    "ScalerNotFoundError",
                    f"Scaler文件不存在: {scaler_path}，请先训练模型",
                    ErrorCodes.SCALER_NOT_FOUND,
                    {"stock_code": stock_code, "scaler_path": str(scaler_path)}
                )
            )

        # 2. 加载数据
        df = db.get_stock_daily(stock_code)

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=create_error_response(
                    "DataNotFoundError",
                    f"未找到股票 {stock_code} 的数据",
                    ErrorCodes.DATA_NOT_FOUND,
                    {"stock_code": stock_code}
                )
            )

        # 3. 数据验证
        current_price = float(df.iloc[-1]["close"])

        if current_price <= 0:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "DataValidationError",
                    f"当前价格异常: {current_price}",
                    ErrorCodes.DATA_QUALITY_ERROR,
                    {"stock_code": stock_code, "current_price": current_price}
                )
            )

        # 4. 准备特征
        stock_feature_builder = FeatureBuilder(config.features)
        stock_feature_builder.load_scaler(str(scaler_path))
        X = stock_feature_builder.prepare_prediction_data(df)

        # 5. 加载模型
        input_size = X.shape[2]
        model = LSTMModel(
            input_size=input_size,
            hidden_size=config.model.get("lstm", {}).get("hidden_size", 128),
            num_layers=config.model.get("lstm", {}).get("num_layers", 2),
            dropout=config.model.get("lstm", {}).get("dropout", 0.2)
        )

        # 6. 使用改进版预测器
        from src.models.improved_predictor import ImprovedPredictor

        predictor = ImprovedPredictor(model, config.system.get("device", "cpu"))
        predictor.load_model(str(model_path))

        # 7. 获取综合预测结果
        result = predictor.get_comprehensive_prediction(
            X,
            current_price,
            stock_code,
            n_simulations=50
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预测失败: {e}")
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "PredictionError",
                f"预测过程中发生错误: {str(e)}",
                ErrorCodes.PREDICTION_ERROR,
                {"stock_code": request.stock_code, "error": str(e)}
            )
        )
"""
