from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import load_config
from src.utils.logger import setup_logger, get_logger
from src.utils.database import Database
from src.data_collector.stock_data import StockDataCollector
from src.feature_engineering.feature_builder import FeatureBuilder
from src.models.lstm_model import LSTMModel
from src.models.predictor import ModelPredictor
from src.models.enhanced_predictor import EnhancedPredictor
import numpy as np

config = load_config()
setup_logger(
    log_dir=config.logging.get("log_dir", "logs"),
    log_file=config.logging.get("log_file", "api.log"),
    level=config.logging.get("level", "INFO")
)
logger = get_logger(__name__)

app = FastAPI(
    title="A股量化深度学习系统",
    description="基于LSTM的股票预测API",
    version="0.1.0"
)

if config.api.get("cors", {}).get("enabled", True):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.get("cors", {}).get("origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

db = Database(config.data.get("database_path", "data/stock_data.db"))
collector = StockDataCollector(db)
feature_builder = FeatureBuilder(config.features)


class PredictRequest(BaseModel):
    stock_code: str
    days: Optional[int] = 5


class TrainRequest(BaseModel):
    stock_codes: List[str]
    start_date: str
    end_date: Optional[str] = None


class DataUpdateRequest(BaseModel):
    stock_codes: Optional[List[str]] = None
    days_back: int = 30


class StockListResponse(BaseModel):
    total: int
    stocks: List[Dict[str, str]]


class PredictionResponse(BaseModel):
    stock_code: str
    prediction: float
    direction: str
    confidence: Optional[Dict] = None


class EnhancedPredictionResponse(BaseModel):
    stock_code: str
    current_price: float
    predicted_price: float
    price_change: float
    price_change_pct: float
    up_probability_pct: float
    down_probability_pct: float
    direction: str
    confidence_level: str
    recommendation: Dict
    monte_carlo: Optional[Dict] = None


@app.get("/")
async def root():
    return {
        "message": "A股量化深度学习系统 API",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/api/stocks", response_model=StockListResponse)
async def get_stock_list():
    try:
        stock_list = collector.get_stock_list()
        
        stocks = stock_list.to_dict("records")
        
        return {
            "total": len(stocks),
            "stocks": stocks[:100]
        }
    except Exception as e:
        logger.error(f"获取股票列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data/update")
async def update_data(request: DataUpdateRequest, background_tasks: BackgroundTasks):
    try:
        stock_codes = request.stock_codes or config.data.get("stock_list", [])
        
        if not stock_codes:
            raise HTTPException(status_code=400, detail="未指定股票代码")
        
        background_tasks.add_task(
            collector.update_all_stocks,
            stock_codes,
            request.days_back
        )
        
        return {
            "message": f"数据更新任务已启动",
            "stock_count": len(stock_codes),
            "days_back": request.days_back
        }
    except Exception as e:
        logger.error(f"更新数据失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/{stock_code}")
async def get_stock_data(
    stock_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100
):
    try:
        df = db.get_stock_daily(stock_code, start_date, end_date)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"未找到股票 {stock_code} 的数据")
        
        df = df.tail(limit)
        
        return {
            "stock_code": stock_code,
            "total": len(df),
            "data": df.to_dict("records")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取股票数据失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: PredictRequest):
    try:
        model_path = Path(config.model.get("model_save_path", "data/models")) / f"{request.stock_code}_model.pth"
        scaler_path = Path(config.model.get("model_save_path", "data/models")) / f"{request.stock_code}_scaler.pkl"
        
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"模型文件不存在: {model_path}，请先训练模型"
            )
        
        if not scaler_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Scaler文件不存在: {scaler_path}，请先训练模型"
            )
        
        df = db.get_stock_daily(request.stock_code)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"未找到股票 {request.stock_code} 的数据")
        
        current_price = float(df.iloc[-1]["close"])
        
        stock_feature_builder = FeatureBuilder(config.features)
        stock_feature_builder.load_scaler(str(scaler_path))
        
        X = stock_feature_builder.prepare_prediction_data(df)
        
        input_size = X.shape[2]
        model = LSTMModel(
            input_size=input_size,
            hidden_size=config.model.get("lstm", {}).get("hidden_size", 128),
            num_layers=config.model.get("lstm", {}).get("num_layers", 2),
            dropout=config.model.get("lstm", {}).get("dropout", 0.2)
        )
        
        predictor = EnhancedPredictor(model, config.system.get("device", "cpu"))
        predictor.load_model(str(model_path))
        
        result = predictor.get_detailed_analysis(X, current_price, use_monte_carlo=False)
        
        prediction = result["price_change_pct"] / 100
        direction = result["direction"]
        
        confidence = {
            "current_price": result["current_price"],
            "predicted_price": result["predicted_price"],
            "price_change": result["price_change"],
            "price_change_pct": result["price_change_pct"],
            "up_probability": result["up_probability_pct"],
            "down_probability": result["down_probability_pct"],
            "confidence_level": result["confidence_level"],
            "recommendation": result["recommendation"]
        }
        
        return {
            "stock_code": request.stock_code,
            "prediction": float(prediction),
            "direction": direction,
            "confidence": confidence
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train")
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    try:
        if not request.stock_codes:
            raise HTTPException(status_code=400, detail="未指定股票代码")
        
        return {
            "message": "训练功能需要在后台运行，请使用独立的训练脚本",
            "stock_codes": request.stock_codes
        }
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/enhanced")
async def predict_enhanced(request: PredictRequest):
    try:
        model_path = Path(config.model.get("model_save_path", "data/models")) / f"{request.stock_code}_model.pth"
        scaler_path = Path(config.model.get("model_save_path", "data/models")) / f"{request.stock_code}_scaler.pkl"
        
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"模型文件不存在: {model_path}，请先训练模型"
            )
        
        if not scaler_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Scaler文件不存在: {scaler_path}，请先训练模型"
            )
        
        df = db.get_stock_daily(request.stock_code)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"未找到股票 {request.stock_code} 的数据")
        
        current_price = float(df.iloc[-1]["close"])
        
        stock_feature_builder = FeatureBuilder(config.features)
        stock_feature_builder.load_scaler(str(scaler_path))
        
        X = stock_feature_builder.prepare_prediction_data(df)
        
        input_size = X.shape[2]
        model = LSTMModel(
            input_size=input_size,
            hidden_size=config.model.get("lstm", {}).get("hidden_size", 128),
            num_layers=config.model.get("lstm", {}).get("num_layers", 2),
            dropout=config.model.get("lstm", {}).get("dropout", 0.2)
        )
        
        predictor = EnhancedPredictor(model, config.system.get("device", "cpu"))
        predictor.load_model(str(model_path))
        
        use_monte_carlo = request.days is not None and request.days > 5
        
        result = predictor.get_detailed_analysis(
            X, 
            current_price,
            use_monte_carlo=use_monte_carlo
        )
        
        return {
            "stock_code": request.stock_code,
            **result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"增强预测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/info/{stock_code}")
async def get_model_info(stock_code: str):
    try:
        model_path = Path(config.model.get("model_save_path", "data/models")) / f"{stock_code}_model.pth"
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"模型文件不存在")
        
        import torch
        checkpoint = torch.load(model_path, map_location="cpu")
        
        return {
            "stock_code": stock_code,
            "model_path": str(model_path),
            "epoch": checkpoint.get("epoch", "unknown"),
            "val_loss": checkpoint.get("val_loss", "unknown"),
            "model_info": checkpoint.get("model_info", {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    host = config.api.get("host", "0.0.0.0")
    port = config.api.get("port", 8000)
    
    logger.info(f"启动API服务: {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=config.api.get("debug", False)
    )
