import sys
from pathlib import Path
import argparse
import torch

sys.path.append(str(Path(__file__).parent))

from src.utils.config import load_config
from src.utils.logger import setup_logger, get_logger
from src.utils.database import Database
from src.data_collector.stock_data import StockDataCollector
from src.feature_engineering.feature_builder import FeatureBuilder
from src.models.lstm_model import LSTMModel
from src.models.trainer import ModelTrainer


def main():
    parser = argparse.ArgumentParser(description="训练股票预测模型")
    parser.add_argument("--stock_code", type=str, required=True, help="股票代码")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--collect_data", action="store_true", help="是否采集数据")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    setup_logger(
        log_dir=config.logging.get("log_dir", "logs"),
        log_file=config.logging.get("log_file", "train.log"),
        level=config.logging.get("level", "INFO")
    )
    logger = get_logger(__name__)
    
    logger.info(f"开始训练股票 {args.stock_code} 的模型")
    
    torch.manual_seed(config.system.get("random_seed", 42))
    
    db = Database(config.data.get("database_path", "data/stock_data.db"))
    
    if args.collect_data:
        logger.info("采集数据...")
        collector = StockDataCollector(db)
        collector.collect_stock_data(
            args.stock_code,
            config.data.get("start_date", "2020-01-01"),
            config.data.get("end_date")
        )
    
    logger.info("加载数据...")
    df = db.get_stock_daily(args.stock_code)
    
    if df.empty:
        logger.error(f"未找到股票 {args.stock_code} 的数据")
        return
    
    logger.info(f"数据加载完成: {len(df)} 条记录")
    
    feature_builder = FeatureBuilder(config.features)

    logger.info("准备训练数据...")

    # 使用新的方法，避免数据泄露
    X_train, y_train, X_val, y_val, X_test, y_test, feature_columns = feature_builder.prepare_train_val_test_data(
        df,
        target_column="close",
        prediction_horizon=config.model.get("prediction", {}).get("horizon", 5),
        train_ratio=1 - config.model.get("training", {}).get("validation_split", 0.2) - config.model.get("training", {}).get("test_split", 0.1),
        val_ratio=config.model.get("training", {}).get("validation_split", 0.2)
    )

    logger.info(f"特征数量: {len(feature_columns)}")

    input_size = X_train.shape[2]
    
    logger.info("初始化模型...")
    model = LSTMModel(
        input_size=input_size,
        hidden_size=config.model.get("lstm", {}).get("hidden_size", 128),
        num_layers=config.model.get("lstm", {}).get("num_layers", 2),
        dropout=config.model.get("lstm", {}).get("dropout", 0.2),
        bidirectional=config.model.get("lstm", {}).get("bidirectional", False)
    )
    
    logger.info(f"模型信息: {model.get_model_info()}")

    # 获取损失函数配置
    loss_config = config.model.get("loss", {})
    loss_type = loss_config.get("type", "hybrid")  # 默认使用hybrid
    loss_alpha = loss_config.get("alpha", 1.0)
    loss_beta = loss_config.get("beta", 0.5)
    loss_gamma = loss_config.get("gamma", 0.3)

    trainer = ModelTrainer(
        model,
        device=config.system.get("device", "cuda"),
        learning_rate=config.model.get("training", {}).get("learning_rate", 0.001),
        weight_decay=config.model.get("training", {}).get("weight_decay", 0.0001),
        loss_type=loss_type,
        loss_alpha=loss_alpha,
        loss_beta=loss_beta,
        loss_gamma=loss_gamma
    )
    
    model_save_path = Path(config.model.get("model_save_path", "data/models")) / f"{args.stock_code}_model.pth"
    
    logger.info("开始训练...")
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=config.model.get("training", {}).get("epochs", 100),
        batch_size=config.model.get("training", {}).get("batch_size", 32),
        early_stopping_patience=config.model.get("training", {}).get("early_stopping_patience", 10),
        save_path=str(model_save_path)
    )
    
    logger.info("评估模型...")
    metrics = trainer.evaluate(X_test, y_test)
    
    logger.info(f"测试集评估结果: {metrics}")
    
    scaler_path = Path(config.model.get("model_save_path", "data/models")) / f"{args.stock_code}_scaler.pkl"
    feature_builder.save_scaler(str(scaler_path))
    
    logger.info(f"训练完成! 模型保存至: {model_save_path}")
    logger.info(f"Scaler保存至: {scaler_path}")


if __name__ == "__main__":
    main()
