import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logger import setup_logger, get_logger
from src.utils.database import Database
from src.data_collector.stock_data import StockDataCollector
from src.feature_engineering.feature_builder import FeatureBuilder
from src.models.lstm_model import LSTMModel
from src.models.trainer import ModelTrainer
from src.models.predictor import ModelPredictor
import torch


def main():
    print("=" * 60)
    print("A股量化深度学习系统 - 快速开始示例")
    print("=" * 60)
    
    config = load_config()
    setup_logger(level="INFO")
    logger = get_logger(__name__)
    
    stock_code = "000001"
    
    print(f"\n1. 初始化数据库...")
    db = Database()
    
    print(f"\n2. 采集股票数据: {stock_code}")
    collector = StockDataCollector(db)
    collector.collect_stock_data(stock_code, "2020-01-01")
    
    print(f"\n3. 加载数据...")
    df = db.get_stock_daily(stock_code)
    print(f"   数据量: {len(df)} 条记录")
    
    print(f"\n4. 特征工程...")
    feature_builder = FeatureBuilder(config.features)
    X, y, feature_columns = feature_builder.prepare_training_data(df)
    print(f"   特征数量: {len(feature_columns)}")
    print(f"   样本数量: {len(X)}")
    
    print(f"\n5. 划分数据集...")
    X_train, y_train, X_val, y_val, X_test, y_test = feature_builder.split_data(X, y)
    print(f"   训练集: {len(X_train)}")
    print(f"   验证集: {len(X_val)}")
    print(f"   测试集: {len(X_test)}")
    
    print(f"\n6. 初始化LSTM模型...")
    model = LSTMModel(
        input_size=X.shape[2],
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    )
    print(f"   模型参数: {model.get_model_info()}")
    
    print(f"\n7. 训练模型...")
    trainer = ModelTrainer(model, device="cpu", learning_rate=0.001)
    
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=10,
        batch_size=32,
        early_stopping_patience=5,
        save_path=f"data/models/{stock_code}_demo_model.pth"
    )
    
    print(f"\n8. 评估模型...")
    metrics = trainer.evaluate(X_test, y_test)
    print(f"   MSE: {metrics['mse']:.6f}")
    print(f"   MAE: {metrics['mae']:.6f}")
    print(f"   方向准确率: {metrics['direction_accuracy']:.4f}")
    
    print(f"\n9. 预测...")
    predictor = ModelPredictor(model, device="cpu")
    
    X_pred = feature_builder.prepare_prediction_data(df)
    prediction = predictor.predict_single(X_pred)
    
    direction = "上涨" if prediction > 0 else "下跌"
    print(f"   预测收益率: {prediction:.4f}")
    print(f"   预测方向: {direction}")
    
    print(f"\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
