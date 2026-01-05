import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.utils.config import load_config
from src.utils.database import Database
from src.feature_engineering.feature_builder import FeatureBuilder
from src.models.lstm_model import LSTMModel
from src.models.enhanced_predictor import EnhancedPredictor

def predict_stock(stock_code):
    config = load_config()
    db = Database("data/stock_data.db")
    
    df = db.get_stock_daily(stock_code)
    current_price = float(df.iloc[-1]["close"])
    
    feature_builder = FeatureBuilder(config.features)
    feature_builder.load_scaler(f"data/models/{stock_code}_scaler.pkl")
    X = feature_builder.prepare_prediction_data(df)
    
    model = LSTMModel(input_size=X.shape[2], hidden_size=128, num_layers=2, dropout=0.2)
    predictor = EnhancedPredictor(model, "cpu")
    predictor.load_model(f"data/models/{stock_code}_model.pth")
    
    result = predictor.get_detailed_analysis(X, current_price)
    
    print("=" * 60)
    print(f"股票: {stock_code}")
    print(f"当前价格: ¥{result['current_price']:.2f}")
    print(f"预测价格: ¥{result['predicted_price']:.2f} ({result['price_change_pct']:+.2f}%)")
    print("=" * 60)
    print(f"上涨概率: {result['up_probability_pct']:.1f}%")
    print(f"下跌概率: {result['down_probability_pct']:.1f}%")
    print(f"置信度: {result['confidence_level']}")
    print("=" * 60)
    rec = result['recommendation']
    print(f"💡 {rec['action']}")
    print(f"   {rec['reason']}")
    print(f"   风险: {rec['risk_level']}")
    print("=" * 60)

if __name__ == "__main__":
    predict_stock("600118")
