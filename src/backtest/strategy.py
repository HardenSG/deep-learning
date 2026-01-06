from typing import Dict, Any
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseStrategy:
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
    
    def generate_signals(
        self,
        predictions: Any,
        positions: Dict,
        cash: float,
        portfolio_value: float
    ) -> Dict[str, Dict]:
        raise NotImplementedError("子类必须实现 generate_signals 方法")


class SimpleStrategy(BaseStrategy):
    def __init__(
        self,
        buy_threshold: float = 0.02,
        sell_threshold: float = -0.01,
        max_position_pct: float = 0.3
    ):
        super().__init__("SimpleStrategy")
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.max_position_pct = max_position_pct
        
        logger.info(f"策略初始化: {self.name}, 买入阈值={buy_threshold}, 卖出阈值={sell_threshold}")
    
    def generate_signals(
        self,
        prediction: float,
        positions: Dict,
        cash: float,
        portfolio_value: float
    ) -> Dict[str, Dict]:
        signals = {}
        
        stock_code = "stock"
        
        if prediction > self.buy_threshold:
            max_amount = portfolio_value * self.max_position_pct
            
            if stock_code in positions:
                current_value = positions[stock_code].market_value
                available_amount = max_amount - current_value
            else:
                available_amount = max_amount
            
            buy_amount = min(available_amount, cash * 0.95)
            
            if buy_amount > 5000:
                signals[stock_code] = {
                    'action': 'BUY',
                    'amount': buy_amount
                }
        
        elif prediction < self.sell_threshold:
            if stock_code in positions:
                signals[stock_code] = {
                    'action': 'SELL',
                    'ratio': 1.0
                }
        
        return signals


class TrendFollowingStrategy(BaseStrategy):
    def __init__(
        self,
        buy_threshold: float = 0.02,
        sell_threshold: float = -0.01,
        stop_loss: float = -0.05,
        take_profit: float = 0.10,
        max_position_pct: float = 0.3,
        position_sizing: str = "equal"
    ):
        super().__init__("TrendFollowingStrategy")
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_position_pct = max_position_pct
        self.position_sizing = position_sizing
        
        logger.info(
            f"策略初始化: {self.name}, "
            f"买入阈值={buy_threshold}, 卖出阈值={sell_threshold}, "
            f"止损={stop_loss}, 止盈={take_profit}"
        )
    
    def generate_signals(
        self,
        prediction: float,
        positions: Dict,
        cash: float,
        portfolio_value: float
    ) -> Dict[str, Dict]:
        signals = {}
        
        stock_code = "stock"
        
        if stock_code in positions:
            position = positions[stock_code]
            profit_pct = position.profit_pct
            
            if profit_pct <= self.stop_loss:
                logger.info(f"触发止损: {profit_pct*100:.2f}%")
                signals[stock_code] = {
                    'action': 'SELL',
                    'ratio': 1.0,
                    'reason': 'stop_loss'
                }
                return signals
            
            if profit_pct >= self.take_profit:
                logger.info(f"触发止盈: {profit_pct*100:.2f}%")
                signals[stock_code] = {
                    'action': 'SELL',
                    'ratio': 0.5,
                    'reason': 'take_profit'
                }
                return signals
        
        if prediction > self.buy_threshold:
            max_amount = portfolio_value * self.max_position_pct
            
            if stock_code in positions:
                current_value = positions[stock_code].market_value
                available_amount = max_amount - current_value
            else:
                available_amount = max_amount
            
            buy_amount = min(available_amount, cash * 0.95)
            
            if buy_amount > 5000:
                signals[stock_code] = {
                    'action': 'BUY',
                    'amount': buy_amount,
                    'reason': 'trend_following'
                }
        
        elif prediction < self.sell_threshold:
            if stock_code in positions:
                signals[stock_code] = {
                    'action': 'SELL',
                    'ratio': 1.0,
                    'reason': 'trend_reversal'
                }
        
        return signals


class MultiStockStrategy(BaseStrategy):
    def __init__(
        self,
        buy_threshold: float = 0.02,
        sell_threshold: float = -0.01,
        max_stocks: int = 5,
        max_position_pct: float = 0.2
    ):
        super().__init__("MultiStockStrategy")
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.max_stocks = max_stocks
        self.max_position_pct = max_position_pct
        
        logger.info(
            f"策略初始化: {self.name}, "
            f"最大持仓数={max_stocks}, 单股最大仓位={max_position_pct}"
        )
    
    def generate_signals(
        self,
        predictions: Dict[str, float],
        positions: Dict,
        cash: float,
        portfolio_value: float
    ) -> Dict[str, Dict]:
        signals = {}
        
        for stock_code in positions.keys():
            if stock_code in predictions:
                pred = predictions[stock_code]
                
                if pred < self.sell_threshold:
                    signals[stock_code] = {
                        'action': 'SELL',
                        'ratio': 1.0
                    }
        
        if len(positions) < self.max_stocks:
            buy_candidates = []
            for stock_code, pred in predictions.items():
                if pred > self.buy_threshold and stock_code not in positions:
                    buy_candidates.append((stock_code, pred))
            
            buy_candidates.sort(key=lambda x: x[1], reverse=True)
            
            available_slots = self.max_stocks - len(positions)
            for stock_code, pred in buy_candidates[:available_slots]:
                max_amount = portfolio_value * self.max_position_pct
                buy_amount = min(max_amount, cash * 0.95 / available_slots)
                
                if buy_amount > 5000:
                    signals[stock_code] = {
                        'action': 'BUY',
                        'amount': buy_amount
                    }
        
        return signals
