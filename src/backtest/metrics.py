import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BacktestMetrics:
    @staticmethod
    def calculate_all_metrics(
        daily_records: pd.DataFrame,
        trades: pd.DataFrame,
        initial_capital: float,
        risk_free_rate: float = 0.03
    ) -> Dict:
        if daily_records.empty:
            return {}
        
        final_value = daily_records['total_value'].iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        daily_records['daily_return'] = daily_records['total_value'].pct_change().fillna(0)
        
        returns = daily_records['daily_return'].values
        sharpe_ratio = BacktestMetrics.calculate_sharpe_ratio(returns, risk_free_rate)
        sortino_ratio = BacktestMetrics.calculate_sortino_ratio(returns, risk_free_rate)
        max_drawdown = BacktestMetrics.calculate_max_drawdown(daily_records['total_value'].values)
        calmar_ratio = abs(total_return / max_drawdown) if max_drawdown != 0 else 0
        
        volatility = returns.std() * np.sqrt(252)
        
        win_trades = 0
        lose_trades = 0
        total_profit = 0
        total_loss = 0
        
        if not trades.empty:
            sell_trades = trades[trades['action'] == 'SELL']
            if not sell_trades.empty:
                win_trades = len(sell_trades[sell_trades['profit'] > 0])
                lose_trades = len(sell_trades[sell_trades['profit'] <= 0])
                total_profit = sell_trades[sell_trades['profit'] > 0]['profit'].sum()
                total_loss = abs(sell_trades[sell_trades['profit'] <= 0]['profit'].sum())
        
        total_sell_trades = win_trades + lose_trades
        win_rate = win_trades / total_sell_trades if total_sell_trades > 0 else 0
        profit_loss_ratio = total_profit / total_loss if total_loss > 0 else 0
        
        avg_win = total_profit / win_trades if win_trades > 0 else 0
        avg_loss = total_loss / lose_trades if lose_trades > 0 else 0
        
        trading_days = len(daily_records)
        years = trading_days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annual_return': annual_return,
            'annual_return_pct': annual_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'volatility': volatility,
            'volatility_pct': volatility * 100,
            'total_trades': len(trades),
            'total_sell_trades': total_sell_trades,
            'win_trades': win_trades,
            'lose_trades': lose_trades,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'profit_loss_ratio': profit_loss_ratio,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'trading_days': trading_days
        }
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.03) -> float:
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        
        if excess_returns.std() == 0:
            return 0.0
        
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.03) -> float:
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
        if len(equity_curve) == 0:
            return 0.0
        
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        
        return abs(drawdown.min())
    
    @staticmethod
    def calculate_calmar_ratio(total_return: float, max_drawdown: float) -> float:
        if max_drawdown == 0:
            return 0.0
        
        return abs(total_return / max_drawdown)
    
    @staticmethod
    def calculate_win_rate(trades: pd.DataFrame) -> float:
        if trades.empty:
            return 0.0
        
        sell_trades = trades[trades['action'] == 'SELL']
        
        if sell_trades.empty:
            return 0.0
        
        win_trades = len(sell_trades[sell_trades['profit'] > 0])
        total_trades = len(sell_trades)
        
        return win_trades / total_trades
    
    @staticmethod
    def calculate_profit_factor(trades: pd.DataFrame) -> float:
        if trades.empty:
            return 0.0
        
        sell_trades = trades[trades['action'] == 'SELL']
        
        if sell_trades.empty:
            return 0.0
        
        total_profit = sell_trades[sell_trades['profit'] > 0]['profit'].sum()
        total_loss = abs(sell_trades[sell_trades['profit'] <= 0]['profit'].sum())
        
        if total_loss == 0:
            return 0.0
        
        return total_profit / total_loss
