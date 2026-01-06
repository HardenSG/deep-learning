import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Position:
    def __init__(self, stock_code: str, shares: int, entry_price: float, entry_date: str):
        self.stock_code = stock_code
        self.shares = shares
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.current_price = entry_price
    
    def update_price(self, price: float):
        self.current_price = price
    
    @property
    def market_value(self) -> float:
        return self.shares * self.current_price
    
    @property
    def cost(self) -> float:
        return self.shares * self.entry_price
    
    @property
    def profit(self) -> float:
        return self.market_value - self.cost
    
    @property
    def profit_pct(self) -> float:
        return (self.current_price - self.entry_price) / self.entry_price


class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = 100000,
        commission_rate: float = 0.0003,
        slippage: float = 0.001,
        min_commission: float = 5.0
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.min_commission = min_commission
        
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Dict] = []
        self.daily_records: List[Dict] = []
        
        logger.info(f"回测引擎初始化: 初始资金={initial_capital}, 手续费率={commission_rate}, 滑点={slippage}")
    
    def calculate_commission(self, amount: float) -> float:
        commission = amount * self.commission_rate
        return max(commission, self.min_commission)
    
    def buy(self, stock_code: str, price: float, date: str, shares: int = None, amount: float = None) -> bool:
        if shares is None and amount is None:
            logger.error("必须指定买入股数或金额")
            return False
        
        buy_price = price * (1 + self.slippage)
        
        if amount is not None:
            shares = int(amount / buy_price / 100) * 100
        
        if shares < 100:
            logger.warning(f"买入股数不足100股: {shares}")
            return False
        
        total_cost = shares * buy_price
        commission = self.calculate_commission(total_cost)
        total_amount = total_cost + commission
        
        if total_amount > self.cash:
            logger.warning(f"资金不足: 需要{total_amount:.2f}, 可用{self.cash:.2f}")
            return False
        
        self.cash -= total_amount
        
        if stock_code in self.positions:
            old_pos = self.positions[stock_code]
            total_shares = old_pos.shares + shares
            avg_price = (old_pos.cost + total_cost) / total_shares
            self.positions[stock_code] = Position(stock_code, total_shares, avg_price, date)
        else:
            self.positions[stock_code] = Position(stock_code, shares, buy_price, date)
        
        self.trades.append({
            'date': date,
            'stock_code': stock_code,
            'action': 'BUY',
            'price': buy_price,
            'shares': shares,
            'amount': total_cost,
            'commission': commission,
            'cash_after': self.cash
        })
        
        logger.debug(f"买入: {stock_code} {shares}股 @{buy_price:.2f}, 手续费{commission:.2f}")
        return True
    
    def sell(self, stock_code: str, price: float, date: str, shares: int = None, ratio: float = 1.0) -> bool:
        if stock_code not in self.positions:
            logger.warning(f"没有持仓: {stock_code}")
            return False
        
        position = self.positions[stock_code]
        
        if shares is None:
            shares = int(position.shares * ratio)
        
        if shares > position.shares:
            logger.warning(f"卖出股数超过持仓: {shares} > {position.shares}")
            shares = position.shares
        
        sell_price = price * (1 - self.slippage)
        total_amount = shares * sell_price
        commission = self.calculate_commission(total_amount)
        stamp_tax = total_amount * 0.001
        
        net_amount = total_amount - commission - stamp_tax
        self.cash += net_amount
        
        profit = (sell_price - position.entry_price) * shares - commission - stamp_tax
        profit_pct = (sell_price - position.entry_price) / position.entry_price
        
        if shares == position.shares:
            del self.positions[stock_code]
        else:
            position.shares -= shares
        
        self.trades.append({
            'date': date,
            'stock_code': stock_code,
            'action': 'SELL',
            'price': sell_price,
            'shares': shares,
            'amount': total_amount,
            'commission': commission,
            'stamp_tax': stamp_tax,
            'profit': profit,
            'profit_pct': profit_pct,
            'cash_after': self.cash
        })
        
        logger.debug(f"卖出: {stock_code} {shares}股 @{sell_price:.2f}, 盈亏{profit:.2f}({profit_pct*100:.2f}%)")
        return True
    
    def update_positions(self, date: str, prices: Dict[str, float]):
        for stock_code, position in self.positions.items():
            if stock_code in prices:
                position.update_price(prices[stock_code])
    
    def get_portfolio_value(self) -> float:
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    def record_daily_status(self, date: str):
        portfolio_value = self.get_portfolio_value()
        positions_value = sum(pos.market_value for pos in self.positions.values())
        
        self.daily_records.append({
            'date': date,
            'cash': self.cash,
            'positions_value': positions_value,
            'total_value': portfolio_value,
            'return': (portfolio_value - self.initial_capital) / self.initial_capital,
            'positions_count': len(self.positions)
        })
    
    def run(
        self,
        predictions: pd.DataFrame,
        prices: pd.DataFrame,
        strategy
    ) -> Dict:
        logger.info("开始回测...")
        
        dates = predictions.index
        
        for date in dates:
            if date not in prices.index:
                continue
            
            pred_row = predictions.loc[date]
            price_row = prices.loc[date]
            
            current_prices = {}
            if isinstance(price_row, pd.Series):
                for stock_code in pred_row.index:
                    if stock_code in price_row.index:
                        current_prices[stock_code] = price_row[stock_code]
            
            self.update_positions(date, current_prices)
            
            signals = strategy.generate_signals(pred_row, self.positions, self.cash, self.get_portfolio_value())
            
            for stock_code, signal in signals.items():
                if stock_code not in current_prices:
                    continue
                
                price = current_prices[stock_code]
                
                if signal['action'] == 'BUY' and signal.get('amount', 0) > 0:
                    self.buy(stock_code, price, str(date), amount=signal['amount'])
                elif signal['action'] == 'SELL' and stock_code in self.positions:
                    self.sell(stock_code, price, str(date), ratio=signal.get('ratio', 1.0))
            
            self.record_daily_status(str(date))
        
        logger.info("回测完成")
        return self.get_results()
    
    def get_results(self) -> Dict:
        if not self.daily_records:
            return {}
        
        df_daily = pd.DataFrame(self.daily_records)
        df_trades = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        final_value = self.get_portfolio_value()
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        df_daily['daily_return'] = df_daily['total_value'].pct_change().fillna(0)
        
        returns = df_daily['daily_return'].values
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(df_daily['total_value'].values)
        
        win_trades = 0
        total_profit = 0
        total_loss = 0
        
        if len(df_trades) > 0:
            sell_trades = df_trades[df_trades['action'] == 'SELL']
            if len(sell_trades) > 0:
                win_trades = len(sell_trades[sell_trades['profit'] > 0])
                total_profit = sell_trades[sell_trades['profit'] > 0]['profit'].sum()
                total_loss = abs(sell_trades[sell_trades['profit'] < 0]['profit'].sum())
        
        win_rate = win_trades / len(df_trades[df_trades['action'] == 'SELL']) if len(df_trades[df_trades['action'] == 'SELL']) > 0 else 0
        profit_loss_ratio = total_profit / total_loss if total_loss > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'total_trades': len(df_trades),
            'win_trades': win_trades,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'profit_loss_ratio': profit_loss_ratio,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'daily_records': df_daily,
            'trades': df_trades
        }
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.03) -> float:
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        
        if excess_returns.std() == 0:
            return 0.0
        
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        if len(equity_curve) == 0:
            return 0.0
        
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        
        return abs(drawdown.min())
