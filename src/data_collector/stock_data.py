import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import time
from ..utils.logger import get_logger
from ..utils.database import Database

logger = get_logger(__name__)


class StockDataCollector:
    def __init__(self, database: Database):
        self.db = database
        logger.info("股票数据采集器初始化完成")
    
    def get_stock_list(self) -> pd.DataFrame:
        try:
            logger.info("获取A股股票列表...")
            stock_info = ak.stock_info_a_code_name()
            stock_info.columns = ["stock_code", "stock_name"]
            
            logger.info(f"获取到 {len(stock_info)} 只股票")
            return stock_info
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return pd.DataFrame()
    
    def get_stock_daily_data(
        self, 
        stock_code: str, 
        start_date: str, 
        end_date: Optional[str] = None,
        adjust: str = "qfq"
    ) -> pd.DataFrame:
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        else:
            end_date = end_date.replace("-", "")
        
        start_date = start_date.replace("-", "")
        
        try:
            logger.debug(f"获取 {stock_code} 从 {start_date} 到 {end_date} 的数据...")
            
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust
            )
            
            if df.empty:
                logger.warning(f"{stock_code} 没有数据")
                return pd.DataFrame()
            
            df = self._process_daily_data(df)
            logger.info(f"获取 {stock_code} 数据成功: {len(df)} 条记录")
            
            return df
            
        except Exception as e:
            logger.error(f"获取 {stock_code} 数据失败: {e}")
            return pd.DataFrame()
    
    def _process_daily_data(self, df: pd.DataFrame) -> pd.DataFrame:
        column_mapping = {
            "日期": "trade_date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "change_pct",
            "涨跌额": "change_amount",
            "换手率": "turnover_rate"
        }
        
        df = df.rename(columns=column_mapping)
        
        required_columns = [
            "trade_date", "open", "high", "low", "close", 
            "volume", "amount"
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
        
        return df[list(column_mapping.values())]
    
    def collect_stock_data(
        self, 
        stock_code: str, 
        start_date: str, 
        end_date: Optional[str] = None,
        force_update: bool = False
    ) -> bool:
        try:
            if not force_update:
                latest_date = self.db.get_latest_date(stock_code)
                if latest_date:
                    start_date = (
                        datetime.strptime(latest_date, "%Y-%m-%d") + timedelta(days=1)
                    ).strftime("%Y-%m-%d")
                    logger.info(f"{stock_code} 增量更新，从 {start_date} 开始")
            
            df = self.get_stock_daily_data(stock_code, start_date, end_date)
            
            if df.empty:
                logger.warning(f"{stock_code} 没有新数据")
                return False
            
            self.db.save_stock_daily(stock_code, df)
            return True
            
        except Exception as e:
            logger.error(f"采集 {stock_code} 数据失败: {e}")
            return False
    
    def collect_multiple_stocks(
        self, 
        stock_codes: List[str], 
        start_date: str, 
        end_date: Optional[str] = None,
        delay: float = 0.5
    ) -> Dict[str, bool]:
        results = {}
        total = len(stock_codes)
        
        logger.info(f"开始采集 {total} 只股票的数据...")
        
        for idx, stock_code in enumerate(stock_codes, 1):
            logger.info(f"[{idx}/{total}] 采集 {stock_code}...")
            
            success = self.collect_stock_data(stock_code, start_date, end_date)
            results[stock_code] = success
            
            if idx < total:
                time.sleep(delay)
        
        success_count = sum(results.values())
        logger.info(f"采集完成: 成功 {success_count}/{total}")
        
        return results
    
    def update_all_stocks(
        self, 
        stock_codes: List[str], 
        days_back: int = 30
    ) -> Dict[str, bool]:
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"更新所有股票数据，时间范围: {start_date} 到 {end_date}")
        
        return self.collect_multiple_stocks(stock_codes, start_date, end_date)
    
    def get_realtime_data(self, stock_code: str) -> Optional[Dict]:
        try:
            df = ak.stock_zh_a_spot_em()
            stock_data = df[df["代码"] == stock_code]
            
            if stock_data.empty:
                logger.warning(f"未找到 {stock_code} 的实时数据")
                return None
            
            data = stock_data.iloc[0].to_dict()
            logger.debug(f"获取 {stock_code} 实时数据成功")
            return data
            
        except Exception as e:
            logger.error(f"获取 {stock_code} 实时数据失败: {e}")
            return None
