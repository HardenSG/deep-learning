import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from .logger import get_logger

logger = get_logger(__name__)


class Database:
    def __init__(self, db_path: str = "data/stock_data.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        self._create_tables()
        logger.info(f"数据库初始化完成: {self.db_path}")
    
    def _create_tables(self):
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS stock_info (
                    stock_code TEXT PRIMARY KEY,
                    stock_name TEXT,
                    industry TEXT,
                    market TEXT,
                    list_date TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS stock_daily (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_code TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    open RE         AL,
           high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    amount REAL,
                    amplitude REAL,
                    change_pct REAL,
                    change_amount REAL,
                    turnover_rate REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(stock_code, trade_date)
                )
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_stock_daily_code_date 
                ON stock_daily(stock_code, trade_date)
            """))
            
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS stock_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_code TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    feature_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(stock_code, trade_date)
                )
            """))
            
            conn.commit()
    
    @contextmanager
    def get_session(self) -> Session:
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"数据库会话错误: {e}")
            raise
        finally:
            session.close()
    
    def save_stock_info(self, stock_info: pd.DataFrame) -> None:
        try:
            stock_info.to_sql("stock_info", self.engine, if_exists="replace", index=False)
            logger.info(f"保存股票信息: {len(stock_info)} 条记录")
        except Exception as e:
            logger.error(f"保存股票信息失败: {e}")
            raise
    
    def save_stock_daily(self, stock_code: str, daily_data: pd.DataFrame) -> None:
        try:
            daily_data = daily_data.copy()
            daily_data["stock_code"] = stock_code
            
            if 'trade_date' in daily_data.columns:
                daily_data['trade_date'] = pd.to_datetime(daily_data['trade_date']).dt.strftime('%Y-%m-%d')
            
            with self.engine.connect() as conn:
                for _, row in daily_data.iterrows():
                    trade_date = row.get('trade_date')
                    if isinstance(trade_date, pd.Timestamp):
                        trade_date = trade_date.strftime('%Y-%m-%d')
                    
                    conn.execute(text("""
                        INSERT OR REPLACE INTO stock_daily 
                        (stock_code, trade_date, open, high, low, close, volume, amount, 
                         amplitude, change_pct, change_amount, turnover_rate)
                        VALUES 
                        (:stock_code, :trade_date, :open, :high, :low, :close, :volume, :amount,
                         :amplitude, :change_pct, :change_amount, :turnover_rate)
                    """), {
                        'stock_code': str(row.get('stock_code')),
                        'trade_date': str(trade_date),
                        'open': float(row.get('open')) if pd.notna(row.get('open')) else None,
                        'high': float(row.get('high')) if pd.notna(row.get('high')) else None,
                        'low': float(row.get('low')) if pd.notna(row.get('low')) else None,
                        'close': float(row.get('close')) if pd.notna(row.get('close')) else None,
                        'volume': float(row.get('volume')) if pd.notna(row.get('volume')) else None,
                        'amount': float(row.get('amount')) if pd.notna(row.get('amount')) else None,
                        'amplitude': float(row.get('amplitude')) if pd.notna(row.get('amplitude')) else None,
                        'change_pct': float(row.get('change_pct')) if pd.notna(row.get('change_pct')) else None,
                        'change_amount': float(row.get('change_amount')) if pd.notna(row.get('change_amount')) else None,
                        'turnover_rate': float(row.get('turnover_rate')) if pd.notna(row.get('turnover_rate')) else None
                    })
                conn.commit()
            
            logger.info(f"保存 {stock_code} 日线数据: {len(daily_data)} 条记录")
        except Exception as e:
            logger.error(f"保存 {stock_code} 日线数据失败: {e}")
            raise
    
    def get_stock_daily(
        self, 
        stock_code: str, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        query = f"SELECT * FROM stock_daily WHERE stock_code = '{stock_code}'"
        
        if start_date:
            query += f" AND trade_date >= '{start_date}'"
        if end_date:
            query += f" AND trade_date <= '{end_date}'"
        
        query += " ORDER BY trade_date"
        
        try:
            df = pd.read_sql(query, self.engine)
            logger.debug(f"查询 {stock_code} 日线数据: {len(df)} 条记录")
            return df
        except Exception as e:
            logger.error(f"查询 {stock_code} 日线数据失败: {e}")
            return pd.DataFrame()
    
    def get_latest_date(self, stock_code: str) -> Optional[str]:
        query = f"""
            SELECT MAX(trade_date) as latest_date 
            FROM stock_daily 
            WHERE stock_code = '{stock_code}'
        """
        
        try:
            result = pd.read_sql(query, self.engine)
            if not result.empty and result["latest_date"].iloc[0]:
                return result["latest_date"].iloc[0]
            return None
        except Exception as e:
            logger.error(f"查询 {stock_code} 最新日期失败: {e}")
            return None
    
    def delete_stock_data(self, stock_code: str) -> None:
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f"DELETE FROM stock_daily WHERE stock_code = '{stock_code}'"))
                conn.commit()
            logger.info(f"删除 {stock_code} 的所有数据")
        except Exception as e:
            logger.error(f"删除 {stock_code} 数据失败: {e}")
            raise
    
    def execute_query(self, query: str) -> pd.DataFrame:
        try:
            return pd.read_sql(query, self.engine)
        except Exception as e:
            logger.error(f"执行查询失败: {e}")
            return pd.DataFrame()
