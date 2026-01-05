import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logger import setup_logger, get_logger
from src.utils.database import Database
from src.data_collector.stock_data import StockDataCollector


def main():
    config = load_config()
    
    setup_logger(
        log_dir=config.logging.get("log_dir", "logs"),
        log_file="data_collection.log",
        level=config.logging.get("level", "INFO")
    )
    logger = get_logger(__name__)
    
    logger.info("开始采集股票数据...")
    
    db = Database(config.data.get("database_path", "data/stock_data.db"))
    collector = StockDataCollector(db)
    
    stock_codes = config.data.get("stock_list", [])
    start_date = config.data.get("start_date", "2020-01-01")
    end_date = config.data.get("end_date")
    
    logger.info(f"股票数量: {len(stock_codes)}")
    logger.info(f"时间范围: {start_date} 到 {end_date or '今天'}")
    
    results = collector.collect_multiple_stocks(stock_codes, start_date, end_date)
    
    success_count = sum(results.values())
    logger.info(f"采集完成: 成功 {success_count}/{len(stock_codes)}")
    
    for stock_code, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"{status} {stock_code}")


if __name__ == "__main__":
    main()
