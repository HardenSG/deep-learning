import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import time
from ..utils.logger import get_logger
from ..utils.database import Database

logger = get_logger(__name__)


class UnifiedDataCollector:
    """
    统一数据采集器 - 支持A股和ETF基金

    使用示例:
        collector = UnifiedDataCollector(db)

        # 采集A股
        collector.collect_data('600519', security_type='stock')

        # 采集ETF
        collector.collect_data('563530', security_type='etf')
    """

    def __init__(self, database: Database):
        self.db = database
        logger.info("统一数据采集器初始化完成（支持A股+ETF）")

    def get_stock_list(self) -> pd.DataFrame:
        """获取A股列表"""
        try:
            logger.info("获取A股股票列表...")
            stock_info = ak.stock_info_a_code_name()
            stock_info.columns = ["code", "name"]
            stock_info['type'] = 'stock'

            logger.info(f"获取到 {len(stock_info)} 只A股")
            return stock_info
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return pd.DataFrame()

    def get_etf_list(self) -> pd.DataFrame:
        """获取ETF基金列表"""
        try:
            logger.info("获取ETF基金列表...")
            etf_info = ak.fund_etf_spot_em()

            # 提取需要的列
            etf_info = etf_info[['代码', '名称']].copy()
            etf_info.columns = ["code", "name"]
            etf_info['type'] = 'etf'

            logger.info(f"获取到 {len(etf_info)} 只ETF")
            return etf_info
        except Exception as e:
            logger.error(f"获取ETF列表失败: {e}")
            return pd.DataFrame()

    def get_all_securities_list(self) -> pd.DataFrame:
        """获取所有证券列表（A股+ETF）"""
        stock_list = self.get_stock_list()
        etf_list = self.get_etf_list()

        all_securities = pd.concat([stock_list, etf_list], ignore_index=True)
        logger.info(f"共获取 {len(all_securities)} 只证券（股票+ETF）")

        return all_securities

    def get_stock_daily_data(
        self,
        stock_code: str,
        start_date: str,
        end_date: Optional[str] = None,
        adjust: str = "qfq"
    ) -> pd.DataFrame:
        """获取A股日线数据"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        else:
            end_date = end_date.replace("-", "")

        start_date = start_date.replace("-", "")

        try:
            logger.debug(f"获取A股 {stock_code} 从 {start_date} 到 {end_date} 的数据...")

            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust
            )

            if df.empty:
                logger.warning(f"未获取到 {stock_code} 的数据")
                return df

            # 标准化列名 - 灵活处理不同数量的列
            # akshare可能返回不同数量的列，我们只重命名已知的前N列
            expected_cols = ['trade_date', 'open', 'close', 'high', 'low', 'volume',
                           'turnover', 'amplitude', 'change_pct', 'change_amount',
                           'turnover_rate']

            # 只重命名我们需要的列（前11列）
            num_cols_to_rename = min(len(df.columns), len(expected_cols))
            new_col_names = expected_cols[:num_cols_to_rename]

            # 如果有额外的列，保留原列名
            if len(df.columns) > num_cols_to_rename:
                new_col_names = new_col_names + df.columns[num_cols_to_rename:].tolist()

            df.columns = new_col_names

            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df['stock_code'] = stock_code
            df['security_type'] = 'stock'

            logger.info(f"获取到 {stock_code} 的 {len(df)} 条数据")
            return df

        except Exception as e:
            logger.error(f"获取A股 {stock_code} 数据失败: {e}")
            return pd.DataFrame()

    def get_etf_daily_data(
        self,
        etf_code: str,
        start_date: str,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取ETF日线数据"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        else:
            end_date = end_date.replace("-", "")

        start_date = start_date.replace("-", "")

        try:
            logger.debug(f"获取ETF {etf_code} 从 {start_date} 到 {end_date} 的数据...")

            # 使用东方财富接口获取ETF历史数据
            df = ak.fund_etf_hist_em(
                symbol=etf_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"  # 前复权
            )

            if df.empty:
                logger.warning(f"未获取到ETF {etf_code} 的数据")
                return df

            # 标准化列名（与股票保持一致）
            df.columns = ['trade_date', 'open', 'close', 'high', 'low', 'volume',
                         'turnover', 'amplitude', 'change_pct', 'change_amount',
                         'turnover_rate']

            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df['stock_code'] = etf_code  # 保持字段名一致
            df['security_type'] = 'etf'

            logger.info(f"获取到ETF {etf_code} 的 {len(df)} 条数据")
            return df

        except Exception as e:
            logger.error(f"获取ETF {etf_code} 数据失败: {e}")
            return pd.DataFrame()

    def get_daily_data(
        self,
        code: str,
        start_date: str,
        end_date: Optional[str] = None,
        security_type: str = 'auto'
    ) -> pd.DataFrame:
        """
        统一接口：获取日线数据（自动识别类型）

        Args:
            code: 证券代码
            start_date: 开始日期
            end_date: 结束日期
            security_type: 'stock', 'etf', 或 'auto'（自动识别）

        Returns:
            DataFrame: 标准化的日线数据
        """
        if security_type == 'auto':
            # 自动识别类型
            if code.startswith('5') or code.startswith('15'):
                security_type = 'etf'
                logger.info(f"自动识别 {code} 为ETF基金")
            else:
                security_type = 'stock'
                logger.info(f"自动识别 {code} 为A股")

        if security_type == 'etf':
            return self.get_etf_daily_data(code, start_date, end_date)
        else:
            return self.get_stock_daily_data(code, start_date, end_date)

    def collect_data(
        self,
        code: str,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        security_type: str = 'auto'
    ) -> bool:
        """
        采集并保存数据

        Args:
            code: 证券代码
            start_date: 开始日期
            end_date: 结束日期
            security_type: 'stock', 'etf', 或 'auto'

        Returns:
            bool: 是否成功
        """
        try:
            df = self.get_daily_data(code, start_date, end_date, security_type)

            if df.empty:
                logger.error(f"未获取到 {code} 的数据")
                return False

            # 准备保存到数据库 - 需要进行列名映射以匹配数据库schema
            df_to_save = df.copy()

            # 列名映射：turnover -> amount（数据库中使用amount字段）
            if 'turnover' in df_to_save.columns:
                df_to_save = df_to_save.rename(columns={'turnover': 'amount'})

            # 数据库schema中的有效列（不包括id和created_at）
            valid_columns = [
                'trade_date', 'open', 'high', 'low', 'close', 'volume',
                'amount', 'amplitude', 'change_pct', 'change_amount', 'turnover_rate'
            ]

            # 只保留数据库schema中存在的列，删除其他所有列
            cols_to_keep = [col for col in valid_columns if col in df_to_save.columns]
            df_to_save = df_to_save[cols_to_keep]

            self.db.save_stock_daily(code, df_to_save)

            logger.info(f"成功保存 {code} 的 {len(df)} 条数据到数据库")
            return True

        except Exception as e:
            logger.error(f"采集 {code} 数据失败: {e}")
            return False

    def collect_multiple(
        self,
        codes: List[str],
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        security_type: str = 'auto'
    ) -> Dict[str, bool]:
        """
        批量采集数据

        Returns:
            Dict[code, success]: 每个代码的采集结果
        """
        results = {}
        total = len(codes)

        logger.info(f"开始批量采集 {total} 只证券的数据...")

        for i, code in enumerate(codes, 1):
            logger.info(f"[{i}/{total}] 正在采集 {code}...")

            try:
                success = self.collect_data(code, start_date, end_date, security_type)
                results[code] = success

                # 避免请求过快
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"采集 {code} 时出错: {e}")
                results[code] = False

        success_count = sum(results.values())
        logger.info(f"批量采集完成: 成功 {success_count}/{total}")

        return results

    def update_data(
        self,
        code: str,
        days_back: int = 30,
        security_type: str = 'auto'
    ) -> bool:
        """
        更新最近N天的数据

        Args:
            code: 证券代码
            days_back: 更新最近几天
            security_type: 证券类型

        Returns:
            bool: 是否成功
        """
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        return self.collect_data(code, start_date, security_type=security_type)


# 保持向后兼容
class StockDataCollector(UnifiedDataCollector):
    """
    股票数据采集器（向后兼容）

    现在继承自UnifiedDataCollector，支持股票和ETF
    """
    def __init__(self, database: Database):
        super().__init__(database)
        logger.info("股票数据采集器初始化完成（已升级支持ETF）")

    def collect_stock_data(
        self,
        stock_code: str,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None
    ) -> bool:
        """保持原有接口兼容性"""
        return self.collect_data(stock_code, start_date, end_date, security_type='auto')
