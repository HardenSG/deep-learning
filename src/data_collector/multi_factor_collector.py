"""
多因子数据采集器

采集资金流向、北向资金、舆情情绪、宏观政策等多维度因子数据
"""
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import time
import re
from ..utils.logger import get_logger

logger = get_logger(__name__)

# 尝试导入增强版情绪分析
try:
    from ..nlp.sentiment_analyzer import SentimentAnalyzer
    USE_ENHANCED_SENTIMENT = True
except ImportError:
    USE_ENHANCED_SENTIMENT = False


class MultiFactorCollector:
    """
    多因子数据采集器

    支持的因子类型:
    1. 资金流向因子 - 主力资金、散户资金流向
    2. 北向资金因子 - 沪股通、深股通资金流向
    3. 舆情情绪因子 - 新闻情绪、讨论热度
    4. 宏观政策因子 - LPR、CPI、PMI、GDP、社融等
    """

    def __init__(self):
        logger.info("多因子数据采集器初始化完成")
        self._cache = {}  # 缓存宏观数据避免重复请求

        # 初始化增强版情绪分析器
        if USE_ENHANCED_SENTIMENT:
            self.sentiment_analyzer = SentimentAnalyzer()
            logger.info("增强版情绪分析器已启用")
        else:
            self.sentiment_analyzer = None

    # ==================== 资金流向因子 ====================

    def get_individual_fund_flow(
        self,
        stock_code: str,
        market: str = 'auto'
    ) -> pd.DataFrame:
        """
        获取个股资金流向数据

        Args:
            stock_code: 股票代码
            market: 市场 ('sh'/'sz'/'bj'/'auto')

        Returns:
            DataFrame: 包含主力、超大单、大单、中单、小单资金流向
        """
        try:
            # 自动判断市场
            if market == 'auto':
                if stock_code.startswith('6'):
                    market = 'sh'
                elif stock_code.startswith(('0', '3')):
                    market = 'sz'
                else:
                    market = 'bj'

            logger.debug(f"获取 {stock_code} 资金流向数据...")
            df = ak.stock_individual_fund_flow(stock=stock_code, market=market)

            if df.empty:
                logger.warning(f"未获取到 {stock_code} 的资金流向数据")
                return df

            # 标准化列名
            df.columns = [
                'trade_date', 'close', 'change_pct',
                'main_net_inflow', 'main_net_inflow_pct',
                'super_large_net_inflow', 'super_large_net_inflow_pct',
                'large_net_inflow', 'large_net_inflow_pct',
                'medium_net_inflow', 'medium_net_inflow_pct',
                'small_net_inflow', 'small_net_inflow_pct'
            ]

            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df['stock_code'] = stock_code

            # 转换数值类型
            numeric_cols = [col for col in df.columns if col not in ['trade_date', 'stock_code']]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            logger.info(f"获取到 {stock_code} 的 {len(df)} 条资金流向数据")
            return df

        except Exception as e:
            logger.error(f"获取 {stock_code} 资金流向失败: {e}")
            return pd.DataFrame()

    def get_market_fund_flow(self) -> pd.DataFrame:
        """
        获取大盘资金流向数据

        Returns:
            DataFrame: 沪深两市整体资金流向
        """
        try:
            logger.debug("获取大盘资金流向数据...")
            df = ak.stock_market_fund_flow()

            if df.empty:
                return df

            # 标准化列名
            df.columns = [
                'trade_date',
                'sh_main_net_inflow', 'sh_main_net_inflow_pct',
                'sh_super_large_net_inflow', 'sh_super_large_net_inflow_pct',
                'sh_large_net_inflow', 'sh_large_net_inflow_pct',
                'sh_medium_net_inflow', 'sh_medium_net_inflow_pct',
                'sh_small_net_inflow', 'sh_small_net_inflow_pct',
                'sz_main_net_inflow', 'sz_main_net_inflow_pct',
                'sz_super_large_net_inflow', 'sz_super_large_net_inflow_pct',
                'sz_large_net_inflow', 'sz_large_net_inflow_pct',
                'sz_medium_net_inflow', 'sz_medium_net_inflow_pct',
                'sz_small_net_inflow', 'sz_small_net_inflow_pct'
            ]

            df['trade_date'] = pd.to_datetime(df['trade_date'])

            logger.info(f"获取到 {len(df)} 条大盘资金流向数据")
            return df

        except Exception as e:
            logger.error(f"获取大盘资金流向失败: {e}")
            return pd.DataFrame()

    def get_sector_fund_flow(self, sector_type: str = 'industry') -> pd.DataFrame:
        """
        获取板块资金流向

        Args:
            sector_type: 'industry'(行业) 或 'concept'(概念)

        Returns:
            DataFrame: 板块资金流向排名
        """
        try:
            logger.debug(f"获取{sector_type}板块资金流向...")

            if sector_type == 'industry':
                df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")
            else:
                df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="概念资金流")

            if df.empty:
                return df

            logger.info(f"获取到 {len(df)} 个板块资金流向数据")
            return df

        except Exception as e:
            logger.error(f"获取板块资金流向失败: {e}")
            return pd.DataFrame()

    # ==================== 北向资金因子 ====================

    def get_north_fund_flow(self, symbol: str = "北向资金") -> pd.DataFrame:
        """
        获取北向资金历史数据

        Args:
            symbol: "北向资金", "沪股通", "深股通", "南向资金", "港股通沪", "港股通深"

        Returns:
            DataFrame: 北向/南向资金流向历史
        """
        try:
            logger.debug(f"获取{symbol}历史数据...")
            df = ak.stock_hsgt_hist_em(symbol=symbol)

            if df.empty:
                return df

            # 根据实际列名进行处理
            # 实际列名: ['日期', '当日成交净买额', '当日余额', '历史累计净买额', ...]
            df = df.rename(columns={
                '日期': 'trade_date',
                '当日成交净买额': 'net_buy_amount',
                '买入成交额': 'buy_amount',
                '卖出成交额': 'sell_amount',
                '历史累计净买额': 'cumulative_net_buy'
            })

            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df['fund_type'] = symbol

            # 转换数值
            for col in ['net_buy_amount', 'buy_amount', 'sell_amount', 'cumulative_net_buy']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            logger.info(f"获取到 {len(df)} 条{symbol}历史数据")
            return df

        except Exception as e:
            logger.error(f"获取{symbol}数据失败: {e}")
            return pd.DataFrame()

    def get_north_hold_stock(self, stock_code: str) -> pd.DataFrame:
        """
        获取北向资金持股数据（个股）

        Args:
            stock_code: 股票代码

        Returns:
            DataFrame: 北向资金持股历史
        """
        try:
            logger.debug(f"获取 {stock_code} 北向持股数据...")

            # 判断是沪股通还是深股通
            if stock_code.startswith('6'):
                indicator = "沪股通"
            else:
                indicator = "深股通"

            df = ak.stock_hsgt_individual_em(symbol=stock_code)

            if df.empty:
                logger.warning(f"未获取到 {stock_code} 的北向持股数据")
                return df

            df['stock_code'] = stock_code
            df['trade_date'] = pd.to_datetime(df.iloc[:, 0])

            logger.info(f"获取到 {stock_code} 的 {len(df)} 条北向持股数据")
            return df

        except Exception as e:
            logger.error(f"获取 {stock_code} 北向持股数据失败: {e}")
            return pd.DataFrame()

    # ==================== 舆情情绪因子 ====================

    def get_stock_news(self, stock_code: str, limit: int = 100) -> pd.DataFrame:
        """
        获取个股新闻数据

        Args:
            stock_code: 股票代码
            limit: 获取条数

        Returns:
            DataFrame: 新闻数据（标题、时间、来源）
        """
        try:
            logger.debug(f"获取 {stock_code} 新闻数据...")
            df = ak.stock_news_em(symbol=stock_code)

            if df.empty:
                return df

            # 根据实际列名进行处理
            # 实际列名: ['关键词', '新闻标题', '新闻内容', '发布时间', '文章来源', '新闻链接']
            df = df.rename(columns={
                '关键词': 'keyword',
                '新闻标题': 'title',
                '新闻内容': 'content',
                '发布时间': 'publish_time',
                '文章来源': 'source',
                '新闻链接': 'url'
            })

            df['stock_code'] = stock_code
            df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')

            # 简单情绪分析（基于关键词）
            df['sentiment_score'] = df['title'].apply(self._analyze_sentiment)

            logger.info(f"获取到 {stock_code} 的 {len(df)} 条新闻")
            return df.head(limit)

        except Exception as e:
            logger.error(f"获取 {stock_code} 新闻失败: {e}")
            return pd.DataFrame()

    def _analyze_sentiment(self, text: str) -> float:
        """
        情绪分析（优先使用增强版）

        Args:
            text: 文本内容

        Returns:
            float: 情绪得分 (-1 到 1)
        """
        if not text or pd.isna(text):
            return 0.0

        # 使用增强版情绪分析器
        if self.sentiment_analyzer:
            result = self.sentiment_analyzer.analyze(text)
            return result['score']

        # 回退到简单关键词分析
        return self._simple_sentiment(text)

    def _simple_sentiment(self, text: str) -> float:
        """简单的情绪分析（基于关键词）"""
        if not text or pd.isna(text):
            return 0.0

        # 正面词汇
        positive_words = [
            '涨停', '大涨', '暴涨', '突破', '创新高', '利好', '增长', '盈利',
            '超预期', '获批', '中标', '签约', '合作', '扩产', '回购', '增持',
            '看好', '推荐', '买入', '强势', '反弹', '拉升', '放量', '主力',
            '龙头', '核心', '领涨', '爆发', '翻倍', '新高', '突破压力'
        ]

        # 负面词汇
        negative_words = [
            '跌停', '大跌', '暴跌', '下跌', '创新低', '利空', '亏损', '下滑',
            '不及预期', '处罚', '违规', '调查', '退市', '减持', '清仓', '质押',
            '爆雷', '暴雷', '风险', '警示', '下调', '看空', '卖出', '弱势',
            '破位', '跳水', '闪崩', '割肉', '套牢', '腰斩', '跌破支撑'
        ]

        text = str(text)
        score = 0.0

        for word in positive_words:
            if word in text:
                score += 0.15

        for word in negative_words:
            if word in text:
                score -= 0.15

        return max(-1.0, min(1.0, score))

    def get_news_sentiment_daily(
        self,
        stock_code: str,
        start_date: str,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        计算每日新闻情绪得分

        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame: 每日情绪汇总
        """
        try:
            news_df = self.get_stock_news(stock_code, limit=500)

            if news_df.empty:
                return pd.DataFrame()

            # 过滤日期
            start_dt = pd.to_datetime(start_date)
            if end_date:
                end_dt = pd.to_datetime(end_date)
            else:
                end_dt = datetime.now()

            news_df = news_df[
                (news_df['publish_time'] >= start_dt) &
                (news_df['publish_time'] <= end_dt)
            ]

            if news_df.empty:
                return pd.DataFrame()

            # 按日期聚合
            news_df['trade_date'] = news_df['publish_time'].dt.date
            daily_sentiment = news_df.groupby('trade_date').agg({
                'sentiment_score': ['mean', 'std', 'count'],
                'title': 'count'
            }).reset_index()

            daily_sentiment.columns = [
                'trade_date', 'sentiment_mean', 'sentiment_std',
                'sentiment_count', 'news_count'
            ]

            daily_sentiment['trade_date'] = pd.to_datetime(daily_sentiment['trade_date'])
            daily_sentiment['stock_code'] = stock_code

            # 填充NaN
            daily_sentiment['sentiment_std'] = daily_sentiment['sentiment_std'].fillna(0)

            logger.info(f"计算得到 {stock_code} 的 {len(daily_sentiment)} 天情绪数据")
            return daily_sentiment

        except Exception as e:
            logger.error(f"计算 {stock_code} 每日情绪失败: {e}")
            return pd.DataFrame()

    def get_cctv_news(self) -> pd.DataFrame:
        """
        获取央视新闻联播文字稿（政策风向标）

        Returns:
            DataFrame: 新闻联播内容
        """
        try:
            logger.debug("获取新闻联播数据...")
            df = ak.news_cctv(date=datetime.now().strftime("%Y%m%d"))

            if df.empty:
                return df

            logger.info(f"获取到 {len(df)} 条新闻联播内容")
            return df

        except Exception as e:
            logger.error(f"获取新闻联播失败: {e}")
            return pd.DataFrame()

    # ==================== 宏观政策因子 ====================

    def get_lpr_rate(self) -> pd.DataFrame:
        """
        获取LPR利率数据

        Returns:
            DataFrame: LPR利率历史
        """
        try:
            if 'lpr' in self._cache:
                return self._cache['lpr']

            logger.debug("获取LPR利率数据...")
            df = ak.macro_china_lpr()

            if df.empty:
                return df

            # 标准化
            df['trade_date'] = pd.to_datetime(df.iloc[:, 0])
            df['lpr_1y'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
            df['lpr_5y'] = pd.to_numeric(df.iloc[:, 2], errors='coerce')

            df = df[['trade_date', 'lpr_1y', 'lpr_5y']]

            self._cache['lpr'] = df
            logger.info(f"获取到 {len(df)} 条LPR利率数据")
            return df

        except Exception as e:
            logger.error(f"获取LPR利率失败: {e}")
            return pd.DataFrame()

    def get_cpi_data(self) -> pd.DataFrame:
        """
        获取CPI数据

        Returns:
            DataFrame: CPI历史数据
        """
        try:
            if 'cpi' in self._cache:
                return self._cache['cpi']

            logger.debug("获取CPI数据...")
            df = ak.macro_china_cpi_yearly()

            if df.empty:
                return df

            # 实际列名: ['商品', '日期', '今值', '预测值', '前值']
            df = df.rename(columns={
                '日期': 'trade_date',
                '今值': 'cpi'
            })

            df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
            df['cpi'] = pd.to_numeric(df['cpi'], errors='coerce')

            df = df[['trade_date', 'cpi']].dropna()

            self._cache['cpi'] = df
            logger.info(f"获取到 {len(df)} 条CPI数据")
            return df

        except Exception as e:
            logger.error(f"获取CPI数据失败: {e}")
            return pd.DataFrame()

    def get_pmi_data(self) -> pd.DataFrame:
        """
        获取PMI数据

        Returns:
            DataFrame: PMI历史数据
        """
        try:
            if 'pmi' in self._cache:
                return self._cache['pmi']

            logger.debug("获取PMI数据...")

            # 制造业PMI
            df_man = ak.index_pmi_man_cx()
            df_man['pmi_type'] = 'manufacturing'

            # 服务业PMI
            df_ser = ak.index_pmi_ser_cx()
            df_ser['pmi_type'] = 'services'

            # 合并
            df = pd.concat([df_man, df_ser], ignore_index=True)

            self._cache['pmi'] = df
            logger.info(f"获取到 {len(df)} 条PMI数据")
            return df

        except Exception as e:
            logger.error(f"获取PMI数据失败: {e}")
            return pd.DataFrame()

    def get_m2_data(self) -> pd.DataFrame:
        """
        获取M2货币供应量数据

        Returns:
            DataFrame: M2历史数据
        """
        try:
            if 'm2' in self._cache:
                return self._cache['m2']

            logger.debug("获取M2数据...")
            df = ak.macro_china_m2_yearly()

            if df.empty:
                return df

            # 实际列名: ['商品', '日期', '今值', '预测值', '前值']
            df = df.rename(columns={
                '日期': 'trade_date',
                '今值': 'm2_yoy'
            })

            df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
            df['m2_yoy'] = pd.to_numeric(df['m2_yoy'], errors='coerce')

            df = df[['trade_date', 'm2_yoy']].dropna()

            self._cache['m2'] = df
            logger.info(f"获取到 {len(df)} 条M2数据")
            return df

        except Exception as e:
            logger.error(f"获取M2数据失败: {e}")
            return pd.DataFrame()

    def get_gdp_data(self) -> pd.DataFrame:
        """
        获取GDP数据

        Returns:
            DataFrame: GDP历史数据
        """
        try:
            if 'gdp' in self._cache:
                return self._cache['gdp']

            logger.debug("获取GDP数据...")
            df = ak.macro_china_gdp_yearly()

            if df.empty:
                return df

            self._cache['gdp'] = df
            logger.info(f"获取到 {len(df)} 条GDP数据")
            return df

        except Exception as e:
            logger.error(f"获取GDP数据失败: {e}")
            return pd.DataFrame()

    def get_shibor_rate(self) -> pd.DataFrame:
        """
        获取SHIBOR利率数据

        Returns:
            DataFrame: SHIBOR历史数据
        """
        try:
            logger.debug("获取SHIBOR数据...")
            df = ak.rate_interbank(market="上海银行同业拆借市场", symbol="Shibor人民币", indicator="隔夜")

            if df.empty:
                return df

            logger.info(f"获取到 {len(df)} 条SHIBOR数据")
            return df

        except Exception as e:
            logger.error(f"获取SHIBOR数据失败: {e}")
            return pd.DataFrame()

    def get_ppi_data(self) -> pd.DataFrame:
        """获取PPI数据"""
        try:
            if 'ppi' in self._cache:
                return self._cache['ppi']

            logger.debug("获取PPI数据...")
            df = ak.macro_china_ppi_yearly()

            if df.empty:
                return df

            df = df.rename(columns={'日期': 'trade_date', '今值': 'ppi'})
            df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
            df['ppi'] = pd.to_numeric(df['ppi'], errors='coerce')
            df = df[['trade_date', 'ppi']].dropna()

            self._cache['ppi'] = df
            logger.info(f"获取到 {len(df)} 条PPI数据")
            return df

        except Exception as e:
            logger.error(f"获取PPI数据失败: {e}")
            return pd.DataFrame()

    def get_social_finance(self) -> pd.DataFrame:
        """获取社会融资规模数据"""
        try:
            if 'social_finance' in self._cache:
                return self._cache['social_finance']

            logger.debug("获取社融数据...")
            df = ak.macro_china_shrzgm()

            if df.empty:
                return df

            # 标准化
            df = df.rename(columns={'月份': 'trade_date', '社会融资规模增量': 'social_finance'})
            df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')

            self._cache['social_finance'] = df
            logger.info(f"获取到 {len(df)} 条社融数据")
            return df

        except Exception as e:
            logger.error(f"获取社融数据失败: {e}")
            return pd.DataFrame()

    def get_margin_trading(self, stock_code: str = None) -> pd.DataFrame:
        """
        获取融资融券数据

        Args:
            stock_code: 股票代码，None则获取市场整体

        Returns:
            DataFrame: 融资融券数据
        """
        try:
            logger.debug("获取融资融券数据...")

            if stock_code:
                # 个股融资融券
                df = ak.stock_margin_detail_szse(date=datetime.now().strftime("%Y%m%d"))
            else:
                # 市场整体融资融券
                df = ak.stock_margin_szse()

            if df.empty:
                return df

            logger.info(f"获取到融资融券数据")
            return df

        except Exception as e:
            logger.error(f"获取融资融券数据失败: {e}")
            return pd.DataFrame()

    def get_market_sentiment_index(self) -> pd.DataFrame:
        """获取市场情绪指标（恐慌贪婪指数相关）"""
        try:
            logger.debug("获取市场情绪指标...")

            # 获取两市成交量作为市场情绪代理
            df = ak.stock_zh_a_hist(
                symbol="000001",  # 上证指数
                period="daily",
                start_date=(datetime.now() - timedelta(days=365)).strftime("%Y%m%d"),
                adjust=""
            )

            if df.empty:
                return df

            df = df.rename(columns={
                '日期': 'trade_date',
                '成交量': 'market_volume',
                '成交额': 'market_amount',
                '涨跌幅': 'market_change_pct'
            })

            df['trade_date'] = pd.to_datetime(df['trade_date'])

            # 计算成交量相对强度
            df['volume_ma5'] = df['market_volume'].rolling(5).mean()
            df['volume_ma20'] = df['market_volume'].rolling(20).mean()
            df['volume_ratio'] = df['market_volume'] / df['volume_ma20']

            # 计算波动率
            df['volatility_20d'] = df['market_change_pct'].rolling(20).std()

            logger.info(f"获取到 {len(df)} 条市场情绪数据")
            return df[['trade_date', 'market_volume', 'volume_ratio', 'volatility_20d']]

        except Exception as e:
            logger.error(f"获取市场情绪指标失败: {e}")
            return pd.DataFrame()

    # ==================== 综合因子采集 ====================

    def collect_all_factors(
        self,
        stock_code: str,
        start_date: str,
        end_date: Optional[str] = None,
        include_sentiment: bool = True,
        include_macro: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        采集个股的全部因子数据

        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            include_sentiment: 是否包含舆情因子
            include_macro: 是否包含宏观因子

        Returns:
            Dict: 各类因子数据
        """
        results = {}

        # 1. 资金流向因子
        logger.info(f"采集 {stock_code} 资金流向因子...")
        fund_flow = self.get_individual_fund_flow(stock_code)
        if not fund_flow.empty:
            results['fund_flow'] = fund_flow
        time.sleep(0.3)

        # 2. 北向资金因子（整体）
        logger.info("采集北向资金因子...")
        north_flow = self.get_north_fund_flow("北向资金")
        if not north_flow.empty:
            results['north_flow'] = north_flow
        time.sleep(0.3)

        # 3. 北向持股因子（个股）
        logger.info(f"采集 {stock_code} 北向持股因子...")
        north_hold = self.get_north_hold_stock(stock_code)
        if not north_hold.empty:
            results['north_hold'] = north_hold
        time.sleep(0.3)

        # 4. 舆情情绪因子
        if include_sentiment:
            logger.info(f"采集 {stock_code} 舆情情绪因子...")
            sentiment = self.get_news_sentiment_daily(stock_code, start_date, end_date)
            if not sentiment.empty:
                results['sentiment'] = sentiment
            time.sleep(0.3)

        # 5. 宏观政策因子
        if include_macro:
            logger.info("采集宏观政策因子...")

            lpr = self.get_lpr_rate()
            if not lpr.empty:
                results['lpr'] = lpr

            cpi = self.get_cpi_data()
            if not cpi.empty:
                results['cpi'] = cpi

            m2 = self.get_m2_data()
            if not m2.empty:
                results['m2'] = m2

        logger.info(f"因子采集完成，共 {len(results)} 类因子数据")
        return results

    def merge_factors_with_price(
        self,
        price_df: pd.DataFrame,
        factors: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        将多因子数据与价格数据合并

        Args:
            price_df: 价格数据（必须包含trade_date列）
            factors: 因子数据字典

        Returns:
            DataFrame: 合并后的数据
        """
        if price_df.empty:
            return price_df

        result = price_df.copy()
        result['trade_date'] = pd.to_datetime(result['trade_date'])

        # 1. 合并资金流向因子
        if 'fund_flow' in factors and not factors['fund_flow'].empty:
            fund_flow = factors['fund_flow'].copy()
            fund_flow['trade_date'] = pd.to_datetime(fund_flow['trade_date'])

            # 选择需要的列
            fund_cols = [
                'trade_date', 'main_net_inflow', 'main_net_inflow_pct',
                'super_large_net_inflow', 'large_net_inflow',
                'medium_net_inflow', 'small_net_inflow'
            ]
            fund_cols = [c for c in fund_cols if c in fund_flow.columns]
            fund_flow = fund_flow[fund_cols]

            result = result.merge(fund_flow, on='trade_date', how='left')
            logger.debug("合并资金流向因子完成")

        # 2. 合并北向资金因子（整体市场）
        if 'north_flow' in factors and not factors['north_flow'].empty:
            north_flow = factors['north_flow'].copy()
            north_flow['trade_date'] = pd.to_datetime(north_flow['trade_date'])

            # 选择需要的列并重命名
            north_cols = ['trade_date', 'net_buy_amount']
            north_flow = north_flow[north_cols].rename(
                columns={'net_buy_amount': 'north_net_buy'}
            )

            result = result.merge(north_flow, on='trade_date', how='left')
            logger.debug("合并北向资金因子完成")

        # 3. 合并舆情因子
        if 'sentiment' in factors and not factors['sentiment'].empty:
            sentiment = factors['sentiment'].copy()
            sentiment['trade_date'] = pd.to_datetime(sentiment['trade_date'])

            sentiment_cols = [
                'trade_date', 'sentiment_mean', 'sentiment_std', 'news_count'
            ]
            sentiment_cols = [c for c in sentiment_cols if c in sentiment.columns]
            sentiment = sentiment[sentiment_cols]

            result = result.merge(sentiment, on='trade_date', how='left')
            logger.debug("合并舆情因子完成")

        # 4. 合并LPR因子（需要前向填充）
        if 'lpr' in factors and not factors['lpr'].empty:
            lpr = factors['lpr'].copy()
            lpr['trade_date'] = pd.to_datetime(lpr['trade_date'])

            result = result.merge(lpr, on='trade_date', how='left')
            # LPR是月度数据，需要前向填充
            result['lpr_1y'] = result['lpr_1y'].ffill()
            result['lpr_5y'] = result['lpr_5y'].ffill()
            logger.debug("合并LPR因子完成")

        # 5. 合并CPI因子（月度数据，前向填充）
        if 'cpi' in factors and not factors['cpi'].empty:
            cpi = factors['cpi'].copy()
            cpi['trade_date'] = pd.to_datetime(cpi['trade_date'])

            result = result.merge(cpi, on='trade_date', how='left')
            result['cpi'] = result['cpi'].ffill()
            logger.debug("合并CPI因子完成")

        # 6. 合并M2因子
        if 'm2' in factors and not factors['m2'].empty:
            m2 = factors['m2'].copy()
            m2['trade_date'] = pd.to_datetime(m2['trade_date'])

            result = result.merge(m2, on='trade_date', how='left')
            result['m2_yoy'] = result['m2_yoy'].ffill()
            logger.debug("合并M2因子完成")

        # 填充NaN（对于新增的因子列）
        factor_cols = [
            'main_net_inflow', 'main_net_inflow_pct',
            'super_large_net_inflow', 'large_net_inflow',
            'medium_net_inflow', 'small_net_inflow',
            'north_net_buy', 'sentiment_mean', 'sentiment_std', 'news_count'
        ]

        for col in factor_cols:
            if col in result.columns:
                result[col] = result[col].fillna(0)

        logger.info(f"因子合并完成，共 {len(result.columns)} 列")
        return result

    def clear_cache(self):
        """清除缓存的宏观数据"""
        self._cache = {}
        logger.info("宏观数据缓存已清除")
