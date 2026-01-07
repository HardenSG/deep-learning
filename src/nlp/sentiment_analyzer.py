"""
增强版情绪分析模块

提供基于关键词和规则的中文金融文本情绪分析
"""
import re
from typing import List, Dict, Tuple
import pandas as pd


class SentimentAnalyzer:
    """
    金融文本情绪分析器

    支持:
    1. 关键词情绪分析
    2. 否定词处理
    3. 程度副词加权
    4. 金融专业术语识别
    """

    def __init__(self):
        # 正面词汇及权重
        self.positive_words = {
            # 强烈正面 (0.3)
            '涨停': 0.3, '暴涨': 0.3, '飙升': 0.3, '狂飙': 0.3, '井喷': 0.3,
            '翻倍': 0.3, '翻番': 0.3, '创历史新高': 0.3, '史上最高': 0.3,
            # 中等正面 (0.2)
            '大涨': 0.2, '上涨': 0.15, '上扬': 0.15, '走高': 0.15, '攀升': 0.2,
            '突破': 0.2, '创新高': 0.2, '新高': 0.15, '反弹': 0.15, '拉升': 0.2,
            '放量': 0.1, '强势': 0.15, '领涨': 0.2, '龙头': 0.15,
            # 基本正面 (0.1)
            '利好': 0.15, '增长': 0.1, '盈利': 0.15, '超预期': 0.2, '业绩增': 0.15,
            '获批': 0.1, '中标': 0.15, '签约': 0.1, '合作': 0.1, '扩产': 0.1,
            '回购': 0.15, '增持': 0.15, '买入': 0.1, '推荐': 0.1, '看好': 0.1,
            '景气': 0.1, '复苏': 0.1, '向好': 0.1, '改善': 0.1, '提升': 0.1,
            '突破压力': 0.15, '站上': 0.1, '收复': 0.1,
            # 资金相关
            '主力流入': 0.15, '资金流入': 0.1, '北向买入': 0.15, '外资增持': 0.15,
            '融资买入': 0.1, '机构加仓': 0.15,
        }

        # 负面词汇及权重
        self.negative_words = {
            # 强烈负面 (-0.3)
            '跌停': -0.3, '暴跌': -0.3, '崩盘': -0.3, '闪崩': -0.3, '熔断': -0.3,
            '腰斩': -0.3, '爆雷': -0.3, '暴雷': -0.3, '退市': -0.3,
            # 中等负面 (-0.2)
            '大跌': -0.2, '下跌': -0.15, '下挫': -0.15, '走低': -0.15, '下滑': -0.15,
            '跳水': -0.2, '杀跌': -0.2, '破位': -0.2, '跌破': -0.15, '创新低': -0.2,
            '缩量': -0.1, '弱势': -0.15, '领跌': -0.2,
            # 基本负面 (-0.1)
            '利空': -0.15, '亏损': -0.15, '不及预期': -0.2, '业绩下滑': -0.15,
            '处罚': -0.15, '违规': -0.15, '调查': -0.1, '减持': -0.15, '清仓': -0.2,
            '质押': -0.1, '爆仓': -0.25, '平仓': -0.15,
            '警示': -0.1, '风险': -0.1, '下调': -0.1, '看空': -0.15, '卖出': -0.1,
            '衰退': -0.15, '萎缩': -0.1, '恶化': -0.15,
            '跌破支撑': -0.15, '失守': -0.1,
            # 资金相关
            '主力流出': -0.15, '资金流出': -0.1, '北向卖出': -0.15, '外资减持': -0.15,
            '融资卖出': -0.1, '机构减仓': -0.15,
        }

        # 否定词
        self.negation_words = [
            '不', '没', '无', '非', '未', '别', '莫', '勿', '否',
            '没有', '不是', '不会', '不能', '未能', '难以',
        ]

        # 程度副词及权重
        self.degree_words = {
            # 强化 (1.5-2.0)
            '极其': 2.0, '极度': 2.0, '非常': 1.8, '特别': 1.7, '十分': 1.6,
            '相当': 1.5, '很': 1.5, '太': 1.5, '超': 1.5, '巨': 1.8,
            '大幅': 1.6, '显著': 1.5, '明显': 1.4, '急剧': 1.7,
            # 弱化 (0.5-0.8)
            '略': 0.6, '稍': 0.6, '略微': 0.5, '稍微': 0.5, '有点': 0.6,
            '小幅': 0.7, '微': 0.5, '轻微': 0.5,
        }

        # 金融实体模式
        self.entity_patterns = {
            'stock': r'[036]\d{5}',  # 股票代码
            'percent': r'[+-]?\d+\.?\d*%',  # 百分比
            'money': r'\d+\.?\d*[万亿]',  # 金额
        }

    def analyze(self, text: str) -> Dict:
        """
        分析文本情绪

        Args:
            text: 输入文本

        Returns:
            Dict: {
                'score': 情绪得分 (-1 到 1),
                'label': 情绪标签 (positive/negative/neutral),
                'confidence': 置信度,
                'details': 详细信息
            }
        """
        if not text or pd.isna(text):
            return {
                'score': 0.0,
                'label': 'neutral',
                'confidence': 0.0,
                'details': {}
            }

        text = str(text)

        # 分词处理（简单按字符滑动窗口）
        score = 0.0
        matched_positive = []
        matched_negative = []

        # 检查否定词位置
        negation_positions = []
        for neg in self.negation_words:
            pos = text.find(neg)
            while pos != -1:
                negation_positions.append((pos, pos + len(neg)))
                pos = text.find(neg, pos + 1)

        # 检查程度副词
        degree_multiplier = 1.0
        for deg, mult in self.degree_words.items():
            if deg in text:
                degree_multiplier = max(degree_multiplier, mult)

        # 正面词汇匹配
        for word, weight in self.positive_words.items():
            if word in text:
                pos = text.find(word)
                # 检查前面是否有否定词
                is_negated = any(
                    neg_start < pos < neg_start + 10
                    for neg_start, neg_end in negation_positions
                )
                if is_negated:
                    score -= weight * degree_multiplier
                    matched_negative.append(f"不{word}")
                else:
                    score += weight * degree_multiplier
                    matched_positive.append(word)

        # 负面词汇匹配
        for word, weight in self.negative_words.items():
            if word in text:
                pos = text.find(word)
                is_negated = any(
                    neg_start < pos < neg_start + 10
                    for neg_start, neg_end in negation_positions
                )
                if is_negated:
                    score -= weight * degree_multiplier  # 双重否定变正面
                    matched_positive.append(f"不{word}")
                else:
                    score += weight * degree_multiplier  # weight本身是负数
                    matched_negative.append(word)

        # 限制得分范围
        score = max(-1.0, min(1.0, score))

        # 确定标签
        if score > 0.1:
            label = 'positive'
        elif score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'

        # 计算置信度
        total_matches = len(matched_positive) + len(matched_negative)
        confidence = min(1.0, total_matches * 0.2 + abs(score))

        return {
            'score': round(score, 4),
            'label': label,
            'confidence': round(confidence, 4),
            'details': {
                'positive_matches': matched_positive,
                'negative_matches': matched_negative,
                'degree_multiplier': degree_multiplier
            }
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """批量分析"""
        return [self.analyze(text) for text in texts]

    def get_daily_sentiment(self, news_df: pd.DataFrame, date_col: str = 'publish_time', text_col: str = 'title') -> pd.DataFrame:
        """
        计算每日情绪汇总

        Args:
            news_df: 新闻数据
            date_col: 日期列名
            text_col: 文本列名

        Returns:
            DataFrame: 每日情绪统计
        """
        if news_df.empty:
            return pd.DataFrame()

        # 分析每条新闻
        results = self.analyze_batch(news_df[text_col].tolist())

        news_df = news_df.copy()
        news_df['sentiment_score'] = [r['score'] for r in results]
        news_df['sentiment_label'] = [r['label'] for r in results]
        news_df['sentiment_confidence'] = [r['confidence'] for r in results]

        # 按日期聚合
        news_df['date'] = pd.to_datetime(news_df[date_col]).dt.date

        daily = news_df.groupby('date').agg({
            'sentiment_score': ['mean', 'std', 'min', 'max', 'count'],
            'sentiment_confidence': 'mean'
        }).reset_index()

        daily.columns = [
            'trade_date', 'sentiment_mean', 'sentiment_std',
            'sentiment_min', 'sentiment_max', 'news_count', 'confidence_mean'
        ]

        daily['trade_date'] = pd.to_datetime(daily['trade_date'])

        # 计算情绪动量
        daily['sentiment_momentum'] = daily['sentiment_mean'].diff()

        # 正负面新闻占比
        positive_counts = news_df[news_df['sentiment_label'] == 'positive'].groupby('date').size()
        negative_counts = news_df[news_df['sentiment_label'] == 'negative'].groupby('date').size()

        daily['positive_ratio'] = daily['trade_date'].dt.date.map(
            lambda d: positive_counts.get(d, 0) / max(1, positive_counts.get(d, 0) + negative_counts.get(d, 0))
        )

        return daily


class FinancialNER:
    """
    金融命名实体识别（简单规则版）
    """

    def __init__(self):
        self.patterns = {
            'stock_code': r'[036]\d{5}',
            'stock_name': r'[\u4e00-\u9fa5]{2,4}(?:股份|集团|银行|证券|保险)',
            'money': r'\d+(?:\.\d+)?[万亿]?元?',
            'percent': r'[+-]?\d+(?:\.\d+)?%',
            'date': r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?',
        }

    def extract(self, text: str) -> Dict[str, List[str]]:
        """提取金融实体"""
        entities = {}
        for name, pattern in self.patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                entities[name] = matches
        return entities
