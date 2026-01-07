"""
测试多因子采集器
"""
import sys
sys.path.insert(0, '.')

from src.data_collector.multi_factor_collector import MultiFactorCollector
import pandas as pd

def test_fund_flow():
    """测试资金流向因子"""
    print("\n" + "="*50)
    print("测试1: 资金流向因子")
    print("="*50)

    collector = MultiFactorCollector()

    # 测试个股资金流向
    df = collector.get_individual_fund_flow("600519")  # 贵州茅台

    if not df.empty:
        print(f"获取到 {len(df)} 条资金流向数据")
        print(f"列名: {list(df.columns)}")
        print(f"最近5条数据:\n{df.head()}")
        return True
    else:
        print("未获取到数据")
        return False


def test_north_flow():
    """测试北向资金因子"""
    print("\n" + "="*50)
    print("测试2: 北向资金因子")
    print("="*50)

    collector = MultiFactorCollector()

    # 测试北向资金历史
    df = collector.get_north_fund_flow("北向资金")

    if not df.empty:
        print(f"获取到 {len(df)} 条北向资金数据")
        print(f"列名: {list(df.columns)}")
        print(f"最近5条数据:\n{df.head()}")
        return True
    else:
        print("未获取到数据")
        return False


def test_sentiment():
    """测试舆情情绪因子"""
    print("\n" + "="*50)
    print("测试3: 舆情情绪因子")
    print("="*50)

    collector = MultiFactorCollector()

    # 测试个股新闻
    df = collector.get_stock_news("600519", limit=10)

    if not df.empty:
        print(f"获取到 {len(df)} 条新闻数据")
        print(f"列名: {list(df.columns)}")
        print(f"新闻标题和情绪得分:")
        for _, row in df.iterrows():
            print(f"  [{row['sentiment_score']:.2f}] {row['title'][:50]}...")
        return True
    else:
        print("未获取到数据")
        return False


def test_macro():
    """测试宏观政策因子"""
    print("\n" + "="*50)
    print("测试4: 宏观政策因子")
    print("="*50)

    collector = MultiFactorCollector()

    # 测试LPR
    print("\n--- LPR利率 ---")
    lpr = collector.get_lpr_rate()
    if not lpr.empty:
        print(f"获取到 {len(lpr)} 条LPR数据")
        print(f"最近5条:\n{lpr.tail()}")

    # 测试CPI
    print("\n--- CPI ---")
    cpi = collector.get_cpi_data()
    if not cpi.empty:
        print(f"获取到 {len(cpi)} 条CPI数据")
        print(f"最近5条:\n{cpi.tail()}")

    # 测试M2
    print("\n--- M2 ---")
    m2 = collector.get_m2_data()
    if not m2.empty:
        print(f"获取到 {len(m2)} 条M2数据")
        print(f"最近5条:\n{m2.tail()}")

    return True


def test_collect_all():
    """测试全部因子采集"""
    print("\n" + "="*50)
    print("测试5: 全部因子采集 (600519)")
    print("="*50)

    collector = MultiFactorCollector()

    factors = collector.collect_all_factors(
        stock_code="600519",
        start_date="2024-01-01",
        include_sentiment=True,
        include_macro=True
    )

    print(f"\n采集到 {len(factors)} 类因子:")
    for name, df in factors.items():
        if not df.empty:
            print(f"  - {name}: {len(df)} 条记录, 列: {list(df.columns)[:5]}...")
        else:
            print(f"  - {name}: 空")

    return len(factors) > 0


def test_merge_factors():
    """测试因子合并"""
    print("\n" + "="*50)
    print("测试6: 因子合并到价格数据")
    print("="*50)

    import akshare as ak

    # 获取价格数据
    print("获取价格数据...")
    price_df = ak.stock_zh_a_hist(
        symbol="600519",
        period="daily",
        start_date="20240101",
        end_date="20240301",
        adjust="qfq"
    )

    # 根据实际列名进行重命名
    # 实际列名: ['日期', '股票代码', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
    price_df = price_df.rename(columns={
        '日期': 'trade_date',
        '股票代码': 'stock_code',
        '开盘': 'open',
        '收盘': 'close',
        '最高': 'high',
        '最低': 'low',
        '成交量': 'volume',
        '成交额': 'turnover',
        '振幅': 'amplitude',
        '涨跌幅': 'change_pct',
        '涨跌额': 'change_amount',
        '换手率': 'turnover_rate'
    })
    price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])

    print(f"价格数据: {len(price_df)} 条")

    # 采集因子
    collector = MultiFactorCollector()
    factors = collector.collect_all_factors(
        stock_code="600519",
        start_date="2024-01-01",
        end_date="2024-03-01",
        include_sentiment=False,  # 舆情数据可能没有这么久
        include_macro=True
    )

    # 合并
    merged = collector.merge_factors_with_price(price_df, factors)

    print(f"\n合并后数据: {len(merged)} 条, {len(merged.columns)} 列")
    print(f"新增列: {[c for c in merged.columns if c not in price_df.columns]}")
    print(f"\n前5条数据:\n{merged.head()}")

    return True


if __name__ == "__main__":
    print("="*60)
    print("多因子采集器测试")
    print("="*60)

    results = {}

    # 运行测试
    try:
        results['资金流向'] = test_fund_flow()
    except Exception as e:
        print(f"资金流向测试失败: {e}")
        results['资金流向'] = False

    try:
        results['北向资金'] = test_north_flow()
    except Exception as e:
        print(f"北向资金测试失败: {e}")
        results['北向资金'] = False

    try:
        results['舆情情绪'] = test_sentiment()
    except Exception as e:
        print(f"舆情情绪测试失败: {e}")
        results['舆情情绪'] = False

    try:
        results['宏观政策'] = test_macro()
    except Exception as e:
        print(f"宏观政策测试失败: {e}")
        results['宏观政策'] = False

    try:
        results['全部采集'] = test_collect_all()
    except Exception as e:
        print(f"全部采集测试失败: {e}")
        results['全部采集'] = False

    try:
        results['因子合并'] = test_merge_factors()
    except Exception as e:
        print(f"因子合并测试失败: {e}")
        results['因子合并'] = False

    # 汇总
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)

    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {name}: {status}")

    total = len(results)
    passed = sum(results.values())
    print(f"\n总计: {passed}/{total} 通过")
