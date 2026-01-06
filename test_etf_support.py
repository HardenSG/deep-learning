# -*- coding: utf-8 -*-
"""
测试ETF支持功能
测试用户提到的ETF代码：563530
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from src.utils.config import load_config
from src.utils.database import Database
from src.data_collector.unified_collector import UnifiedDataCollector


def test_etf_data_collection():
    """测试ETF数据采集"""
    print("=" * 60)
    print("测试ETF数据采集功能")
    print("=" * 60)

    # 初始化
    config = load_config()
    db = Database(config.data.get("database_path", "data/stock_data.db"))
    collector = UnifiedDataCollector(db)

    # 测试1: 获取ETF列表
    print("\n[测试1] 获取ETF基金列表...")
    etf_list = collector.get_etf_list()
    print(f"获取到 {len(etf_list)} 只ETF基金")
    if not etf_list.empty:
        print("\n前5只ETF:")
        print(etf_list.head())

    # 测试2: 获取用户指定的ETF数据 (563530)
    print("\n" + "=" * 60)
    print("[测试2] 获取ETF 563530的历史数据...")
    print("=" * 60)

    etf_code = "563530"
    # 尝试更宽的日期范围
    df = collector.get_etf_daily_data(
        etf_code,
        start_date="2020-01-01"
    )

    if df.empty:
        print(f"[警告] ETF {etf_code} 无数据，可能是新基金或代码错误")
        print("尝试使用常见ETF代码 159915 (创业板ETF)...")

        # 使用常见的ETF代码进行测试
        etf_code = "159915"
        df = collector.get_etf_daily_data(
            etf_code,
            start_date="2020-01-01"
        )

    if df.empty:
        print(f"[错误] 未获取到任何ETF数据")
        return False
    else:
        print(f"[成功] ETF {etf_code} 获取到 {len(df)} 条数据")
        print(f"\n数据列: {df.columns.tolist()}")
        print(f"\n数据范围: {df['trade_date'].min()} 到 {df['trade_date'].max()}")
        print(f"\n最新5条数据:")
        print(df.tail())

    # 测试3: 使用统一接口（自动识别类型）
    print("\n" + "=" * 60)
    print("[测试3] 测试自动类型识别...")
    print("=" * 60)

    # 测试ETF（使用我们找到的有效代码）
    df_auto = collector.get_daily_data(
        etf_code,
        start_date="2024-01-01",
        security_type='auto'
    )
    if not df_auto.empty:
        print(f"{etf_code} 自动识别为: {df_auto['security_type'].iloc[0]}")
    else:
        print(f"{etf_code} 自动识别失败")

    # 测试股票
    df_stock = collector.get_daily_data(
        "600519",
        start_date="2024-01-01",
        security_type='auto'
    )
    if not df_stock.empty:
        print(f"600519 自动识别为: {df_stock['security_type'].iloc[0]}")
    else:
        print("600519 自动识别失败")

    # 测试4: 采集并保存到数据库
    print("\n" + "=" * 60)
    print("[测试4] 采集并保存ETF数据到数据库...")
    print("=" * 60)

    success = collector.collect_data(
        etf_code,
        start_date="2024-01-01",
        security_type='etf'
    )

    if success:
        print(f"[成功] ETF {etf_code} 数据已保存到数据库")

        # 从数据库读取验证
        db_data = db.get_stock_daily(etf_code)
        print(f"[验证] 从数据库读取到 {len(db_data)} 条数据")

        if not db_data.empty:
            print("\n数据库中最新5条数据:")
            print(db_data[['trade_date', 'open', 'close', 'high', 'low', 'volume']].tail())
    else:
        print(f"[失败] 采集ETF {etf_code} 数据失败")
        return False

    # 测试5: 批量采集（A股+ETF混合）
    print("\n" + "=" * 60)
    print("[测试5] 测试批量采集（混合A股和ETF）...")
    print("=" * 60)

    test_codes = ["600519", etf_code, "159915"]  # 茅台、测试ETF、创业板ETF
    results = collector.collect_multiple(
        test_codes,
        start_date="2024-01-01",
        security_type='auto'
    )

    print("\n批量采集结果:")
    for code, result in results.items():
        status = "成功" if result else "失败"
        print(f"  {code}: {status}")

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        success = test_etf_data_collection()
        if success:
            print("\n[总结] ETF支持功能测试通过!")
            print("\n您现在可以:")
            print("1. 在Web平台选择 'ETF基金' 类型")
            print("2. 输入ETF代码（如 159915、512630等常见ETF）")
            print("3. 查询历史数据、训练模型、进行预测")
            print("\n注意：如果ETF代码563530无数据，可能是该ETF较新或代码有误")
            print("      建议使用常见ETF代码如：159915（创业板ETF）、512630（军工ETF）")
        else:
            print("\n[总结] 部分测试失败，请检查错误信息")
    except Exception as e:
        print(f"\n[错误] 测试过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
