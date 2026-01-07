# 普通基金支持方案 - 解决方案文档

## 问题分析

### 根本原因
- **003494** (富国天惠成长混合(LOF)C) 和 **001056** (华银健康生活主题灵活配置) 是**普通开放式基金**，不是ETF
- 现有的 `UnifiedDataCollector` 只支持A股和ETF，使用的API无法获取普通基金数据

### 测试结果
✅ akshare **完全支持**普通基金数据获取
- 基金列表: 25,950只基金
- 003494: 2,144条净值数据 (2017-03-23 至今)
- 001056: 2,586条净值数据 (2015-03-27 至今)

---

## 解决方案

### 方案1: 使用akshare扩展功能 (推荐)

#### 优点
- ✅ 免费、开源
- ✅ 无需token
- ✅ 数据完整 (2000+条历史净值)
- ✅ 与现有系统兼容

#### 实现
已创建 `unified_collector_v2.py`，新增功能:

```python
# 1. 自动识别证券类型
collector = UnifiedDataCollectorV2()
df = collector.get_daily_data('003494', '2020-01-01', security_type='auto')
# 自动识别为基金并获取数据

# 2. 支持三种类型
collector.get_daily_data(code, start_date, security_type='stock')  # A股
collector.get_daily_data(code, start_date, security_type='etf')    # ETF
collector.get_daily_data(code, start_date, security_type='fund')   # 基金
```

#### 核心API
```python
# akshare基金接口
ak.fund_name_em()  # 获取所有基金列表
ak.fund_open_fund_info_em(
    symbol='003494',
    indicator='单位净值走势',
    period='全部'
)
```

#### 数据格式统一
基金净值数据会被转换为与股票兼容的格式:
- `单位净值` → `close` (收盘价)
- `日增长率` → `change_pct`
- OHLV字段填充为净值 (基金无盘中价格)

---

### 方案2: Tushare Pro

#### 优点
- 数据质量高、更新及时
- 支持股票+基金+期货等
- 官方API，长期稳定

#### 缺点
- 需要注册token (免费)
- 有积分和频率限制
- 基金历史数据需要一定积分

#### 接口
```python
import tushare as ts
ts.set_token('你的token')
pro = ts.pro_api()

# 基金净值
df = pro.fund_nav(ts_code='003494.OF')

# 基金基本信息
df = pro.fund_basic(market='O')  # O=开放式
```

注册: https://tushare.pro/register

---

### 方案3: efinance (东方财富)

#### 优点
- 完全免费
- 无需token
- 支持基金实时行情

#### 缺点
- 非官方API
- 稳定性不如官方接口

#### 安装使用
```bash
pip install efinance

# Python代码
import efinance as ef
df = ef.fund.get_quote_history('003494')
```

---

## 推荐实施步骤

### 立即可用方案 (0成本)

使用已创建的 `unified_collector_v2.py`:

```python
from src.data_collector.unified_collector_v2 import UnifiedDataCollectorV2

# 初始化
collector = UnifiedDataCollectorV2(database)

# 采集基金数据
df = collector.get_daily_data(
    code='003494',
    start_date='2020-01-01',
    security_type='auto'  # 自动识别
)

# 批量采集
codes = ['003494', '001056', '600519', '563530']
for code in codes:
    collector.get_daily_data(code, '2020-01-01', security_type='auto')
```

### 集成到现有系统

1. **替换采集器** (unified_collector.py → unified_collector_v2.py)
2. **保持接口兼容** (现有代码无需修改)
3. **自动类型识别** (代码自动判断股票/ETF/基金)

---

## 数据平台对比

| 平台 | 股票 | ETF | 基金 | 费用 | Token | 限制 |
|------|------|-----|------|------|-------|------|
| **akshare** | ✅ | ✅ | ✅ | 免费 | 否 | 无 |
| **tushare** | ✅ | ✅ | ✅ | 免费 | 是 | 积分 |
| **efinance** | ✅ | ✅ | ✅ | 免费 | 否 | 稳定性 |
| **聚宽** | ✅ | ✅ | ✅ | 付费 | 是 | 贵 |
| **Wind** | ✅ | ✅ | ✅ | 付费 | 是 | 非常贵 |

**结论**: akshare是最适合个人项目的选择

---

## 关键差异说明

### 股票 vs 基金数据结构

**股票/ETF (OHLCV)**:
- Open (开盘价)
- High (最高价)
- Low (最低价)
- Close (收盘价)
- Volume (成交量)

**基金 (NAV)**:
- 单位净值 (每日一个值)
- 累计净值
- 日增长率

### 在V2中的处理
```python
# 基金净值映射为收盘价
{
    'close': 单位净值,
    'open': 单位净值,   # 基金无盘中价格
    'high': 单位净值,
    'low': 单位净值,
    'volume': 0,        # 基金无成交量
    'change_pct': 日增长率
}
```

这样做的好处:
- ✅ 与现有模型训练代码兼容
- ✅ 技术指标计算正常工作 (基于close价格)
- ✅ 数据库结构无需修改

---

## 测试结果

### 测试文件
- `test_stock_codes.py` - 验证代码不在股票/ETF列表
- `test_fund_support.py` - 测试各平台基金支持
- `test_fund_api.py` - 验证akshare基金API
- `unified_collector_v2.py` - 完整的扩展采集器

### 测试数据
```
003494 (富国天惠成长混合C):
  ✅ 2,144条净值数据
  ✅ 日期范围: 2017-03-23 至 2026-01-07
  ✅ 最新净值: 2.8925

001056 (华银健康生活主题):
  ✅ 2,586条净值数据
  ✅ 日期范围: 2015-03-27 至 2026-01-07
  ✅ 最新净值: 1.361
```

---

## 下一步建议

### 短期 (立即可用)
1. 使用 `unified_collector_v2.py` 采集基金数据
2. 测试模型训练效果
3. 验证预测准确性

### 中期 (优化)
1. 考虑是否需要区分基金和股票的预测模型
2. 评估基金特有指标 (如同类排名)
3. 添加基金持仓分析

### 长期 (扩展)
1. 支持债券基金、货币基金
2. 集成Tushare获取更多基础数据
3. 添加基金评级、风险指标

---

## 常见问题

**Q: 基金数据能用于训练模型吗?**
A: 可以！基金净值被映射为close价格，技术指标计算正常，可以直接训练。

**Q: 预测效果会好吗?**
A: 基金波动比股票小，预测相对容易，但收益也更稳定。

**Q: 需要修改现有代码吗?**
A: 只需替换采集器，其他代码无需修改。V2采集器向后兼容。

**Q: 支持哪些基金类型?**
A: 开放式基金 (混合型、股票型、债券型等)，暂不支持封闭式基金。

---

## 联系与反馈

如有问题，请参考:
- akshare文档: https://akshare.akfamily.xyz/
- tushare文档: https://tushare.pro/document/2
- 本项目issues

**创建时间**: 2026-01-08
**版本**: 1.0
