# A股量化深度学习系统

基于Python和LSTM深度学习模型的A股市场量化交易系统，提供自动化数据采集、特征工程、模型训练和预测API服务。

## ✨ 主要功能

- 📊 **自动化数据采集**: 使用akshare获取A股市场数据，支持3000+股票
- 🔧 **丰富的技术指标**: 30+技术指标特征工程（MA、MACD、RSI、KDJ、布林带等）
- 🤖 **LSTM深度学习**: 基于PyTorch的LSTM/注意力机制模型
- 📈 **智能预测**: 预测未来N天的股票收益率和涨跌方向
- 🚀 **REST API**: FastAPI提供高性能API服务
- 🐳 **Docker部署**: 一键容器化部署
- ⏰ **定时任务**: 自动化数据更新和模型训练

## 📋 系统要求

- Python 3.8+
- CUDA (可选，用于GPU加速)
- Docker & Docker Compose (用于容器化部署)

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd 金融量化
```

### 2. 安装依赖

#### 方式一：使用虚拟环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

#### 方式二：使用Docker

```bash
docker-compose up -d
```

### 3. 配置系统

编辑 `config/config.yaml` 文件，配置股票列表、模型参数等：

```yaml
data:
  stock_list:
    - "000001"  # 平安银行
    - "600519"  # 贵州茅台
    # 添加更多股票...
```

### 4. 采集数据

```bash
python -c "
from src.utils.config import load_config
from src.utils.database import Database
from src.data_collector.stock_data import StockDataCollector

config = load_config()
db = Database()
collector = StockDataCollector(db)

# 采集单只股票
collector.collect_stock_data('000001', '2020-01-01')

# 批量采集
stock_codes = config.data.get('stock_list', [])
collector.collect_multiple_stocks(stock_codes, '2020-01-01')
"
```

### 5. 训练模型

```bash
# 训练单只股票模型
python train.py --stock_code 000001 --collect_data

# 参数说明:
# --stock_code: 股票代码
# --collect_data: 是否先采集数据
# --config: 配置文件路径 (默认: config/config.yaml)
```

### 6. 启动API服务

```bash
# 直接运行
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 或使用Docker
docker-compose up -d
```

访问 http://localhost:8000/docs 查看API文档

## 📚 API使用示例

### 获取股票列表

```bash
curl http://localhost:8000/api/stocks
```

### 更新股票数据

```bash
curl -X POST http://localhost:8000/api/data/update \
  -H "Content-Type: application/json" \
  -d '{
    "stock_codes": ["000001", "600519"],
    "days_back": 30
  }'
```

### 获取股票历史数据

```bash
curl http://localhost:8000/api/data/000001?limit=100
```

### 预测股票走势

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "stock_code": "000001",
    "days": 5
  }'
```

### 获取模型信息

```bash
curl http://localhost:8000/api/model/info/000001
```

## 🏗️ 项目结构

```
金融量化/
├── config/                 # 配置文件
│   └── config.yaml
├── data/                   # 数据目录
│   ├── raw/               # 原始数据
│   ├── processed/         # 处理后数据
│   └── models/            # 训练好的模型
├── logs/                   # 日志文件
├── src/                    # 源代码
│   ├── data_collector/    # 数据采集
│   │   ├── stock_data.py
│   │   └── scheduler.py
│   ├── feature_engineering/ # 特征工程
│   │   ├── technical_indicators.py
│   │   └── feature_builder.py
│   ├── models/            # 深度学习模型
│   │   ├── lstm_model.py
│   │   ├── trainer.py
│   │   └── predictor.py
│   ├── utils/             # 工具函数
│   │   ├── config.py
│   │   ├── database.py
│   │   └── logger.py
│   pi/└── a               # API服务
│       └── main.py
├── train.py               # 训练脚本
├── requirements.txt       # 依赖包
├── Dockerfile            # Docker配置
├── docker-compose.yml    # Docker Compose配置
└── README.md             # 项目文档
```

## 🔧 配置说明

### 数据配置

- `stock_list`: 要跟踪的股票列表
- `start_date`: 历史数据起始日期
- `update_schedule`: 数据更新时间

### 模型配置

- `hidden_size`: LSTM隐藏层大小 (默认: 128)
- `num_layers`: LSTM层数 (默认: 2)
- `dropout`: Dropout比率 (默认: 0.2)
- `epochs`: 训练轮数 (默认: 100)
- `batch_size`: 批次大小 (默认: 32)
- `learning_rate`: 学习率 (默认: 0.001)

### 特征工程配置

- `window_size`: 时间窗口大小 (默认: 60天)
- `technical_indicators`: 技术指标参数
- `normalization`: 归一化方法 (minmax/standard)

## 📊 技术指标

系统支持以下技术指标：

- **趋势指标**: MA (5/10/20/30/60), EMA (12/26), MACD
- **动量指标**: RSI, KDJ, CCI
- **波动率指标**: ATR, 布林带
- **成交量指标**: OBV, 量比
- **价格变化**: 1/5/10/20日涨跌幅

## 🎯 模型说明

### LSTM模型

- 多层LSTM网络
- 全连接层输出
- Dropout防止过拟合
- 支持双向LSTM

### 注意力LSTM模型

- LSTM + 注意力机制
- 自动学习重要时间步
- 提高预测准确性

### 训练策略

- Early Stopping: 防止过拟合
- Learning Rate Scheduler: 动态调整学习率
- Gradient Clipping: 防止梯度爆炸

## 📈 评估指标

- **MSE**: 均方误差
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **方向准确率**: 涨跌方向预测准确率

## 🐳 Docker部署

### 构建镜像

```bash
docker build -t quant-system .
```

### 运行ker-compose 容器

```bash
docup -d
```

### 查看日志

```bash
docker-compose logs -f
```

### 停止服务

```bash
docker-compose down
```

## ⚠️ 注意事项

1. **数据源**: 使用免费的akshare数据源，无需API密钥
2. **模型用途**: 本系统仅用于学习和研究，预测结果仅供参考
3. **风险提示**: 股票投资有勿直接用于实盘风险，请交易
4. **GPU加速**: 建议使用GPU训练模型，CPU也可运行但速度较慢
5. **数据质量**: 建议定期更新数据以保证模型准确性

## 🔄 更新日志

### v0.1.0 (2026-01-06)

- ✅ 初始版本发布
- ✅ 实现数据采集模块
- ✅ 实现特征工程模块
- ✅ 实现LSTM模型
- ✅ 实现API服务
- ✅ Docker部署支持

## 📝 TODO

- [ ] 添加更多模型 (GRU, Transformer)
- [ ] 实现回测系统
- [ ] 添加可视化界面
- [ ] 支持更多数据源
- [ ] 实现模型集成
- [ ] 添加单元测试

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 📧 联系方式

如有问题，请提交Issue或联系作者。

---

**免责声明**: 本项目仅用于学习和研究目的，不构成任何投资建议。股市有风险，投资需谨慎。
