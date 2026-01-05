# 快速开始指南

## 🚀 5分钟快速体验

### 方式一：使用Windows启动脚本 (推荐)

1. **双击运行 `start.bat`**
2. **选择 "1" 安装依赖**
3. **选择 "5" 运行快速示例**

就这么简单！

### 方式二：命令行方式

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行快速示例
python examples/quick_start.py
```

## 📝 完整使用流程

### 步骤1: 环境准备

```bash
# 创建虚拟环境 (可选但推荐)
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 步骤2: 配置系统

编辑 `config/config.yaml`，添加你想要跟踪的股票：

```yaml
data:
  stock_list:
    - "000001"  # 平安银行
    - "600519"  # 贵州茅台
    - "000858"  # 五粮液
    # 添加更多...
```

### 步骤3: 采集数据

```bash
# 方式1: 使用脚本
python scripts/collect_data.py

# 方式2: 使用Python代码
python -c "
from src.utils.config import load_config
from src.utils.database import Database
from src.data_collector.stock_data import StockDataCollector

config = load_config()
base()
db = Datacollector = StockDataCollector(db)

# 采集单只股票
collector.collect_stock_data('000001', '2020-01-01')
"
```

### 步骤4: 训练模型

```bash
# 训练单只股票 (会自动采集数据)
python train.py --stock_code 000001 --collect_data

# 只训练不采集数据
python train.py --stock_code 000001

# 批量训练所有配置的股票
python scripts/batch_train.py
```

训练过程会显示：
- 数据加载进度
- 特征工程信息
- 训练进度和损失
- 评估指标

### 步骤5: 启动API服务

```bash
# 方式1: 直接运行
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 方式2: 使用Docker
docker-compose up -d
```

访问 http://localhost:8000/docs 查看API文档

### 步骤6: 使用API

#### 方式1: 使用浏览器
打开 http://localhost:8000/docs，在Swagger UI中测试API

#### 方式2: 使用curl
```bash
# 获取股票列表
curl http://localhost:8000/api/stocks

# 获取股票数据
curl http://localhost:8000/api/data/000001?limit=10

# 预测股票
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"stock_code": "000001", "days": 5}'
```

#### 方式3: 使用Python
```python
# 运行示例客户端
python examples/api_client.py

# 或者自己写代码
import requests

response = requests.post(
    "http://localhost:8000/api/predict",
    json={"stock_code": "000001", "days": 5}
)
print(response.json())
```

## 🎯 常见使用场景

### 场景1: 快速测试单只股票

```bash
# 一条命令完成数据采集+训练+预测
python train.py --stock_code 000001 --collect_data

# 然后启动API
python -m uvicorn src.api.main:app --port 8000

# 访问预测接口
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"stock_code": "000001"}'
```

### 场景2: 批量处理多只股票

```bash
# 1. 配置股票列表 (编辑 config/config.yaml)
# 2. 批量采集数据
python scripts/collect_data.py

# 3. 批量训练
python scripts/batch_train.py

# 4. 启动API服务
docker-compose up -d
```

### 场景3: 定时更新和预测

```python
# 创建定时任务脚本
from src.data_collector.scheduler import DataScheduler
from src.data_collector.stock_data import StockDataCollector
from src.utils.database import Database

db = Database()
collector = StockDataCollector(db)
scheduler = DataScheduler()

# 每天15:30更新数据
def update_data():
    stock_codes = ["000001", "600519"]
    collector.update_all_stocks(stock_codes, days_back=5)

scheduler.add_daily_update_job(update_data, hour=15, minute=30)
scheduler.start()

# 保持运行
import time
while True:
    time.sleep(60)
```

## 🐳 Docker快速部署

```bash
# 1. 构建并启动
docker-compose up -d

# 2. 查看日志
docker-compose logs -f

# 3. 停止服务
docker-compose down

# 4. 重启服务
docker-compose restart
```

## ⚡ 性能优化建议

### 1. 使用GPU加速

编辑 `config/config.yaml`:
```yaml
system:
  device: "cuda"  # 改为 cuda
```

### 2. 调整批次大小

```yaml
model:
  training:
    batch_size: 64  # 根据显存调整
```

### 3. 减少训练轮数 (快速测试)

```yaml
model:
  training:
    epochs: 20  # 默认100，测试时可减少
```

## 🔧 故障排除

### 问题1: 安装依赖失败

```bash
# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题2: 数据采集失败

```python
# 检查网络连接
# 尝试单独采集一只股票
python -c "
from src.utils.database import Database
from src.data_collector.stock_data import StockDataCollector

db = Database()
collector = StockDataCollector(db)
collector.collect_stock_data('000001', '2023-01-01')
"
```

### 问题3: 模型训练内存不足

```yaml
# 减小批次大小
model:
  training:
    batch_size: 16  # 默认32

# 减小窗口大小
features:
  window_size: 30  # 默认60
```

### 问题4: API无法访问

```bash
# 检查端口是否被占用
netstat -ano | findstr :8000

# 更换端口
python -m uvicorn src.api.main:app --port 8001
```

## 📚 下一步学习

1. **阅读完整文档**: [README.md](README.md)
2. **查看项目总结**: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
3. **研究示例代码**: `examples/` 目录
4. **自定义配置**: 修改 `config/config.yaml`
5. **扩展功能**: 添加新的技术指标或模型

## 💡 提示

- 首次运行建议使用少量股票测试
- 训练时间取决于数据量和硬件配置
- 建议定期备份 `data/` 目录
- 生产环境建议使用PostgreSQL替代SQLite

## 🎓 学习资源

- **akshare文档**: https://akshare.akfamily.xyz/
- **PyTorch教程**: https://pytorch.org/tutorials/
- **FastAPI文档**: https://fastapi.tiangolo.com/

---

**祝你使用愉快！** 🚀

如有问题，请查看 [README.md](README.md) 或提交Issue。
