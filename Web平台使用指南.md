# Web平台快速启动指南

## 🚀 10秒启动

### 步骤1: 安装Streamlit

```bash
pip install streamlit plotly
```

### 步骤2: 启动平台

```bash
streamlit run app.py
```

就这么简单！浏览器会自动打开 http://localhost:8501

---

## 📱 平台功能演示

### 功能1: 🔍 股票查询

1. 输入股票代码（如 `600519`）
2. 选择显示天数
3. 点击"查询"
4. 查看：
   - 实时价格和涨跌幅
   - 交互式K线图
   - 成交量柱状图
   - 历史数据表格

### 功能2: 🤖 股票预测

1. 输入已训练模型的股票代码
2. 点击"开始预测"
3. 获得：
   - 价格预测（当前价 → 预测价 → 涨跌幅）
   - 概率分析（上涨/下跌/大涨/大跌概率）
   - 交易建议（买入/观望/卖出 + 仓位建议）
   - 止损止盈价格
   - 风险评估（波动率、VaR、综合风险）

### 功能3: 🏋️ 模型训练

1. 输入新股票代码
2. 调整高级选项（可选）
   - 训练轮数
   - 批次大小
   - 隐藏层大小
3. 点击"开始训练"
4. 实时查看：
   - 训练进度（5个步骤）
   - 详细日志
   - 损失曲线
   - 性能指标

---

## 🎨 界面预览

```
┌─────────────────────────────────────────────────┐
│  📈 A股量化交易平台                              │
├─────────────────────────────────────────────────┤
│                                                  │
│  导航菜单                                        │
│  ● 🔍 股票查询                                   │
│  ○ 🤖 股票预测                                   │
│  ○ 🏋️ 模型训练                                   │
│                                                  │
│  平台功能:                                       │
│  - 查询股票历史数据                             │
│  - AI智能预测涨跌                               │
│  - 一键训练新模型                               │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

## 🌐 部署选项

### 方案1: 本地运行（最简单）

```bash
streamlit run app.py
```

- ✅ 零配置
- ✅ 本地访问
- ❌ 无法外网访问

### 方案2: 局域网共享

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

然后局域网内访问: `http://你的IP:8501`

- ✅ 同事可访问
- ✅ 不需要公网
- ❌ 仅限局域网

### 方案3: 云部署（免费）

#### Streamlit Cloud（推荐）

1. 代码上传到 GitHub
2. 访问 https://streamlit.io/cloud
3. 连接 GitHub 仓库
4. 选择 `app.py`
5. 点击 Deploy

- ✅ 完全免费
- ✅ 自动HTTPS
- ✅ 全球访问
- ✅ 自动更新

#### Hugging Face Spaces

1. 访问 https://huggingface.co/spaces
2. 创建新 Space（选择 Streamlit）
3. 上传 `app.py` 和依赖
4. 自动部署

- ✅ 完全免费
- ✅ GPU支持
- ✅ 稳定可靠

---

## 📦 完整依赖列表

创建 `requirements_web.txt`:

```txt
streamlit==1.28.0
plotly==5.17.0
pandas==2.1.0
numpy==1.25.0
torch==2.0.1
scikit-learn==1.3.0
akshare==1.11.0
```

安装：
```bash
pip install -r requirements_web.txt
```

---

## ⚙️ 配置选项

### 自定义端口

```bash
streamlit run app.py --server.port 8080
```

### 禁用文件监控（生产环境）

```bash
streamlit run app.py --server.fileWatcherType none
```

### 配置文件

创建 `.streamlit/config.toml`:

```toml
[server]
port = 8501
headless = true

[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

---

## 🔒 安全建议（如果需要）

虽然你说不需要鉴权，但如果需要简单保护：

### 方法1: HTTP Basic Auth

```python
# 在 app.py 顶部添加
import streamlit as st

def check_password():
    def password_entered():
        if st.session_state["password"] == "your_password":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("密码", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("密码", type="password", on_change=password_entered, key="password")
        st.error("密码错误")
        return False
    else:
        return True

if not check_password():
    st.stop()

# 原有代码...
```

### 方法2: IP白名单

使用 Nginx 反向代理：

```nginx
location / {
    allow 192.168.1.0/24;
    deny all;
    proxy_pass http://localhost:8501;
}
```

---

## 🐛 常见问题

### Q1: 页面加载慢？

**A**: 使用缓存装饰器：

```python
@st.cache_data
def load_stock_data(stock_code):
    return db.get_stock_daily(stock_code)
```

### Q2: 训练时页面卡住？

**A**: 这是正常的，训练是同步操作。可以改为后台任务：

```python
import threading

def train_in_background(stock_code):
    # 训练代码...
    pass

if train_button:
    thread = threading.Thread(target=train_in_background, args=(stock_code,))
    thread.start()
    st.info("后台训练中，可继续使用其他功能")
```

### Q3: 如何自动刷新数据？

**A**: 在侧边栏添加：

```python
if st.sidebar.button("🔄 刷新数据"):
    st.cache_data.clear()
    st.experimental_rerun()
```

---

## 📊 性能优化

### 1. 启用缓存

```python
@st.cache_resource
def load_model(stock_code):
    # 模型加载代码
    pass

@st.cache_data(ttl=3600)  # 缓存1小时
def get_stock_data(stock_code):
    # 数据加载代码
    pass
```

### 2. 数据分页

```python
# 只显示前100条
st.dataframe(df.head(100))
```

### 3. 异步加载

```python
with st.spinner("加载中..."):
    data = load_data()
```

---

## 🎯 扩展功能建议

### 1. 批量预测

```python
stock_codes = st.multiselect("选择多个股票", ["600519", "000001", "600036"])
if st.button("批量预测"):
    for code in stock_codes:
        # 预测逻辑
        pass
```

### 2. 导出报告

```python
import io
import base64

def generate_pdf_report(result):
    # 生成PDF
    pass

if st.button("下载报告"):
    pdf = generate_pdf_report(result)
    b64 = base64.b64encode(pdf).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">下载PDF</a>'
    st.markdown(href, unsafe_allow_html=True)
```

### 3. 实时数据推送

```python
import time

placeholder = st.empty()

while True:
    data = get_realtime_data()
    with placeholder.container():
        st.metric("实时价格", data['price'])
    time.sleep(5)
```

---

## 🎉 总结

### 优势

✅ **开发速度快** - 100行代码实现完整平台
✅ **无需前端知识** - 纯Python开发
✅ **自带UI组件** - 输入框、图表、按钮全都有
✅ **部署简单** - 一行命令启动
✅ **免费部署** - Streamlit Cloud零成本

### 对比其他方案

| 方案 | 开发时间 | 代码量 | 难度 | 推荐度 |
|------|---------|--------|------|--------|
| **Streamlit** | 2小时 | 100-200行 | ⭐☆☆☆☆ | ⭐⭐⭐⭐⭐ |
| Flask + HTML | 2天 | 500+行 | ⭐⭐⭐☆☆ | ⭐⭐⭐☆☆ |
| Django | 3天 | 1000+行 | ⭐⭐⭐⭐☆ | ⭐⭐☆☆☆ |
| React + FastAPI | 1周 | 2000+行 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐☆ |

**结论**: 对于你的需求（简单、无鉴权、快速上线），Streamlit是完美选择！

---

**启动命令**: `streamlit run app.py`
**访问地址**: http://localhost:8501
**部署时间**: < 5分钟
