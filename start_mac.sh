#!/bin/bash

# Mac版一键启动脚本

echo "=================================="
echo "  A股量化交易平台"
echo "=================================="
echo ""

# 检查虚拟环境
if [ -d "venv" ]; then
    echo "🔧 激活虚拟环境..."
    source venv/bin/activate
fi

# 检查Streamlit
if ! command -v streamlit &> /dev/null; then
    echo "❌ 未找到Streamlit"
    echo ""
    echo "请先运行安装脚本："
    echo "  ./install_mac.sh"
    exit 1
fi

# 检查app.py
if [ ! -f "app.py" ]; then
    echo "❌ 未找到app.py"
    echo ""
    echo "请确保在项目根目录运行此脚本"
    exit 1
fi

# 启动平台
echo "🚀 启动Web平台..."
echo ""
echo "浏览器将自动打开: http://localhost:8501"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""
echo "=================================="
echo ""

streamlit run app.py
