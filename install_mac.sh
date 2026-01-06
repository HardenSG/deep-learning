#!/bin/bash

# Mac版一键安装脚本

echo "=================================="
echo "  A股量化交易平台 - Mac安装"
echo "=================================="
echo ""

# 检查Python
echo "📋 检查Python环境..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✅ $PYTHON_VERSION"
else
    echo "❌ 未找到Python3"
    echo ""
    echo "请先安装Python 3.8+:"
    echo "  brew install python3"
    exit 1
fi

# 检查pip
echo ""
echo "📋 检查pip..."
if command -v pip3 &> /dev/null; then
    echo "✅ pip3 已安装"
else
    echo "❌ 未找到pip3"
    exit 1
fi

# 创建虚拟环境（可选）
echo ""
read -p "是否创建虚拟环境? (推荐) [Y/n]: " CREATE_VENV
CREATE_VENV=${CREATE_VENV:-Y}

if [[ $CREATE_VENV =~ ^[Yy]$ ]]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
    echo "✅ 虚拟环境创建完成"

    echo "🔧 激活虚拟环境..."
    source venv/bin/activate
    echo "✅ 虚拟环境已激活"
fi

# 安装核心依赖
echo ""
echo "📥 安装核心依赖..."
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
    echo "✅ 核心依赖安装完成"
else
    echo "⚠️  未找到requirements.txt，跳过"
fi

# 安装Web平台依赖
echo ""
echo "📥 安装Web平台依赖..."
pip3 install streamlit plotly

echo ""
echo "✅ 安装完成！"
echo ""
echo "=================================="
echo "  启动平台："
echo "=================================="
echo ""
if [[ $CREATE_VENV =~ ^[Yy]$ ]]; then
    echo "  source venv/bin/activate"
fi
echo "  streamlit run app.py"
echo ""
echo "或者运行快捷脚本："
echo "  ./start_mac.sh"
echo ""
echo "=================================="
