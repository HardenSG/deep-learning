@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ==================================
echo   A股量化交易平台
echo ==================================
echo.

:: 检查虚拟环境
if exist "venv\Scripts\activate.bat" (
    echo [*] 激活虚拟环境...
    call venv\Scripts\activate.bat
)

:: 检查Streamlit
where streamlit >nul 2>nul
if %errorlevel% neq 0 (
    echo [X] 未找到Streamlit
    echo.
    echo 请先安装依赖：
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: 检查app.py
if not exist "app.py" (
    echo [X] 未找到app.py
    echo.
    echo 请确保在项目根目录运行此脚本
    echo.
    pause
    exit /b 1
)

:: 启动平台
echo [*] 启动Web平台...
echo.
echo 浏览器将自动打开: http://localhost:8501
echo.
echo 按 Ctrl+C 停止服务
echo.
echo ==================================
echo.

streamlit run app.py

pause
