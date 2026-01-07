# -*- coding: utf-8 -*-
"""
A股量化交易 Web 平台 - Streamlit版本

功能：
1. 股票查询 - 查看历史数据和K线图
2. 股票预测 - AI预测涨跌和交易建议
3. 模型训练 - 离线训练新股票模型
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import time
from datetime import datetime

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from src.utils.config import load_config
from src.utils.database import Database
from src.utils.random_utils import set_seed
from src.data_collector.stock_data import StockDataCollector
from src.data_collector.unified_collector import UnifiedDataCollector
from src.feature_engineering.feature_builder import FeatureBuilder
from src.models.lstm_model import LSTMModel
from src.models.improved_predictor import ImprovedPredictor
import torch

# 页面配置
st.set_page_config(
    page_title="A股量化交易平台 (支持ETF)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载配置
@st.cache_resource
def load_app_config():
    return load_config()

@st.cache_resource
def get_database():
    config = load_app_config()
    return Database(config.data.get("database_path", "data/stock_data.db"))

config = load_app_config()
db = get_database()

# 侧边栏导航
st.sidebar.title("📈 量化交易平台")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "导航菜单",
    ["🔍 股票/ETF查询", "🤖 股票/ETF预测", "🏋️ 模型训练", "📊 策略回测"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("""
**平台功能**
- 查询股票/ETF/基金历史数据
- AI智能预测涨跌
- 一键训练新模型
- 策略回测分析

**支持类型**
- ✅ A股股票
- ✅ ETF基金
- ✅ 普通基金

**技术栈**
- LSTM深度学习
- Monte Carlo模拟
- 技术指标分析
- 量化回测引擎
""")


# ==================== 页面1：股票/ETF查询 ====================
if page == "🔍 股票/ETF查询":
    st.title("🔍 股票/ETF/基金数据查询")
    st.markdown("查看股票/ETF/基金历史数据、K线图和技术指标")

    # 证券类型选择
    security_type = st.radio(
        "证券类型",
        ["A股股票", "ETF基金", "普通基金"],
        horizontal=True,
        help="选择要查询的证券类型"
    )

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if security_type == "A股股票":
            default_code = "600519"
            help_text = "输入6位股票代码，如 600519（贵州茅台）"
        elif security_type == "ETF基金":
            default_code = "563530"
            help_text = "输入6位ETF代码，如 563530（中证800ETF）"
        else:
            default_code = "003494"
            help_text = "输入6位基金代码，如 003494（富国天惠成长混合C）"

        stock_code = st.text_input(
            "证券代码",
            value=default_code,
            help=help_text
        )

    with col2:
        days = st.selectbox("显示天数", [30, 60, 90, 180, 365], index=1)

    with col3:
        if st.button("🔍 查询", type="primary"):
            st.session_state.query_clicked = True

    if st.session_state.get("query_clicked", False):
        with st.spinner("正在加载数据..."):
            df = db.get_stock_daily(stock_code)

            if df.empty:
                st.error(f"❌ 未找到股票 {stock_code} 的数据")
            else:
                df = df.tail(days)

                # 基本信息
                st.markdown("### 📊 基本信息")
                col1, col2, col3, col4, col5 = st.columns(5)

                latest = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else latest

                with col1:
                    st.metric(
                        "最新价",
                        f"¥{latest['close']:.2f}",
                        f"{(latest['close'] - prev['close']):.2f}"
                    )
                with col2:
                    change_pct = (latest['close'] - prev['close']) / prev['close'] * 100
                    st.metric("涨跌幅", f"{change_pct:+.2f}%")
                with col3:
                    st.metric("最高价", f"¥{latest['high']:.2f}")
                with col4:
                    st.metric("最低价", f"¥{latest['low']:.2f}")
                with col5:
                    st.metric("成交量", f"{latest['volume']/10000:.0f}万")

                # K线图
                st.markdown("### 📈 K线图")

                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.7, 0.3],
                    subplot_titles=('K线图', '成交量')
                )

                # K线
                fig.add_trace(
                    go.Candlestick(
                        x=df['trade_date'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name='K线'
                    ),
                    row=1, col=1
                )

                # 成交量
                colors = ['red' if df['close'].iloc[i] >= df['open'].iloc[i] else 'green'
                         for i in range(len(df))]
                fig.add_trace(
                    go.Bar(x=df['trade_date'], y=df['volume'], name='成交量', marker_color=colors),
                    row=2, col=1
                )

                fig.update_layout(
                    title=f"{stock_code} K线图",
                    height=600,
                    xaxis_rangeslider_visible=False
                )

                st.plotly_chart(fig, use_container_width=True)

                # 数据表格
                st.markdown("### 📋 历史数据")
                display_df = df[['trade_date', 'open', 'high', 'low', 'close', 'volume']].copy()
                display_df.columns = ['日期', '开盘', '最高', '最低', '收盘', '成交量']
                st.dataframe(display_df.tail(20), use_container_width=True)


# ==================== 页面2：股票/ETF预测 ====================
elif page == "🤖 股票/ETF预测":
    st.title("🤖 AI股票/ETF/基金预测")
    st.markdown("基于LSTM深度学习模型的智能预测")

    # 证券类型选择
    security_type = st.radio(
        "证券类型",
        ["A股股票", "ETF基金", "普通基金"],
        horizontal=True,
        help="选择要预测的证券类型"
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        if security_type == "A股股票":
            default_code = "600519"
            help_text = "输入已训练过模型的股票代码"
        elif security_type == "ETF基金":
            default_code = "563530"
            help_text = "输入已训练过模型的ETF代码"
        else:
            default_code = "003494"
            help_text = "输入已训练过模型的基金代码"

        stock_code = st.text_input(
            "证券代码",
            value=default_code,
            help=help_text
        )

    with col2:
        if st.button("🚀 开始预测", type="primary"):
            st.session_state.predict_clicked = True

    if st.session_state.get("predict_clicked", False):
        model_path = Path(config.model.get("model_save_path", "data/models")) / f"{stock_code}_model.pth"
        scaler_path = Path(config.model.get("model_save_path", "data/models")) / f"{stock_code}_scaler.pkl"

        if not model_path.exists():
            st.error(f"❌ 模型不存在！请先在【模型训练】页面训练 {stock_code} 的模型")
        else:
            with st.spinner("AI正在分析中..."):
                try:
                    # 加载数据
                    df = db.get_stock_daily(stock_code)
                    if df.empty:
                        st.error(f"❌ 未找到股票 {stock_code} 的数据")
                    else:
                        current_price = float(df.iloc[-1]["close"])

                        # 准备特征
                        feature_builder = FeatureBuilder(config.features)
                        feature_builder.load_scaler(str(scaler_path))
                        X = feature_builder.prepare_prediction_data(df)

                        # 加载模型
                        input_size = X.shape[2]
                        
                        # 先尝试按当前配置初始化
                        model = LSTMModel(
                            input_size=input_size,
                            hidden_size=config.model.get("lstm", {}).get("hidden_size", 128),
                            num_layers=config.model.get("lstm", {}).get("num_layers", 2),
                            dropout=config.model.get("lstm", {}).get("dropout", 0.2)
                        )

                        predictor = ImprovedPredictor(model, device="cpu")
                        
                        try:
                            predictor.load_model(str(model_path))
                        except RuntimeError as e:
                            # 处理模型结构不匹配的问题（例如旧模型是2层，新配置是1层）
                            if "Unexpected key(s) in state_dict" in str(e) or "size mismatch" in str(e):
                                st.warning("检测到模型结构与当前配置不一致，正在尝试适配旧模型...")
                                
                                # 读取checkpoint获取原始模型参数
                                checkpoint = torch.load(str(model_path), map_location="cpu")
                                model_info = checkpoint.get("model_info", {})
                                
                                if model_info:
                                    # 使用保存的模型参数重新初始化
                                    model = LSTMModel(
                                        input_size=model_info.get("input_size", input_size),
                                        hidden_size=model_info.get("hidden_size", 128),
                                        num_layers=model_info.get("num_layers", 2),
                                        dropout=config.model.get("lstm", {}).get("dropout", 0.2), # Dropout不影响权重加载
                                        bidirectional=model_info.get("bidirectional", False)
                                    )
                                    
                                    # 重新加载
                                    predictor = ImprovedPredictor(model, device="cpu")
                                    predictor.load_model(str(model_path))
                                    st.success("✅ 已成功适配并加载旧模型")
                                else:
                                    st.error("无法适配旧模型：缺少元数据。请重新训练模型。")
                                    raise e
                            else:
                                raise e

                        # 预测
                        result = predictor.get_comprehensive_prediction(
                            X, current_price, stock_code, n_simulations=30
                        )

                        # 显示结果
                        st.success("✅ 预测完成！")

                        # 价格预测
                        st.markdown("### 💰 价格预测")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric(
                                "当前价格",
                                f"¥{result['price']['current']:.2f}"
                            )
                        with col2:
                            st.metric(
                                "预测价格",
                                f"¥{result['price']['predicted']:.2f}",
                                f"{result['price']['change_pct']:+.2f}%"
                            )
                        with col3:
                            direction = result['probability']['direction']
                            emoji = "📈" if direction == "上涨" else "📉"
                            st.metric("预测方向", f"{emoji} {direction}")

                        # 概率分析
                        st.markdown("### 📊 概率分析")
                        col1, col2, col3, col4 = st.columns(4)

                        prob = result['probability']
                        with col1:
                            st.metric("上涨概率", f"{prob['up']:.1f}%")
                        with col2:
                            st.metric("下跌概率", f"{prob['down']:.1f}%")
                        with col3:
                            st.metric("大涨概率(>5%)", f"{prob['large_up']:.1f}%")
                        with col4:
                            st.metric("大跌概率(<-5%)", f"{prob['large_down']:.1f}%")

                        # 交易建议
                        st.markdown("### 💡 交易建议")
                        signals = result['trading_signals']

                        col1, col2 = st.columns(2)

                        with col1:
                            action_color = {
                                "强烈买入": "🟢", "买入": "🟢", "小幅买入": "🟡",
                                "观望": "🟡", "减仓观望": "🟠",
                                "卖出/空仓": "🔴", "不建议买入": "🔴"
                            }
                            st.info(f"""
**操作建议**: {action_color.get(signals['action'], '🟡')} **{signals['action']}**

**理由**: {signals['reason']}

**置信度**: {signals['confidence']}
                            """)

                        with col2:
                            st.warning(f"""
**仓位建议**: {signals['position']['suggested_pct']}%

**止损价**: ¥{signals['stop_loss']['price']:.2f} ({signals['stop_loss']['pct']:.1f}%)

**止盈价**: ¥{signals['take_profit']['price']:.2f} ({signals['take_profit']['pct']:.1f}%)
                            """)

                        # 风险评估
                        st.markdown("### ⚠️ 风险评估")
                        risk = result['risk_metrics']

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            vol_level = risk['volatility']['level']
                            vol_color = {"低": "🟢", "中": "🟡", "高": "🔴"}
                            st.metric(
                                "波动率等级",
                                f"{vol_color.get(vol_level, '🟡')} {vol_level}",
                                f"年化 {risk['volatility']['annualized']*100:.1f}%"
                            )

                        with col2:
                            st.metric(
                                "VaR (95%置信)",
                                f"{risk['value_at_risk']['var_95_pct']:.2f}%"
                            )

                        with col3:
                            risk_level = signals['risk_assessment']['overall_risk']
                            risk_color = {"低": "success", "中": "warning", "高": "error"}
                            st.metric("综合风险", f"{risk_level}")

                        # 数据质量
                        quality = result['metadata']['data_quality']
                        if quality['score'] < 90:
                            st.warning(f"⚠️ 数据质量: {quality['score']}/100 - {quality['level']}")

                        # 免责声明
                        st.markdown("---")
                        st.caption(result['disclaimer']['warning'])

                except Exception as e:
                    st.error(f"❌ 预测失败: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())


# ==================== 页面3：模型训练 ====================
elif page == "🏋️ 模型训练":
    st.title("🏋️ 模型训练")
    st.markdown("训练新股票/ETF/基金的AI预测模型")

    st.info("""
**训练说明**
1. 选择证券类型（A股/ETF/基金）
2. 输入证券代码
3. 系统会自动采集数据
4. 自动计算技术指标
5. 训练LSTM模型
6. 保存模型供预测使用

⏱️ 训练时间：约2-5分钟（取决于数据量）
    """)

    # 证券类型选择
    security_type = st.radio(
        "证券类型",
        ["A股股票", "ETF基金", "普通基金"],
        horizontal=True,
        help="选择要训练的证券类型"
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        if security_type == "A股股票":
            default_code = "000001"
            help_text = "输入6位股票代码"
        elif security_type == "ETF基金":
            default_code = "563530"
            help_text = "输入6位ETF代码"
        else:
            default_code = "003494"
            help_text = "输入6位基金代码"

        stock_code = st.text_input(
            "证券代码",
            value=default_code,
            help=help_text
        )

    with col2:
        st.markdown("&nbsp;")
        train_button = st.button("🚀 开始训练", type="primary")

    # 高级选项
    with st.expander("⚙️ 高级选项"):
        col1, col2, col3 = st.columns(3)
        with col1:
            epochs = st.slider("训练轮数", 10, 200, 50)
        with col2:
            batch_size = st.slider("批次大小", 4, 128, 64)
        with col3:
            learning_rate = st.number_input(
                "学习率",
                value=0.0001,
                min_value=0.0001,
                max_value=0.1,
                step=0.0001,
                format="%.4f"
            )
            
        col4, col5 = st.columns(2)
        with col4:
            hidden_size = st.slider("隐藏层大小", 8, 256, 64)
        with col5:
            window_size = st.slider("窗口大小", 3, 20, 5, help="滑动窗口长度 (建议3-5天)")
        
        seed = st.number_input("随机种子", value=42, step=1)

    if train_button:
        set_seed(seed)
        st.markdown("---")
        st.subheader("📋 训练日志")

        # 进度显示
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_area = st.empty()

        logs = []

        def add_log(msg):
            logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
            log_area.code("\n".join(logs[-20:]))  # 只显示最后20条

        try:
            # 1. 数据采集
            status_text.text("📥 步骤 1/5: 采集数据...")
            progress_bar.progress(20)

            sec_type_map = {"A股股票": "stock", "ETF基金": "etf", "普通基金": "fund"}
            sec_type_code = sec_type_map[security_type]

            add_log(f"开始采集 {security_type} {stock_code} 的数据")

            collector = UnifiedDataCollector(db)
            success = collector.collect_data(
                stock_code,
                start_date="2020-01-01",
                security_type=sec_type_code
            )

            if not success:
                st.error(f"❌ 数据采集失败，请检查代码 {stock_code} 是否正确")
                st.stop()

            add_log("✅ 数据采集完成")

            # 2. 加载数据
            status_text.text("📊 步骤 2/5: 加载和验证数据...")
            progress_bar.progress(40)

            df = db.get_stock_daily(stock_code)
            add_log(f"加载了 {len(df)} 条历史数据")

            if df.empty:
                st.error("❌ 数据为空，无法训练")
                st.stop()

            # 3. 特征工程
            status_text.text("🔧 步骤 3/5: 计算技术指标和特征...")
            progress_bar.progress(60)

            # 更新配置
            config.features["window_size"] = window_size
            
            feature_builder = FeatureBuilder(config.features)
            X_train, y_train, X_val, y_val, X_test, y_test, features = \
                feature_builder.prepare_train_val_test_data(df)

            add_log(f"✅ 特征工程完成，共 {len(features)} 个特征")
            add_log(f"训练集: {len(X_train)} 样本")
            add_log(f"验证集: {len(X_val)} 样本")
            add_log(f"测试集: {len(X_test)} 样本")

            # 4. 训练模型
            status_text.text("🤖 步骤 4/5: 训练LSTM模型...")
            progress_bar.progress(70)

            from src.models.trainer import ModelTrainer

            model = LSTMModel(
                input_size=X_train.shape[2],
                hidden_size=hidden_size,
                num_layers=2,
                dropout=0.2
            )

            add_log(f"模型参数: {model.get_model_info()['total_params']:,}")

            trainer = ModelTrainer(
                model,
                device="cpu",
                learning_rate=learning_rate,
                loss_type="mse",
                loss_alpha=1.0,
                loss_beta=0.5
            )

            model_path = Path("data/models") / f"{stock_code}_model.pth"
            model_path.parent.mkdir(parents=True, exist_ok=True)

            add_log(f"开始训练... (epochs={epochs})")

            # 训练（简化版，不显示每个epoch）
            history = trainer.train(
                X_train, y_train,
                X_val, y_val,
                epochs=epochs,
                batch_size=batch_size,
                early_stopping_patience=10,
                save_path=str(model_path)
            )

            progress_bar.progress(90)
            add_log(f"✅ 训练完成！最佳验证损失: {history['best_val_loss']:.6f}")

            # 5. 评估和保存
            status_text.text("📊 步骤 5/5: 评估模型性能...")

            metrics = trainer.evaluate(X_test, y_test, scaler=feature_builder.y_scaler)
            add_log(f"测试集评估结果 (已还原真实数值):")
            add_log(f"  MSE: {metrics['mse']:.6f}")
            add_log(f"  MAE: {metrics['mae']:.6f}")
            add_log(f"  RMSE: {metrics['rmse']:.6f}")

            # 保存 scaler
            scaler_path = Path("data/models") / f"{stock_code}_scaler.pkl"
            feature_builder.save_scaler(str(scaler_path))
            add_log(f"✅ Scaler已保存")

            progress_bar.progress(100)
            status_text.text("✅ 训练完成！")

            # 显示结果
            st.success(f"""
### 🎉 训练成功！

**股票代码**: {stock_code}
**训练轮数**: {len(history['train_losses'])}
**最佳验证损失**: {history['best_val_loss']:.6f}
**方向准确率**: {metrics['direction_accuracy']*100:.2f}%

模型已保存，可以在【股票预测】页面使用！
            """)

            # 绘制损失曲线
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=history['train_losses'],
                mode='lines',
                name='训练损失'
            ))
            fig.add_trace(go.Scatter(
                y=history['val_losses'],
                mode='lines',
                name='验证损失'
            ))
            fig.update_layout(
                title="训练过程",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ 训练失败: {str(e)}")
            add_log(f"错误: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


# ==================== 页面4：策略回测 ====================
elif page == "📊 策略回测":
    st.title("📊 策略回测分析")
    st.markdown("使用历史数据回测交易策略，评估策略表现")
    
    from src.backtest.engine import BacktestEngine
    from src.backtest.strategy import SimpleStrategy, TrendFollowingStrategy
    from src.feature_engineering.feature_builder import FeatureBuilder
    import numpy as np
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        stock_code = st.text_input(
            "股票/ETF代码",
            value="600519",
            help="输入要回测的股票/ETF代码"
        )
    
    with col2:
        st.markdown("###")
        check_model = st.button("🔍 检查模型", use_container_width=True)
    
    model_path = Path("data/models") / f"{stock_code}_model.pth"
    scaler_path = Path("data/models") / f"{stock_code}_scaler.pkl"
    
    if check_model or st.session_state.get("backtest_ready", False):
        if not model_path.exists():
            st.error(f"❌ 未找到模型文件: {stock_code}")
            st.info("💡 请先在【模型训练】页面训练该股票的模型")
        else:
            st.success(f"✅ 找到模型: {stock_code}")
            st.session_state["backtest_ready"] = True
    
    if st.session_state.get("backtest_ready", False):
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            initial_capital = st.number_input(
                "初始资金 (元)",
                min_value=10000,
                max_value=10000000,
                value=100000,
                step=10000
            )
        
        with col2:
            strategy_type = st.selectbox(
                "交易策略",
                ["简单策略", "趋势跟踪策略"],
                help="选择回测使用的交易策略"
            )
        
        with col3:
            backtest_days = st.number_input(
                "回测天数",
                min_value=30,
                max_value=365,
                value=90,
                step=30
            )
        
        st.markdown("### 策略参数")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            buy_threshold = st.slider(
                "买入阈值 (%)",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.5,
                help="预测收益率超过此值时买入"
            ) / 100
        
        with col2:
            sell_threshold = st.slider(
                "卖出阈值 (%)",
                min_value=-10.0,
                max_value=0.0,
                value=-1.0,
                step=0.5,
                help="预测收益率低于此值时卖出"
            ) / 100
        
        with col3:
            max_position = st.slider(
                "最大仓位 (%)",
                min_value=10,
                max_value=100,
                value=30,
                step=10,
                help="单只股票最大仓位比例"
            ) / 100
        
        with col4:
            commission_rate = st.slider(
                "手续费率 (%)",
                min_value=0.01,
                max_value=0.1,
                value=0.03,
                step=0.01,
                help="交易手续费率"
            ) / 100
        
        if strategy_type == "趋势跟踪策略":
            col1, col2 = st.columns(2)
            
            with col1:
                stop_loss = st.slider(
                    "止损线 (%)",
                    min_value=-20.0,
                    max_value=-1.0,
                    value=-5.0,
                    step=1.0,
                    help="亏损超过此值时止损"
                ) / 100
            
            with col2:
                take_profit = st.slider(
                    "止盈线 (%)",
                    min_value=1.0,
                    max_value=50.0,
                    value=10.0,
                    step=1.0,
                    help="盈利超过此值时止盈"
                ) / 100
        
        st.markdown("---")
        
        if st.button("🚀 开始回测", type="primary", use_container_width=True):
            with st.spinner("正在运行回测..."):
                try:
                    df = db.get_stock_daily(stock_code)
                    
                    if df.empty:
                        st.error("❌ 未找到历史数据，请先采集数据")
                    else:
                        df = df.tail(backtest_days + 100)
                        
                        st.info(f"📊 加载了 {len(df)} 条历史数据")
                        
                        feature_builder = FeatureBuilder(config.features)

                        from src.models.lstm_model import LSTMModel
                        import torch

                        # 加载checkpoint获取模型参数
                        checkpoint = torch.load(str(model_path), map_location='cpu')
                        model_info = checkpoint.get("model_info", {})

                        # 使用保存的模型参数或默认值
                        model = LSTMModel(
                            input_size=model_info.get("input_size", 36),
                            hidden_size=model_info.get("hidden_size", 128),
                            num_layers=model_info.get("num_layers", 2),
                            dropout=0.2
                        )

                        model.load_state_dict(checkpoint['model_state_dict'])
                        model.eval()

                        feature_builder.load_scaler(str(scaler_path))
                        
                        predictions = []
                        dates = []
                        prices = []
                        
                        for i in range(60, len(df)):
                            hist_data = df.iloc[:i+1].copy()
                            
                            try:
                                features = feature_builder.build_features(hist_data)
                                
                                if len(features) > 0:
                                    X, y = feature_builder.prepare_training_data(
                                        features,
                                        target_column="close",
                                        prediction_horizon=5,
                                        fit_scaler=False
                                    )
                                    
                                    if len(X) > 0:
                                        X_tensor = torch.FloatTensor(X).unsqueeze(0)
                                        
                                        with torch.no_grad():
                                            pred = model(X_tensor).cpu().numpy().flatten()
                                        
                                        if len(pred) > 0:
                                            avg_pred = float(pred[0])
                                            current_price = hist_data['close'].iloc[-1]
                                            pred_return = (avg_pred - current_price) / current_price
                                            
                                            predictions.append(pred_return)
                                            dates.append(hist_data.index[-1])
                                            prices.append(current_price)
                            except Exception as e:
                                continue
                        
                        if len(predictions) < 10:
                            st.error("❌ 预测数据不足，无法进行回测")
                        else:
                            pred_df = pd.DataFrame({
                                'stock': predictions
                            }, index=dates)
                            
                            price_df = pd.DataFrame({
                                'stock': prices
                            }, index=dates)
                            
                            if strategy_type == "简单策略":
                                strategy = SimpleStrategy(
                                    buy_threshold=buy_threshold,
                                    sell_threshold=sell_threshold,
                                    max_position_pct=max_position
                                )
                            else:
                                strategy = TrendFollowingStrategy(
                                    buy_threshold=buy_threshold,
                                    sell_threshold=sell_threshold,
                                    stop_loss=stop_loss,
                                    take_profit=take_profit,
                                    max_position_pct=max_position
                                )
                            
                            engine = BacktestEngine(
                                initial_capital=initial_capital,
                                commission_rate=commission_rate,
                                slippage=0.001
                            )
                            
                            results = engine.run(pred_df, price_df, strategy)
                            
                            if results:
                                st.success("✅ 回测完成！")
                                
                                st.markdown("### 📈 回测结果")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric(
                                        "总收益率",
                                        f"{results['total_return_pct']:.2f}%",
                                        delta=f"{results['total_return_pct']:.2f}%"
                                    )
                                
                                with col2:
                                    st.metric(
                                        "最终资产",
                                        f"¥{results['final_value']:,.0f}",
                                        delta=f"¥{results['final_value'] - results['initial_capital']:,.0f}"
                                    )
                                
                                with col3:
                                    st.metric(
                                        "夏普比率",
                                        f"{results['sharpe_ratio']:.2f}",
                                        help="风险调整后收益，>1为良好"
                                    )
                                
                                with col4:
                                    st.metric(
                                        "最大回撤",
                                        f"{results['max_drawdown_pct']:.2f}%",
                                        delta=f"-{results['max_drawdown_pct']:.2f}%",
                                        delta_color="inverse"
                                    )
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric(
                                        "交易次数",
                                        f"{results['total_trades']}",
                                        help="总交易次数（买入+卖出）"
                                    )
                                
                                with col2:
                                    st.metric(
                                        "胜率",
                                        f"{results['win_rate_pct']:.1f}%",
                                        help="盈利交易占比"
                                    )
                                
                                with col3:
                                    st.metric(
                                        "盈亏比",
                                        f"{results['profit_loss_ratio']:.2f}",
                                        help="平均盈利/平均亏损"
                                    )
                                
                                with col4:
                                    st.metric(
                                        "总盈利",
                                        f"¥{results['total_profit']:,.0f}",
                                        delta=f"¥{results['total_profit']:,.0f}"
                                    )
                                
                                st.markdown("---")
                                
                                st.markdown("### 📊 资产曲线")
                                
                                daily_df = results['daily_records']
                                
                                fig = go.Figure()
                                
                                fig.add_trace(go.Scatter(
                                    x=daily_df['date'],
                                    y=daily_df['total_value'],
                                    mode='lines',
                                    name='总资产',
                                    line=dict(color='#1f77b4', width=2)
                                ))
                                
                                fig.add_hline(
                                    y=initial_capital,
                                    line_dash="dash",
                                    line_color="gray",
                                    annotation_text="初始资金"
                                )
                                
                                fig.update_layout(
                                    title="资产变化曲线",
                                    xaxis_title="日期",
                                    yaxis_title="资产 (元)",
                                    height=400,
                                    hovermode='x unified'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("### 💰 现金与持仓")
                                    
                                    fig2 = go.Figure()
                                    
                                    fig2.add_trace(go.Scatter(
                                        x=daily_df['date'],
                                        y=daily_df['cash'],
                                        mode='lines',
                                        name='现金',
                                        stackgroup='one',
                                        fillcolor='lightgreen'
                                    ))
                                    
                                    fig2.add_trace(go.Scatter(
                                        x=daily_df['date'],
                                        y=daily_df['positions_value'],
                                        mode='lines',
                                        name='持仓市值',
                                        stackgroup='one',
                                        fillcolor='lightblue'
                                    ))
                                    
                                    fig2.update_layout(
                                        title="资产构成",
                                        xaxis_title="日期",
                                        yaxis_title="金额 (元)",
                                        height=350
                                    )
                                    
                                    st.plotly_chart(fig2, use_container_width=True)
                                
                                with col2:
                                    st.markdown("### 📈 收益率分布")
                                    
                                    fig3 = go.Figure()
                                    
                                    fig3.add_trace(go.Histogram(
                                        x=daily_df['daily_return'] * 100,
                                        nbinsx=30,
                                        name='日收益率',
                                        marker_color='lightblue'
                                    ))
                                    
                                    fig3.update_layout(
                                        title="日收益率分布",
                                        xaxis_title="收益率 (%)",
                                        yaxis_title="频数",
                                        height=350
                                    )
                                    
                                    st.plotly_chart(fig3, use_container_width=True)
                                
                                if not results['trades'].empty:
                                    st.markdown("### 📋 交易记录")
                                    
                                    trades_df = results['trades'].copy()
                                    trades_df['date'] = pd.to_datetime(trades_df['date'])
                                    
                                    st.dataframe(
                                        trades_df[[
                                            'date', 'action', 'price', 'shares',
                                            'amount', 'commission'
                                        ]].tail(20),
                                        use_container_width=True,
                                        height=300
                                    )
                                    
                                    csv = trades_df.to_csv(index=False, encoding='utf-8-sig')
                                    st.download_button(
                                        label="📥 下载完整交易记录",
                                        data=csv,
                                        file_name=f"{stock_code}_backtest_trades.csv",
                                        mime="text/csv"
                                    )
                            else:
                                st.error("❌ 回测失败，请检查数据")
                
                except Exception as e:
                    st.error(f"❌ 回测出错: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())


# 页脚
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 量化交易平台 v2.5 (支持ETF)")
st.sidebar.caption("Powered by LSTM + Streamlit")
