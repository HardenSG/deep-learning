import sys
from pathlib import Path
import subprocess
from datetime import datetime
import json
import time

sys.path.append(str(Path(__file__).parent))

from src.utils.logger import setup_logger, get_logger

setup_logger(
    log_dir="logs",
    log_file=f"batch_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level="INFO"
)
logger = get_logger(__name__)

PYTHON_PATH = sys.executable

STOCK_CODES = [
    "601698",
    "600118", 
    "600879",
    "000547",
    "301232",
    "301005",
    "603986",
    "688981",
    "688012",
    "605358",
    "688261",
    "688041",
    "603061",
    "688385",
    "688766"
]

def check_model_exists(stock_code):
    model_path = Path(f"data/models/{stock_code}_model.pth")
    return model_path.exists()

def train_stock(stock_code):
    if check_model_exists(stock_code):
        logger.info(f"⊙ {stock_code} 模型已存在，跳过训练")
        print(f"⊙ {stock_code} 模型已存在，跳过训练")
        return {"stock_code": stock_code, "status": "skipped", "message": "模型已存在"}
    
    logger.info(f"▶ 开始训练 {stock_code}...")
    print(f"\n{'='*60}")
    print(f"正在训练: {stock_code}")
    print(f"{'='*60}")
    
    try:
        cmd = [
            PYTHON_PATH,
            "train.py",
            "--stock_code", stock_code,
            "--collect_data"
        ]
        
        start_time = time.time()
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='ignore',
            bufsize=1
        )
        
        output_lines = []
        for line in process.stdout:
            line = line.strip()
            if line:
                output_lines.append(line)
                
                keywords = [
                    'Epoch', 'epoch', 
                    'Loss', 'loss', 'MSE', 'mse',
                    'lr', '学习率',
                    '准确率', 'accuracy', 'Direction', '方向',
                    '验证', 'Val', 'val',
                    'Train', 'train',
                    '最佳', 'Best', 'best',
                    '采集', '数据', '特征', '训练'
                ]
                
                if any(keyword in line for keyword in keywords):
                    print(f"  {line}")
                    logger.info(line)
        
        process.wait()
        elapsed_time = time.time() - start_time
        
        if process.returncode == 0:
            logger.info(f"✓ {stock_code} 训练成功 (耗时: {elapsed_time:.1f}秒)")
            print(f"✓ {stock_code} 训练成功 (耗时: {elapsed_time:.1f}秒)\n")
            return {
                "stock_code": stock_code, 
                "status": "success", 
                "message": "训练成功",
                "elapsed_time": elapsed_time
            }
        else:
            error_msg = '\n'.join(output_lines[-10:]) if output_lines else "未知错误"
            logger.error(f"✗ {stock_code} 训练失败: {error_msg}")
            print(f"✗ {stock_code} 训练失败\n")
            return {
                "stock_code": stock_code, 
                "status": "failed", 
                "error": error_msg,
                "elapsed_time": elapsed_time
            }
            
    except Exception as e:
        logger.error(f"✗ {stock_code} 训练出错: {e}")
        print(f"✗ {stock_code} 训练出错: {e}\n")
        return {"stock_code": stock_code, "status": "failed", "error": str(e)}

def main():
    results = {
        "success": [],
        "failed": [],
        "skipped": [],
        "start_time": datetime.now().isoformat()
    }
    
    print("\n" + "=" * 60)
    print("批量训练开始")
    print(f"股票数量: {len(STOCK_CODES)}")
    print(f"股票列表: {', '.join(STOCK_CODES)}")
    print("=" * 60)
    
    logger.info("=" * 60)
    logger.info("批量训练开始")
    logger.info(f"股票数量: {len(STOCK_CODES)}")
    logger.info(f"股票列表: {', '.join(STOCK_CODES)}")
    logger.info("=" * 60)
    
    for idx, stock_code in enumerate(STOCK_CODES, 1):
        print(f"\n[{idx}/{len(STOCK_CODES)}] 处理 {stock_code}")
        logger.info(f"\n[{idx}/{len(STOCK_CODES)}] 处理 {stock_code}")
        
        result = train_stock(stock_code)
        
        if result["status"] == "success":
            results["success"].append(result)
        elif result["status"] == "failed":
            results["failed"].append(result)
        else:
            results["skipped"].append(result)
        
        if idx < len(STOCK_CODES):
            time.sleep(1)
    
    results["end_time"] = datetime.now().isoformat()
    
    print("\n" + "=" * 60)
    print("批量训练完成!")
    print("=" * 60)
    print(f"总数: {len(STOCK_CODES)}")
    print(f"✓ 成功: {len(results['success'])}")
    print(f"✗ 失败: {len(results['failed'])}")
    print(f"⊙ 跳过: {len(results['skipped'])}")
    
    logger.info("\n" + "=" * 60)
    logger.info("批量训练完成!")
    logger.info("=" * 60)
    logger.info(f"总数: {len(STOCK_CODES)}")
    logger.info(f"✓ 成功: {len(results['success'])}")
    logger.info(f"✗ 失败: {len(results['failed'])}")
    logger.info(f"⊙ 跳过: {len(results['skipped'])}")
    
    if results["success"]:
        print("\n成功列表:")
        logger.info("\n成功列表:")
        for item in results["success"]:
            msg = f"  ✓ {item['stock_code']} (耗时: {item.get('elapsed_time', 0):.1f}秒)"
            print(msg)
            logger.info(msg)
    
    if results["failed"]:
        print("\n失败列表:")
        logger.info("\n失败列表:")
        for item in results["failed"]:
            msg = f"  ✗ {item['stock_code']}: {item.get('error', '未知错误')[:100]}"
            print(msg)
            logger.info(msg)
    
    if results["skipped"]:
        print("\n跳过列表:")
        logger.info("\n跳过列表:")
        for item in results["skipped"]:
            msg = f"  ⊙ {item['stock_code']}: {item['message']}"
            print(msg)
            logger.info(msg)
    
    print("=" * 60)
    logger.info("=" * 60)
    
    result_file = Path("logs") / f"batch_train_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存至: {result_file}")
    logger.info(f"\n结果已保存至: {result_file}")

if __name__ == "__main__":
    main()
