import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logger(
    log_dir: str = "logs",
    log_file: str = "quant_system.log",
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "30 days",
    compression: str = "zip"
) -> None:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger.remove()
    
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )
    
    logger.add(
        log_path / log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation=rotation,
        retention=retention,
        compression=compression,
        encoding="utf-8"
    )
    
    logger.info(f"日志系统初始化完成，日志文件: {log_path / log_file}")


def get_logger(name: Optional[str] = None):
    if name:
        return logger.bind(name=name)
    return logger
