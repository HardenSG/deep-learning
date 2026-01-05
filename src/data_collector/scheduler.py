from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
from typing import List, Callable
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataScheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.jobs = {}
        logger.info("数据调度器初始化完成")
    
    def add_daily_update_job(
        self, 
        func: Callable, 
        hour: int = 15, 
        minute: int = 30,
        job_id: str = "daily_update"
    ) -> None:
        trigger = CronTrigger(
            day_of_week="mon-fri",
            hour=hour,
            minute=minute
        )
        
        job = self.scheduler.add_job(
            func,
            trigger=trigger,
            id=job_id,
            name="每日数据更新",
            replace_existing=True
        )
        
        self.jobs[job_id] = job
        logger.info(f"添加定时任务: {job_id}, 执行时间: 每个交易日 {hour:02d}:{minute:02d}")
    
    def add_interval_job(
        self,
        func: Callable,
        minutes: int = 60,
        job_id: str = "interval_job"
    ) -> None:
        job = self.scheduler.add_job(
            func,
            "interval",
            minutes=minutes,
            id=job_id,
            name=f"间隔任务({minutes}分钟)",
            replace_existing=True
        )
        
        self.jobs[job_id] = job
        logger.info(f"添加间隔任务: {job_id}, 间隔: {minutes} 分钟")
    
    def remove_job(self, job_id: str) -> None:
        if job_id in self.jobs:
            self.scheduler.remove_job(job_id)
            del self.jobs[job_id]
            logger.info(f"移除任务: {job_id}")
    
    def start(self) -> None:
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("调度器已启动")
    
    def shutdown(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("调度器已关闭")
    
    def list_jobs(self) -> List[str]:
        return list(self.jobs.keys())
    
    def get_job_info(self, job_id: str) -> dict:
        if job_id in self.jobs:
            job = self.jobs[job_id]
            return {
                "id": job.id,
                "name": job.name,
                "next_run_time": job.next_run_time
            }
        return {}
