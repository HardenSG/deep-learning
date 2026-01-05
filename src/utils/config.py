import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        self._process_config()
    
    def _process_config(self):
        if self._config.get("data", {}).get("end_date") is None:
            self._config["data"]["end_date"] = datetime.now().strftime("%Y-%m-%d")
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def __getitem__(self, key: str) -> Any:
        return self.get(key)
    
    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None
    
    @property
    def data(self) -> Dict[str, Any]:
        return self._config.get("data", {})
    
    @property
    def features(self) -> Dict[str, Any]:
        return self._config.get("features", {})
    
    @property
    def model(self) -> Dict[str, Any]:
        return self._config.get("model", {})
    
    @property
    def api(self) -> Dict[str, Any]:
        return self._config.get("api", {})
    
    @property
    def logging(self) -> Dict[str, Any]:
        return self._config.get("logging", {})
    
    @property
    def backtest(self) -> Dict[str, Any]:
        return self._config.get("backtest", {})
    
    @property
    def system(self) -> Dict[str, Any]:
        return self._config.get("system", {})


def load_config(config_path: Optional[str] = None) -> Config:
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)
