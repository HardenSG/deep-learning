import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json
from datetime import datetime
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001
    ):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        
        logger.info(f"模型训练器初始化完成，设备: {self.device}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(batch_X)
            loss = self.criterion(outputs.squeeze(), batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"开始训练: epochs={epochs}, batch_size={batch_size}")
        
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            logger.info(
                f"Epoch [{epoch+1}/{epochs}] - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                
                if save_path:
                    self.save_checkpoint(save_path, epoch, val_loss)
                    logger.info(f"模型已保存 (Val Loss: {val_loss:.6f})")
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info(f"训练完成! 最佳验证损失: {self.best_val_loss:.6f}")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss
        }
    
    def save_checkpoint(self, save_path: str, epoch: int, val_loss: float) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "model_info": self.model.get_model_info() if hasattr(self.model, "get_model_info") else {}
        }
        
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(self, load_path: str) -> Dict:
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.best_val_loss = checkpoint.get("val_loss", float("inf"))
        
        logger.info(f"模型加载自: {load_path}")
        
        return checkpoint
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test)
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                predictions.extend(outputs.cpu().numpy().flatten())
                actuals.extend(batch_y.numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(mse)
        
        direction_accuracy = np.mean((predictions > 0) == (actuals > 0))
        
        metrics = {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "direction_accuracy": float(direction_accuracy)
        }
        
        logger.info(f"评估结果: MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}, 方向准确率={direction_accuracy:.4f}")
        
        return metrics
