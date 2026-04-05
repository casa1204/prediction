"""Transformer 예측 모델 — PyTorch 기반 Transformer 모델 구현.

시퀀스 데이터를 입력받아 XRP 가격을 예측한다.

Requirements: 5.2, 5.3, 5.4
"""

import logging
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from prediction.base_model import BaseModel, PredictionResult

logger = logging.getLogger(__name__)


class _PositionalEncoding(nn.Module):
    """Transformer용 위치 인코딩."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class _TransformerNetwork(nn.Module):
    """PyTorch Transformer 네트워크."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = _PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # 마지막 타임스텝의 출력 사용
        last_out = x[:, -1, :]
        return self.fc(last_out).squeeze(-1)


class TransformerModel(BaseModel):
    """PyTorch 기반 Transformer 예측 모델.

    Parameters
    ----------
    feature_names : list[str] | None
        피처 이름 리스트.
    seq_length : int
        입력 시퀀스 길이 (기본 30).
    d_model : int
        Transformer 모델 차원 (기본 64).
    nhead : int
        어텐션 헤드 수 (기본 4).
    num_layers : int
        Transformer 인코더 레이어 수 (기본 2).
    epochs : int
        학습 에포크 수 (기본 50).
    batch_size : int
        배치 크기 (기본 32).
    learning_rate : float
        학습률 (기본 0.001).
    """

    def __init__(
        self,
        feature_names: list[str] | None = None,
        seq_length: int = 30,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ):
        super().__init__(feature_names)
        self.seq_length = seq_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self._model: _TransformerNetwork | None = None
        self._device = torch.device("cpu")
        self._X_mean: np.ndarray | None = None
        self._X_std: np.ndarray | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._feature_importance: dict[str, float] = {}

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Transformer 모델 학습."""
        if len(X) <= self.seq_length:
            raise ValueError(
                f"데이터 길이({len(X)})가 시퀀스 길이({self.seq_length})보다 커야 합니다."
            )

        if not self.feature_names or len(self.feature_names) != X.shape[1]:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # 정규화
        self._X_mean = X.mean(axis=0)
        self._X_std = X.std(axis=0)
        self._X_std[self._X_std == 0] = 1.0
        X_norm = (X - self._X_mean) / self._X_std

        self._y_mean = float(y.mean())
        self._y_std = float(y.std())
        if self._y_std == 0:
            self._y_std = 1.0
        y_norm = (y - self._y_mean) / self._y_std

        # 시퀀스 생성
        X_seq, y_seq = self._create_sequences(X_norm, y_norm)

        n_features = X.shape[1]
        self._model = _TransformerNetwork(
            input_size=n_features,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
        ).to(self._device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        self._model.train()
        best_loss = float("inf")
        patience_counter = 0
        patience = 5

        # 검증 데이터 분할 (마지막 20%)
        val_split = max(1, int(len(X_seq) * 0.2))
        X_train_t = torch.FloatTensor(X_seq[:-val_split]).to(self._device)
        y_train_t = torch.FloatTensor(y_seq[:-val_split]).to(self._device)
        X_val_t = torch.FloatTensor(X_seq[-val_split:]).to(self._device)
        y_val_t = torch.FloatTensor(y_seq[-val_split:]).to(self._device)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            total_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self._model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # 검증 손실 계산
            self._model.eval()
            with torch.no_grad():
                val_pred = self._model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
            self._model.train()

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(loader)
                logger.debug(
                    "Transformer Epoch %d/%d, Loss: %.6f, Val: %.6f",
                    epoch + 1, self.epochs, avg_loss, val_loss,
                )

            if patience_counter >= patience:
                logger.info("Transformer early stopping at epoch %d (val_loss=%.6f)", epoch + 1, val_loss)
                break

        self._is_trained = True
        self._compute_feature_importance(X_seq)
        logger.info("Transformer 모델 학습 완료: %d 샘플, %d 피처", len(X_seq), n_features)

    def predict(self, X: np.ndarray, timeframe: str) -> PredictionResult:
        """Transformer 모델로 예측 수행."""
        if timeframe not in ("short", "mid", "long"):
            raise ValueError(f"유효하지 않은 timeframe: {timeframe}")
        if not self._is_trained or self._model is None:
            raise RuntimeError("모델이 학습되지 않았습니다.")

        X_norm = (X - self._X_mean) / self._X_std

        if len(X_norm) < self.seq_length:
            pad_len = self.seq_length - len(X_norm)
            padding = np.zeros((pad_len, X_norm.shape[1]))
            X_norm = np.vstack([padding, X_norm])

        X_seq = X_norm[-self.seq_length:].reshape(1, self.seq_length, -1)
        X_tensor = torch.FloatTensor(X_seq).to(self._device)

        self._model.eval()
        with torch.no_grad():
            pred_norm = self._model(X_tensor).item()

        predicted_price = pred_norm * self._y_std + self._y_mean

        # 방향 확률 계산: 현재 종가(close) vs 예측 가격
        close_idx = self.feature_names.index("close") if "close" in self.feature_names else 0
        last_price = float(X[-1, close_idx])
        price_change_ratio = (predicted_price - last_price) / max(abs(last_price), 1e-8)

        up_prob = 1.0 / (1.0 + np.exp(-price_change_ratio * 100))
        up_probability = round(up_prob * 100, 2)
        down_probability = round(100.0 - up_probability, 2)

        confidence = min(100.0, max(0.0, abs(price_change_ratio) * 500))
        confidence = round(confidence, 2)

        return PredictionResult(
            predicted_price=round(predicted_price, 6),
            up_probability=up_probability,
            down_probability=down_probability,
            confidence=confidence,
            timeframe=timeframe,
            feature_importance=self.get_feature_importance(),
        )

    def save(self, path: str) -> None:
        """모델을 파일로 저장."""
        if self._model is None:
            raise RuntimeError("저장할 모델이 없습니다.")

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "model_state": self._model.state_dict(),
            "config": {
                "input_size": self._model.input_projection.in_features,
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_layers": self.num_layers,
                "seq_length": self.seq_length,
            },
            "normalization": {
                "X_mean": self._X_mean,
                "X_std": self._X_std,
                "y_mean": self._y_mean,
                "y_std": self._y_std,
            },
            "feature_names": self.feature_names,
            "feature_importance": self._feature_importance,
        }, path)
        logger.info("Transformer 모델 저장: %s", path)

    def load(self, path: str) -> None:
        """저장된 모델 로드."""
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)

        config = checkpoint["config"]
        self.d_model = config["d_model"]
        self.nhead = config["nhead"]
        self.num_layers = config["num_layers"]
        self.seq_length = config["seq_length"]

        self._model = _TransformerNetwork(
            input_size=config["input_size"],
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
        ).to(self._device)
        self._model.load_state_dict(checkpoint["model_state"])

        norm = checkpoint["normalization"]
        self._X_mean = norm["X_mean"]
        self._X_std = norm["X_std"]
        self._y_mean = norm["y_mean"]
        self._y_std = norm["y_std"]

        self.feature_names = checkpoint.get("feature_names", [])
        self._feature_importance = checkpoint.get("feature_importance", {})
        self._is_trained = True
        logger.info("Transformer 모델 로드: %s", path)

    def get_feature_importance(self) -> dict[str, float]:
        """피처 기여도 반환 (gradient 기반). 합 ≈ 1.0."""
        if self._feature_importance:
            return dict(self._feature_importance)
        return super().get_feature_importance()

    # ── 내부 메서드 ──────────────────────────────────────────

    def _create_sequences(
        self, X: np.ndarray, y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """시계열 데이터를 시퀀스로 변환."""
        X_seq, y_seq = [], []
        for i in range(len(X) - self.seq_length):
            X_seq.append(X[i: i + self.seq_length])
            y_seq.append(y[i + self.seq_length])
        return np.array(X_seq), np.array(y_seq)

    def _compute_feature_importance(self, X_seq: np.ndarray) -> None:
        """Gradient 기반 피처 기여도 계산."""
        if self._model is None or len(X_seq) == 0:
            return

        self._model.eval()
        n_samples = min(100, len(X_seq))
        X_sample = torch.FloatTensor(X_seq[:n_samples]).to(self._device)
        X_sample.requires_grad_(True)

        pred = self._model(X_sample)
        pred.sum().backward()

        if X_sample.grad is not None:
            importance = X_sample.grad.abs().mean(dim=(0, 1)).cpu().numpy()
            total = importance.sum()
            if total > 0:
                importance = importance / total

            self._feature_importance = {}
            for i, name in enumerate(self.feature_names):
                if i < len(importance):
                    self._feature_importance[name] = float(importance[i])
